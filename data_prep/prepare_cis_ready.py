"""Vision-aware length filter + per-row budget stats, run once, saved to disk.

Reads `<stem>_single_image.parquet`, applies the same multimodal prompt-length
filter that verl's `RLHFDataset.maybe_filter_out_long_prompts` would apply at
runtime, drops rows whose tokenized prompt exceeds max_prompt_length, AND
attaches per-row token-budget stats so the CIS swap map can avoid overflowing
the model's context after an image swap:

  * `cis_total_tokens`  — prompt + image tokens (the row's runtime budget use)
  * `cis_text_tokens`   — chat-templated prompt tokens without any image
  * `cis_image_tokens`  — image expansion contribution = total - text

The constraint the swap-map constructor enforces at runtime is
`text[i] + image[σ(i)] ≤ max_prompt_length`. Without these columns it has no
way to know whether a candidate σ(i) fits the borrowing row's budget — which
is exactly the failure mode the smoke v5 hit (CF row with 2606-tok prompt
overflowed max_model_len=2048).

Why offline? The filter is slow on Qwen2.5-VL (full processor() per row, which
decodes the image and expands visual tokens). Running it every launch wastes
CPU and provides no caching across machines. Doing it once yields a parquet
that the training loop can load instantly with
`data.filter_overlong_prompts=False`.

Why ProcessPoolExecutor instead of `datasets.map`? `datasets.map` rewrites
the entire Arrow table per shard — for ViRL39K that means re-serializing
~11 MB × image bytes 35k × 16 shards on every checkpoint, ~80× slowdown vs.
the pure-filter path. We compute stats out-of-band and patch the columns onto
the original Arrow table at the end.

Default max_prompt_length matches the training config (1024).

Usage:
    python recipes/cis_grpo/prepare_cis_ready.py \
        [--data-dir data/virl39k] [--max-prompt-length 1024] \
        [--num-workers 16] [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback

# Critical: must set BEFORE numpy / transformers import. With 16 worker
# processes each defaulting to OMP/MKL threading on a 48-core box, you get
# 16 * 24 = 384 contending threads and per-row throughput collapses to <3 ex/s.
# Pinning to 1 thread per worker brings throughput back to ~10× higher.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_PROJECT, os.path.join(_PROJECT, "verl")):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_PAIRS = [
    ("train_single_image.parquet", "train_cis_ready.parquet"),
    ("val_single_image.parquet", "val_cis_ready.parquet"),
    ("val_100_single_image.parquet", "val_100_cis_ready.parquet"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# Globals shared via fork() COW into workers — set in parent before Pool start.
_PARENT_PROMPTS: list | None = None
_PARENT_IMAGES: list | None = None
_PARENT_MAX_LEN: int = 1024

# Globals initialized per worker so we don't pay AutoProcessor cost per row.
_WORKER_PROCESSOR = None
_WORKER_TOKENIZER = None
_WORKER_PROCESS_IMAGE = None


def _worker_init(model_path: str) -> None:
    """Per-worker init: build processor/tokenizer/process_image once.

    Inherits _PARENT_PROMPTS / _PARENT_IMAGES / _PARENT_MAX_LEN via fork() COW.
    Each worker has its own AutoProcessor instance (HF processors are not
    fork-safe to share between workers — Rust tokenizer locks).
    """
    global _WORKER_PROCESSOR, _WORKER_TOKENIZER, _WORKER_PROCESS_IMAGE
    from transformers import AutoProcessor, AutoTokenizer
    from verl.utils.dataset.vision_utils import process_image
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    _WORKER_PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _WORKER_PROCESS_IMAGE = process_image


def _build_messages_inline(prompt) -> list:
    return [dict(m) for m in prompt]


def _row_stats(prompt, images_raw, max_len: int) -> tuple[int, int, int]:
    """Return (total_tokens, text_only_tokens, image_only_tokens) for one row.

    On failure returns (max_len + 1, 0, 0) so the row is dropped by the filter.
    """
    import re
    proc = _WORKER_PROCESSOR
    try:
        messages = _build_messages_inline(prompt)
        images = [_WORKER_PROCESS_IMAGE(img) for img in images_raw] if images_raw else None
        offset = 0
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            parts: list = []
            for seg in re.split("(<image>)", content):
                if seg == "<image>":
                    parts.append({"type": "image"})
                    offset += 1
                elif seg:
                    parts.append({"type": "text", "text": seg})
            m["content"] = parts
        if images is not None and offset != len(images):
            return max_len + 1, 0, 0

        raw_prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text_only = len(proc.tokenizer(text=raw_prompt, add_special_tokens=False)["input_ids"])
        if images:
            total = len(proc(text=[raw_prompt], images=images)["input_ids"][0])
        else:
            total = text_only
        image_only = max(0, total - text_only)
        return total, text_only, image_only
    except Exception:
        traceback.print_exc()
        return max_len + 1, 0, 0


def _process_one(i: int) -> tuple[int, int, int, int]:
    """Worker entrypoint for one row index — reads from parent COW globals."""
    t, x, m = _row_stats(_PARENT_PROMPTS[i], _PARENT_IMAGES[i], _PARENT_MAX_LEN)
    return i, t, x, m


def filter_one(src: str, dst: str, model_path: str, max_len: int, num_workers: int, overwrite: bool) -> None:
    if not os.path.exists(src):
        log.info("[skip] missing input %s", src)
        return
    if os.path.exists(dst) and not overwrite:
        n = pq.read_metadata(dst).num_rows
        log.info("[exists] %s (%d rows) — pass --overwrite to rebuild", dst, n)
        return

    log.info("loading %s", src)
    t_load = time.time()
    src_tbl = pq.read_table(src)
    n_before = src_tbl.num_rows
    log.info("  read parquet in %.1fs  rows=%d", time.time() - t_load, n_before)

    # Populate parent globals so forked workers inherit prompts/images COW-style
    # — no per-worker reload of the (potentially huge) image column.
    global _PARENT_PROMPTS, _PARENT_IMAGES, _PARENT_MAX_LEN
    _PARENT_PROMPTS = src_tbl.column("prompt").to_pylist()
    _PARENT_IMAGES = src_tbl.column("images").to_pylist()
    _PARENT_MAX_LEN = max_len
    log.info("  materialized prompts/images for COW share with workers (num_workers=%d)", num_workers)

    totals = np.full(n_before, max_len + 1, dtype=np.int32)
    texts = np.zeros(n_before, dtype=np.int32)
    imgs = np.zeros(n_before, dtype=np.int32)

    t0 = time.time()
    last_logged = 0
    completed_rows = 0
    ctx = mp.get_context("fork")
    chunksize = max(1, n_before // (num_workers * 32))
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(model_path,)) as pool:
        for i, t, x, m in pool.imap_unordered(_process_one, range(n_before), chunksize=chunksize):
            totals[i] = t
            texts[i] = x
            imgs[i] = m
            completed_rows += 1
            now = time.time()
            if now - last_logged >= 5.0:
                pct = 100.0 * completed_rows / n_before
                rate = completed_rows / max(now - t0, 1e-6)
                eta = (n_before - completed_rows) / max(rate, 1e-6)
                log.info("  progress: %d/%d (%.1f%%)  rate=%.1f ex/s  eta=%.0fs", completed_rows, n_before, pct, rate, eta)
                last_logged = now
    elapsed = time.time() - t0
    log.info("  done: %d rows in %.1fs (%.1f ex/s)", n_before, elapsed, n_before / max(elapsed, 1e-6))

    keep = totals <= max_len
    n_after = int(keep.sum())
    log.info("  %d -> %d (%.2f%% kept) in %.1fs", n_before, n_after, 100 * n_after / max(n_before, 1), elapsed)
    log.info(
        "  image_tokens (kept): min=%d  median=%d  max=%d",
        int(imgs[keep].min()) if n_after else 0,
        int(np.median(imgs[keep])) if n_after else 0,
        int(imgs[keep].max()) if n_after else 0,
    )
    log.info(
        "  text_tokens  (kept): min=%d  median=%d  max=%d",
        int(texts[keep].min()) if n_after else 0,
        int(np.median(texts[keep])) if n_after else 0,
        int(texts[keep].max()) if n_after else 0,
    )

    # Filter the in-memory src_tbl, append new int32 stat columns, then write.
    mask = pa.array(keep)
    full = src_tbl.filter(mask)
    full = full.append_column("cis_total_tokens", pa.array(totals[keep], type=pa.int32()))
    full = full.append_column("cis_text_tokens", pa.array(texts[keep], type=pa.int32()))
    full = full.append_column("cis_image_tokens", pa.array(imgs[keep], type=pa.int32()))

    log.info("writing %s", dst)
    pq.write_table(full, dst)
    log.info("  wrote %d rows to %s", n_after, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(_PROJECT, "data/virl39k"))
    p.add_argument("--model-path", default=os.path.join(_PROJECT, "Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for src_name, dst_name in DEFAULT_PAIRS:
        filter_one(
            os.path.join(args.data_dir, src_name),
            os.path.join(args.data_dir, dst_name),
            model_path=args.model_path,
            max_len=args.max_prompt_length,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
