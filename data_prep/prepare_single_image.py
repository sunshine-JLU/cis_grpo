"""One-shot preprocessor: keep rows usable by CIS-GRPO without runtime asserts.

Two filters are applied, both required for `_build_messages` to succeed:
  1. `len(images) == 1` — CIS swap replaces the images field; differing counts
     between source and target row would mismatch the prompt's `<image>`
     placeholder count, failing the `image_offset != len(images)` assert.
  2. Exactly one `<image>` placeholder across the prompt's message contents.
     ViRL39K has ~2% rows where the placeholder was lost during conversion
     but the image stayed; those silently fail the same assert (offset 0,
     len 1) at filter time and spam stderr from inside HF datasets multiproc
     workers, dragging length-filter throughput down 5-10x.

Outputs `<stem>_single_image.parquet` next to each input parquet.

Usage:
    python recipes/cis_grpo/prepare_single_image.py \
        [--data-dir data/virl39k] [--overwrite]
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PAIRS = [
    ("train.parquet", "train_single_image.parquet"),
    ("val.parquet", "val_single_image.parquet"),
    ("val_100.parquet", "val_100_single_image.parquet"),
]


def _count_image_placeholders(prompt) -> int:
    if prompt is None:
        return 0
    total = 0
    for msg in prompt:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += content.count("<image>")
    return total


def filter_one(src: str, dst: str, overwrite: bool) -> None:
    if not os.path.exists(src):
        print(f"[skip] missing input: {src}")
        return
    if os.path.exists(dst) and not overwrite:
        n = pq.read_metadata(dst).num_rows
        print(f"[exists] {dst} ({n} rows) — pass --overwrite to rebuild")
        return
    df = pq.read_table(src).to_pandas()
    n_before = len(df)
    img_ok = df["images"].map(lambda x: x is not None and len(x) == 1)
    ph_ok = df["prompt"].map(lambda p: _count_image_placeholders(p) == 1)
    mask = img_ok & ph_ok
    df1 = df[mask].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(df1, preserve_index=False), dst)
    pct = 100.0 * len(df1) / max(n_before, 1)
    drop_img = int((~img_ok).sum())
    drop_ph = int((img_ok & ~ph_ok).sum())
    print(
        f"[wrote] {dst}: {n_before} -> {len(df1)} ({pct:.2f}% kept)  "
        f"dropped: {drop_img} multi/zero-image, {drop_ph} placeholder-mismatch"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="/root/autodl-tmp/data/virl39k",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for src_name, dst_name in DEFAULT_PAIRS:
        filter_one(
            os.path.join(args.data_dir, src_name),
            os.path.join(args.data_dir, dst_name),
            args.overwrite,
        )


if __name__ == "__main__":
    main()
