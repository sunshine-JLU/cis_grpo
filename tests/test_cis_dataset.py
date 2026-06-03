"""Smoke test for CISGrpoDataset on a 32-row toy parquet."""

import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
for p in (_PROJECT, os.path.join(_PROJECT, "verl")):
    if p not in sys.path:
        sys.path.insert(0, p)

from recipes.cis_grpo.cis_dataset import CISGrpoDataset  # noqa: E402


def main():
    proj = os.environ.get("PROJECT_ROOT", _PROJECT)
    src = f"{proj}/data/virl39k/train.parquet"
    print(f"Loading 32 rows from {src} ...")
    df = pq.read_table(src).to_pandas().head(32)

    tmp = tempfile.mkdtemp(prefix="cis_test_")
    try:
        toy = f"{tmp}/toy.parquet"
        pq.write_table(pa.Table.from_pandas(df), toy)

        tok_path = f"{proj}/Qwen2.5-VL-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        processor = AutoProcessor.from_pretrained(tok_path)

        cfg = DictConfig(
            {
                "prompt_key": "prompt",
                "image_key": "images",
                "video_key": "videos",
                "max_prompt_length": 4096,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "cache_dir": f"{tmp}/cache",
                "cis": {"swap_offset": 0, "swap_seed": 7, "shuffle_swap_each_epoch": True},
            }
        )
        ds = CISGrpoDataset(toy, tokenizer, cfg, processor)
        print(f"len(ds) = {len(ds)}  (expected 64)")
        assert len(ds) == 64, "expected 32 real + 32 cf = 64 rows"

        real = ds[0]
        cf = ds[1]

        # Check uid shared between real and cf of the same logical prompt.
        assert real["uid"] == cf["uid"], f"uid mismatch: {real['uid']} vs {cf['uid']}"
        assert real["extra_info"]["is_cf"] is False
        assert cf["extra_info"]["is_cf"] is True
        assert cf["extra_info"]["cf_source_index"] != 0, "CF must borrow an image from a different row"

        # Check uids differ across logical prompts.
        u2_real = ds[2]["uid"]
        assert u2_real != real["uid"], "different logical prompts must have different uids"

        # Spot-check that real and CF have different image content in raw_prompt.
        def get_image_payload(row):
            for msg in row["raw_prompt"]:
                content = msg["content"]
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image":
                            return part.get("image")
            return None

        real_img = get_image_payload(real)
        cf_img = get_image_payload(cf)
        assert real_img is not None and cf_img is not None, "both rows must carry an image"
        assert str(real_img) != str(cf_img), f"real and CF images should differ: {real_img} vs {cf_img}"

        # Verify ground truth is preserved on CF row (we swap image, not GT).
        assert real["reward_model"]["ground_truth"] == cf["reward_model"]["ground_truth"]

        # Verify GT-different constraint: gts[swap_map[i]] != gts[i] for all i
        # (except for any identity-swap fallbacks injected when no budget-feasible
        # partner exists — these are rare on real data but possible on toy inputs).
        gts = ds._gts
        identity_fallbacks = 0
        for i in range(len(ds._swap_map)):
            j = int(ds._swap_map[i])
            if j == i:
                identity_fallbacks += 1
                continue
            assert gts[j] != gts[i], f"GT collision at i={i}: gt={gts[i]!r}"
        print(f"  swap map: {identity_fallbacks} identity fallback(s)")

        # Verify budget constraint when columns exist (post-prepare_cis_ready.py).
        # If the toy parquet didn't carry them, the dataset falls back to "no
        # budget guard" — accept that for the smoke test.
        if any(ds._image_tokens) and any(ds._text_tokens):
            txt = ds._text_tokens
            img = ds._image_tokens
            budget = ds._max_prompt_length
            for i in range(len(ds._swap_map)):
                j = int(ds._swap_map[i])
                if i == j:
                    continue
                assert txt[i] + img[j] <= budget, (
                    f"budget violation at i={i}: txt[i]={txt[i]} + img[j={j}]={img[j]} > {budget}"
                )
            print("  swap-map budget constraint OK")

        # Verify single-image filter: every retained row must have exactly 1 image.
        image_key = ds.image_key
        for i in range(len(ds.dataframe)):
            imgs = ds.dataframe[i].get(image_key)
            assert imgs is not None, f"row {i} has no images field after filter"
            n_img = len(imgs) if isinstance(imgs, (list, tuple)) else 1
            assert n_img == 1, f"row {i} has {n_img} images after single-image filter"

        # Verify epoch reshuffling changes the swap map.
        old_swap = list(ds._swap_map)
        ds.set_epoch(1)
        new_swap = list(ds._swap_map)
        assert old_swap != new_swap, "swap map should reshuffle between epochs"
        # And the new map still satisfies the constraints (allowing for some
        # identity fallbacks if budget makes a row infeasible).
        for i in range(len(ds._swap_map)):
            j = int(ds._swap_map[i])
            if i == j:
                continue
            assert gts[j] != gts[i], f"epoch=1 GT violation at i={i}"

        # Module path must be importable so HF datasets num_proc workers can
        # unpickle the class. After our pkg:// fix it should be a real dotted name.
        mod = type(ds).__module__
        print(f"  CISGrpoDataset.__module__ = {mod!r}")
        # When loaded via the test (direct import) it's recipes.cis_grpo.cis_dataset.
        # When loaded by verl via pkg:// it's also recipes.cis_grpo.cis_dataset.
        # If it ever starts with 'custom_module_' the multiproc filter will break.
        assert not mod.startswith("custom_module_"), (
            f"module name {mod!r} is unimportable in HF datasets workers — "
            "use data.custom_cls.path=pkg://... not the .py file form"
        )

        print("CISGrpoDataset self-test: ALL ASSERTIONS PASSED")
        print(f"  real uid={real['uid']}  cf source idx={cf['extra_info']['cf_source_index']}")
        print(f"  real img={real_img}")
        print(f"  cf   img={cf_img}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
