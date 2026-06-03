"""
Preprocess ViRL39K (TIGER-Lab) into verl-compatible parquet format.

Source schema: question, answer, PassRate_32BTrained, PassRate_7BBase, category,
source, qid, image (list of relative paths like "images/Processed-XXX-0.jpg").

Output schema (matches verl/examples/data_preprocess/geo3k.py):
    data_source, prompt (chat), images (list of dicts), ability, reward_model, extra_info

We keep `PassRate_*` and `category` in extra_info so downstream filters and
the CIS-GRPO swap-source policy can use them without re-loading the raw file.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed "
    "within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
)


def build_row(example, idx, split, img_dir, source_name):
    question = example["question"]
    answer = example["answer"]
    image_rels = example["image"]

    image_dicts = []
    for rel in image_rels:
        abs_path = (img_dir / rel).resolve()
        if not abs_path.is_file():
            return None
        image_dicts.append({"image": str(abs_path)})
    if not image_dicts:
        return None

    prompt = question.strip() + " " + INSTRUCTION
    gt = answer

    return {
        "data_source": source_name,
        "prompt": [{"role": "user", "content": prompt}],
        "images": image_dicts,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": gt,
            "question": question,
            "qid": example["qid"],
            "category": example["category"],
            "source": example["source"],
            "pass_rate_7b": float(example["PassRate_7BBase"]),
            "pass_rate_32b": float(example["PassRate_32BTrained"]),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to ViRL39K parquet.")
    p.add_argument("--img_dir", required=True, help="Dir containing the 'images/' subdir.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_pass_rate_7b", type=float, default=-1.0,
        help="Drop rows whose 7B base pass rate is below this. Use -1 to keep all (-1 entries are 'unannotated', kept).",
    )
    p.add_argument(
        "--max_pass_rate_7b", type=float, default=2.0,
        help="Drop rows whose 7B base pass rate is above this (typical: 0.875 to remove trivial samples).",
    )
    p.add_argument("--source_name", default="ViRL39K")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    print(f"Loading {args.input} ...", flush=True)
    table = pq.read_table(args.input)
    df = table.to_pandas()
    print(f"  rows: {len(df)}", flush=True)

    if args.min_pass_rate_7b > -1.0 or args.max_pass_rate_7b < 2.0:
        before = len(df)
        keep = (df["PassRate_7BBase"] == -1.0) | (
            (df["PassRate_7BBase"] >= args.min_pass_rate_7b)
            & (df["PassRate_7BBase"] <= args.max_pass_rate_7b)
        )
        df = df[keep].reset_index(drop=True)
        print(f"  pass-rate filter: {before} -> {len(df)}", flush=True)

    random.seed(args.seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    val_idx = set(indices[: args.val_size])

    train_rows, val_rows = [], []
    n_dropped = 0
    for i, ex in enumerate(df.to_dict(orient="records")):
        split = "val" if i in val_idx else "train"
        row = build_row(ex, i, split, img_dir, args.source_name)
        if row is None:
            n_dropped += 1
            continue
        (val_rows if split == "val" else train_rows).append(row)
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{len(df)}  (dropped {n_dropped})", flush=True)

    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}  Dropped (missing images): {n_dropped}", flush=True)

    def write(rows, name):
        if not rows:
            print(f"  WARN: no rows for {name}", flush=True)
            return
        t = pa.Table.from_pylist(rows)
        path = out_dir / name
        pq.write_table(t, path)
        print(f"  wrote {path}  ({len(rows)} rows, {path.stat().st_size/1e6:.1f} MB)", flush=True)

    write(train_rows, "train.parquet")
    write(val_rows, "val.parquet")

    summary = {
        "source": args.input,
        "img_dir": str(img_dir),
        "total_input_rows": int(len(df)),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_dropped_missing_image": int(n_dropped),
        "pass_rate_filter": {
            "min_7b": args.min_pass_rate_7b,
            "max_7b": args.max_pass_rate_7b,
        },
        "seed": args.seed,
    }
    with open(out_dir / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
