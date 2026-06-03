"""
Evaluate merged HF checkpoint on val_100_cis_ready.parquet using vLLM.
No CIS doubling — single inference per prompt, mean@1 metric.
"""

from __future__ import annotations

import argparse, json, os, re, sys

import pandas as pd
from mathruler.grader import extract_boxed_content, grade_answer
from PIL import Image
from tqdm import tqdm

_FORMAT_RE = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)


def eval_one(solution_str: str, ground_truth: str) -> dict[str, float]:
    pred = extract_boxed_content(solution_str or "")
    gt = extract_boxed_content(ground_truth) or ground_truth
    if gt.startswith("\\boxed{"):
        gt = gt[len("\\boxed{"):-1] if gt.endswith("}") else gt[len("\\boxed{"):]
    acc = 1.0 if grade_answer(pred, gt) else 0.0
    fmt = 1.0 if _FORMAT_RE.fullmatch(solution_str or "") else 0.0
    return {"acc": acc, "format": fmt}


def convert_prompt(prompt_msgs, image_path: str, processor):
    """Convert ViRL39K prompt (with <image> text) to vision-token format."""
    new_msgs = []
    for msg in prompt_msgs:
        content = msg["content"]
        if isinstance(content, str) and "<image>" in content:
            # Replace <image> with structured image reference
            img = Image.open(image_path).convert("RGB")
            parts = content.split("<image>", 1)
            new_content = []
            if parts[0].strip():
                new_content.append({"type": "text", "text": parts[0]})
            new_content.append({"type": "image", "image": img})
            if len(parts) > 1 and parts[1].strip():
                new_content.append({"type": "text", "text": parts[1]})
            new_msgs.append({"role": msg["role"], "content": new_content})
        elif isinstance(content, str):
            new_msgs.append(msg)
        else:
            new_msgs.append(msg)

    text = processor.apply_chat_template(new_msgs, tokenize=False, add_generation_prompt=True)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_parquet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--max_tokens", type=int, default=1024)
    args = p.parse_args()

    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    df = pd.read_parquet(args.val_parquet)
    print(f"Loaded {len(df)} val examples", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens
    )

    # Build prompts with vision tokens
    prompts = []
    failed = 0
    for _, row in df.iterrows():
        img_info = row["images"]
        if img_info is not None and len(img_info) > 0:
            img_path = img_info[0]["image"]
            if os.path.exists(img_path):
                try:
                    text = convert_prompt(row["prompt"], img_path, processor)
                    img = Image.open(img_path).convert("RGB")
                    prompts.append({
                        "prompt": text,
                        "multi_modal_data": {"image": img},
                    })
                    continue
                except Exception as e:
                    print(f"  WARN: failed to load image {img_path}: {e}", flush=True)

        failed += 1
        prompts.append({"prompt": "failed", "multi_modal_data": {}})

    print(f"Prepared {len(prompts)} prompts ({failed} failed image loads)", flush=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("vLLM ready, running inference...", flush=True)

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    ground_truths = [row["reward_model"]["ground_truth"] for _, row in df.iterrows()]
    accs, fmts = [], []

    for i, out in enumerate(outputs):
        resp = out.outputs[0].text
        r = eval_one(resp, ground_truths[i])
        accs.append(r["acc"])
        fmts.append(r["format"])

    mean_acc = sum(accs) / len(accs) if accs else 0.0
    mean_fmt = sum(fmts) / len(fmts) if fmts else 0.0

    out = {
        "model_path": args.model_path,
        "val_parquet": args.val_parquet,
        "n_examples": len(df),
        "acc/mean@1": mean_acc,
        "format/mean@1": mean_fmt,
        "acc_correct": int(sum(accs)),
        "acc_total": len(accs),
        "format_correct": int(sum(fmts)),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nRESULTS:", flush=True)
    print(f"  acc/mean@1   = {mean_acc:.4f}  ({int(sum(accs))}/{len(accs)})", flush=True)
    print(f"  format/mean@1 = {mean_fmt:.4f}  ({int(sum(fmts))}/{len(fmts)})", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
