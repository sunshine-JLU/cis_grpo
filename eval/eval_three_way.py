"""
Three-way evaluation: Base vs Baseline GRPO vs CIS-GRPO on InternVL3.5-2B.
Uses vLLM for inference, tokenizer for chat template formatting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
from mathruler.grader import extract_boxed_content, grade_answer
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

_FORMAT_RE = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)


def eval_one(solution_str: str, ground_truth: str) -> dict[str, float]:
    pred = extract_boxed_content(solution_str or "")
    gt = extract_boxed_content(ground_truth) or ground_truth
    if gt.startswith("\\boxed{"):
        gt = gt[len("\\boxed{"):-1] if gt.endswith("}") else gt[len("\\boxed{"):]
    acc = 1.0 if grade_answer(pred, gt) else 0.0
    fmt = 1.0 if _FORMAT_RE.fullmatch(solution_str or "") else 0.0
    return {"acc": acc, "format": fmt}


def prepare_prompts(df, tokenizer):
    """Convert ViRL39K dataframe rows to vLLM prompt dicts."""
    prompts = []
    failed = 0
    for _, row in df.iterrows():
        imgs = row["images"]
        if imgs is not None and len(imgs) > 0:
            img_path = imgs[0]["image"]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    msgs = row["prompt"]
                    if not isinstance(msgs, list):
                        msgs = [{"role": "user", "content": str(msgs)}]
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    prompts.append({
                        "prompt": text,
                        "multi_modal_data": {"image": img},
                    })
                    continue
                except Exception as e:
                    print(f"  WARN: image {img_path}: {e}", flush=True)
        failed += 1
        prompts.append({"prompt": "failed", "multi_modal_data": {}})
    return prompts, failed


def run_eval(model_path: str, val_parquet: str, output_path: str, tp: int = 2, gpu_mem: float = 0.5):
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    df = pd.read_parquet(val_parquet)
    print(f"Loaded {len(df)} val examples", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts, n_failed = prepare_prompts(df, tokenizer)
    print(f"Prepared {len(prompts)} prompts ({n_failed} failed)", flush=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=6144,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("vLLM ready, running inference...", flush=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({len(outputs)/elapsed:.1f} samples/s)", flush=True)

    ground_truths = [row["reward_model"]["ground_truth"] for _, row in df.iterrows()]
    results = []
    accs, fmts = [], []

    for i, out in enumerate(outputs):
        resp = out.outputs[0].text
        r = eval_one(resp, ground_truths[i])
        results.append({"response": resp, "gt": ground_truths[i], **r})
        accs.append(r["acc"])
        fmts.append(r["format"])

    mean_acc = sum(accs) / len(accs) if accs else 0.0
    mean_fmt = sum(fmts) / len(fmts) if fmts else 0.0

    out = {
        "model_path": model_path,
        "val_parquet": val_parquet,
        "n_examples": len(df),
        "n_failed_images": n_failed,
        "inference_time_s": elapsed,
        "acc/mean@1": mean_acc,
        "format/mean@1": mean_fmt,
        "acc_correct": int(sum(accs)),
        "acc_total": len(accs),
        "format_correct": int(sum(fmts)),
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  acc/mean@1   = {mean_acc:.4f}  ({int(sum(accs))}/{len(accs)})", flush=True)
    print(f"  format/mean@1 = {mean_fmt:.4f}  ({int(sum(fmts))}/{len(fmts)})", flush=True)
    print(f"Saved to {output_path}", flush=True)

    # Free GPU memory
    del llm
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    time.sleep(3)

    return out


def main():
    base_model = "/root/autodl-tmp/InternVL3_5-2B"
    grpo_ckpt = "/root/autodl-tmp/checkpoints/cis_grpo/baseline_grpo_internvl3_5_2b_20260603_0815/global_step_200/actor/huggingface"
    cis_ckpt = "/root/autodl-tmp/checkpoints/cis_grpo/cis_grpo_internvl3_5_2b_v4fmtonly_20260603_2052/global_step_200/actor/huggingface"
    val_parquet = "/root/autodl-tmp/data/virl39k/val_cis_ready.parquet"

    results = {}
    for name, path in [("base", base_model), ("grpo", grpo_ckpt), ("cis_grpo", cis_ckpt)]:
        out_path = f"/root/autodl-tmp/eval_{name}_result.json"
        r = run_eval(path, val_parquet, out_path)
        results[name] = r

    print(f"\n{'='*60}")
    print("THREE-WAY COMPARISON")
    print(f"{'='*60}")
    for name in ["base", "grpo", "cis_grpo"]:
        r = results[name]
        print(f"  {name:10s}  acc@1={r['acc/mean@1']:.4f}  format@1={r['format/mean@1']:.4f}  ({r['acc_correct']}/{r['acc_total']})")

    with open("/root/autodl-tmp/eval_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull comparison saved to /root/autodl-tmp/eval_comparison.json")


if __name__ == "__main__":
    main()
