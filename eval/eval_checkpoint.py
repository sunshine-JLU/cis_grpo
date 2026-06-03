"""
Evaluate a merged HF checkpoint on val_100_cis_ready.parquet.
Computes mean@1 accuracy WITHOUT CIS doubling — each prompt is a single
standard inference pass, unlike CIS-GRPO which doubles into (real, CF) pairs.

Usage:
  python recipes/cis_grpo/eval_checkpoint.py \
    --model_path /path/to/merged_hf_model \
    --val_parquet /root/.../val_100_cis_ready.parquet \
    --output /tmp/eval_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import pandas as pd
import torch
from mathruler.grader import extract_boxed_content, grade_answer
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_parquet", required=True)
    p.add_argument("--output", default="/tmp/cis_eval_result.json")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def load_model(model_path: str):
    print(f"Loading model from {model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def generate_batch(model, processor, prompts, images, max_new_tokens):
    """Generate responses for a batch of (prompt, image) pairs."""
    batch_messages = []
    batch_images = []
    for prompt, imgs in zip(prompts, images):
        msgs = list(prompt)  # copy
        if imgs is not None and len(imgs) > 0:
            img_path = imgs[0]["image"]
            if os.path.exists(img_path):
                batch_images.append(Image.open(img_path).convert("RGB"))
            else:
                batch_images.append(None)
        else:
            batch_images.append(None)
        batch_messages.append(msgs)

    # Filtered batch: only examples with valid images
    valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
    valid_messages = [batch_messages[i] for i in valid_indices]
    valid_images = [batch_images[i] for i in valid_indices]

    if not valid_messages:
        return [""] * len(prompts)

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in valid_messages
    ]

    inputs = processor(
        text=texts,
        images=valid_images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    # Decode only the new tokens
    input_lens = inputs.input_ids.shape[1]
    responses = []
    for i, idx in enumerate(valid_indices):
        gen_ids = outputs[i][input_lens:]
        resp = processor.decode(gen_ids, skip_special_tokens=True)
        responses.append((idx, resp))

    # Fill in blanks for images that failed to load
    full_responses = [""] * len(prompts)
    for idx, resp in responses:
        full_responses[idx] = resp
    return full_responses


_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def eval_one(solution_str: str, ground_truth: str) -> dict[str, float]:
    pred = extract_boxed_content(solution_str or "")
    gt = extract_boxed_content(ground_truth) or ground_truth
    gt = gt.replace("\\boxed{", "").replace("}", "")
    acc = 1.0 if grade_answer(pred, gt) else 0.0
    fmt = 1.0 if _FORMAT_RE.fullmatch(solution_str or "") else 0.0
    return {"acc": acc, "format": fmt}


def main():
    args = parse_args()
    df = pd.read_parquet(args.val_parquet)
    print(f"Loaded {len(df)} val examples", flush=True)

    model, processor = load_model(args.model_path)

    prompts = df["prompt"].tolist()
    images = df["images"].tolist()
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in df.iterrows()]

    results = []
    accs = []
    fmts = []

    batch_size = args.batch_size
    for start in tqdm(range(0, len(prompts), batch_size), desc="Eval"):
        end = min(start + batch_size, len(prompts))
        batch_prompts = [p[0] if isinstance(p, list) and len(p) == 1 else p for p in prompts[start:end]]
        batch_imgs = images[start:end]
        batch_gts = ground_truths[start:end]

        # Process one at a time (Qwen2-VL batch gen can be finicky)
        for i in range(len(batch_prompts)):
            p = prompts[start + i]
            imgs_list = images[start + i]
            gt_str = ground_truths[start + i]

            if isinstance(p, list) and len(p) > 0:
                msgs = list(p)
            else:
                msgs = [{"role": "user", "content": str(p)}]

            img = None
            if imgs_list is not None and len(imgs_list) > 0:
                img_path = imgs_list[0]["image"]
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")

            if img is None:
                results.append({"acc": 0.0, "format": 0.0, "error": "no_image"})
                accs.append(0.0)
                fmts.append(0.0)
                continue

            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

            input_len = inputs.input_ids.shape[1]
            gen_ids = outputs[0][input_len:]
            resp = processor.decode(gen_ids, skip_special_tokens=True)

            r = eval_one(resp, gt_str)
            results.append(r)
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
        "acc_sum": sum(accs),
        "format_sum": sum(fmts),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nRESULTS:", flush=True)
    print(f"  acc/mean@1   = {mean_acc:.4f}  ({int(sum(accs))}/{len(accs)})", flush=True)
    print(f"  format/mean@1 = {mean_fmt:.4f}", flush=True)
    print(f"Saved to {args.output}", flush=True)
    return out


if __name__ == "__main__":
    main()
