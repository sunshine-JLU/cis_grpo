"""Merge FSDP2 checkpoint shards into a single HuggingFace model.

Usage:
  python merge_fsdp_checkpoint.py \
    --input_dir /path/to/global_step_N/actor \
    --output_dir /path/to/merged_model \
    --base_model /root/autodl-tmp/InternVL3_5-2B
"""

import argparse
import os
import shutil
import sys

import torch
import torch.distributed.tensor  # needed to load DTensors  # noqa: F401


def merge_shards(input_dir: str) -> dict[str, torch.Tensor]:
    rank_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith("model_world_size_") and f.endswith(".pt")]
    )
    if not rank_files:
        raise FileNotFoundError(f"No model_world_size_*.pt files found in {input_dir}")

    print(f"Loading {len(rank_files)} shards...")
    all_shards = {}
    for rf in rank_files:
        path = os.path.join(input_dir, rf)
        print(f"  {rf} ...")
        d = torch.load(path, map_location="cpu", weights_only=False)
        all_shards[rf] = d

    first_shard = all_shards[rank_files[0]]
    merged = {}

    for key in first_shard:
        tensors = []
        for rf in rank_files:
            t = all_shards[rf][key]
            if hasattr(t, "_local_tensor"):
                tensors.append(t._local_tensor)
            else:
                tensors.append(t)

        # Check if all local tensors are the same shape as full → replicated parameter
        full_shape = first_shard[key].shape
        if len(tensors) == 1 or all(t.shape == full_shape for t in tensors):
            merged[key] = tensors[0].clone()
        else:
            # Sharded — concatenate along shard dim
            placements = first_shard[key].placements
            shard_dim = None
            for p in placements:
                if hasattr(p, "dim") and p.dim >= 0:
                    shard_dim = p.dim
                    break
            if shard_dim is not None:
                merged[key] = torch.cat(tensors, dim=shard_dim)
            else:
                merged[key] = tensors[0].clone()

    print(f"Merged {len(merged)} parameters")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="/root/autodl-tmp/InternVL3_5-2B")
    args = p.parse_args()

    merged_sd = merge_shards(args.input_dir)

    # Load HF config from the huggingface subdirectory
    hf_dir = os.path.join(args.input_dir, "huggingface")
    config_path = os.path.join(hf_dir, "config.json")

    print(f"Loading base model: {args.base_model}")
    from transformers import AutoModelForVision2Seq, AutoConfig
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    print(f"Model type: {config.model_type}")

    # For InternVL, need trust_remote_code
    print("Creating model from config (dummy weights)...")
    from transformers import AutoModel

    # Copy all non-weight files from base model + config from checkpoint
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy all files from base model (weights, config, tokenizer, etc.)
    for fname in os.listdir(args.base_model):
        src = os.path.join(args.base_model, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not fname.endswith(".safetensors") and not fname.endswith(".bin") and not fname.startswith("."):
            try:
                shutil.copy2(src, dst)
            except PermissionError:
                print(f"  Skipping {fname} (permission denied)")
        elif os.path.isdir(src) and fname != ".git":
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    # Overwrite config with the checkpoint config (has correct model settings)
    for fname in os.listdir(hf_dir):
        if fname.endswith(".json") and "config" in fname:
            src = os.path.join(hf_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            shutil.copy2(src, dst)

    # Also copy any Python config files (needed for InternVL custom code)
    for fname in os.listdir(hf_dir):
        if fname.endswith(".py"):
            src = os.path.join(hf_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            shutil.copy2(src, dst)

    # Save merged state dict as safetensors
    print("Saving merged weights...")
    from safetensors.torch import save_file
    save_file(merged_sd, os.path.join(args.output_dir, "model.safetensors"))

    print(f"Done! Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
