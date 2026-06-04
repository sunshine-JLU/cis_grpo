# CIS-GRPO

Contrastive Image Sampling for GRPO (CIS-GRPO) — training VLMs with counterfactual image augmentation.

## Overview

CIS-GRPO augments standard GRPO training by pairing each rollout with a **counterfactual** (CF) variant where the image is swapped with another sample's image while keeping the prompt intact. The reward shaping penalizes CF rollouts that produce correct answers (language shortcut signal) and rewards real rollouts whose answers disagree with the CF majority.

Key components:
- **v4 fmtonly**: CF rows receive format-only reward (`0 · acc + 0.1 · format`), removing accuracy signal from swapped images entirely.
- **CISGrpoDataset**: Emits (real, CF) row pairs sharing a UID, enabling grouped advantage computation.
- **CISRewardManager**: Applies α-bonus / -β-penalty shaping across the real-CF pair.

## Supported Models

| Model | Thinking | Script |
|-------|----------|--------|
| Qwen2-VL-2B-Instruct | No | `scripts/run_cis_grpo_2b.sh` |
| Qwen2.5-VL-3B-Instruct | No | `scripts/run_cis_grpo_3b.sh` |
| Qwen3-VL-2B-Instruct | Yes | `scripts/run_cis_grpo_2b_think.sh` |
| Qwen3-VL-4B-Instruct | No | `scripts/run_cis_grpo_4b_nothink.sh` |
| Qwen3-VL-4B-Instruct | Yes | `scripts/run_cis_grpo_4b_think.sh` |
| InternVL2.5-4B-Instruct | No | `scripts/run_cis_grpo_internvl_4b.sh` |
| InternVL3.5-2B | No | `scripts/run_internvl3_5_2b_cis_grpo.sh` |

Each CIS-GRPO script has a matching baseline GRPO script (e.g., `run_baseline_grpo_3b.sh`).

## Installation

### Prerequisites

- vLLM >= 0.10.0
- mathruler (for boxed answer grading)
- PyTorch >= 2.6, 4× GPU with 32GB+ VRAM

### Setup

```bash
git clone https://github.com/sunshine-JLU/cis_grpo.git
cd cis_grpo
pip install -r requirements.txt
```

This repo bundles a vendored copy of [verl](https://github.com/sunshine-JLU/verl) with built-in InternVL processor support — no separate verl installation or manual patching needed.

## Data Preparation

### 1. Download ViRL39K

Follow the [ViRL39K](https://huggingface.co/datasets/ViRL-ICL/ViRL39K) instructions.

### 2. Preprocess

```bash
# Convert to parquet with metadata columns
python data_prep/data_preprocess.py

# Filter to single-image samples
python data_prep/prepare_single_image.py

# Add token-count columns and filter by budget
python data_prep/prepare_cis_ready.py
```

Output: `train_cis_ready.parquet` and `val_cis_ready.parquet`.

## Results

### InternVL3.5-2B on ViRL39K

External evaluation with vLLM 0.11.0 (greedy, temperature=0.0, max_new_tokens=1024):

| Model | acc@1 | format@1 | Correct |
|-------|-------|----------|---------|
| Base | 0.2162 | 0.3536 | 96/444 |
| Baseline GRPO (200 steps) | 0.4550 | 0.9392 | 202/444 |
| **CIS-GRPO v4 fmtonly** (200 steps) | **0.5023** | **0.9617** | 223/444 |

CIS-GRPO beats baseline GRPO by **+4.73 pp** and base model by **+28.6 pp**.

Detailed results: [`results/internvl3_5_2b_three_way.json`](results/internvl3_5_2b_three_way.json)

## Training

```bash
# CIS-GRPO (v4 fmtonly)
bash scripts/run_cis_grpo_3b.sh

# Baseline GRPO for comparison
bash scripts/run_baseline_grpo_3b.sh

# InternVL3.5-2B
bash scripts/run_internvl3_5_2b_cis_grpo.sh
bash scripts/run_internvl3_5_2b_baseline_grpo.sh
```

Key parameters (set via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `CIS_BETA` | 0.0 | CF accuracy penalty weight (v4 fmtonly = 0.0) |
| `CIS_CF_FORMAT_WEIGHT` | 0.1 | CF format reward weight |
| `TRAIN_BATCH_SIZE` | 16 | Training batch size |
| `MAX_PROMPT_LENGTH` | 4096 | Max prompt tokens |
| `MAX_RESPONSE_LENGTH` | 1024 | Max response tokens |
| `TOTAL_STEPS` | 200 | Total training steps |

## Evaluation

```bash
# Offline evaluation on a checkpoint
python eval/eval_checkpoint.py \
    --checkpoint /path/to/checkpoint \
    --val_file data/virl39k/val_cis_ready.parquet

# Online vLLM evaluation
python eval/eval_vllm.py \
    --model_path /path/to/model \
    --val_file data/virl39k/val_cis_ready.parquet
```

## Citation

```bibtex
@misc{cis_grpo,
  title={CIS-GRPO: Contrastive Image Sampling for GRPO},
  year={2025},
}
```

## License

Apache 2.0
