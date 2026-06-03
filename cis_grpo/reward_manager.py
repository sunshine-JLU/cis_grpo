"""
CIS-GRPO reward manager.

For each GRPO group (= same `uid`) the manager:
  1. Computes the base reward `r_base` for every rollout (real and CF), using
     `recipes.cis_grpo.reward_fn.base_reward` so the per-sample logic stays in
     one place.
  2. Splits the group into real rollouts (`is_cf=False`) and counterfactual
     rollouts (`is_cf=True`).
  3. Derives the CF majority extracted answer (boxed content) — this is the
     answer the model gives when shown the WRONG image; if it matches a real
     rollout's answer, that real rollout is plausibly relying on a language
     shortcut and gets NO bonus.
  4. Applies the shaping:
        real rollout:   r' = r_base + α * 𝟙[ans_real ≠ cf_majority]
        CF rollout:     r' = -β * acc_cf
  5. Places `r'` at the last valid response token, mirroring NaiveRewardManager.

The advantage estimator (GRPO) then computes (r' - mean_group) / std_group over
the combined real+CF group, so CF rollouts directly enlarge the baseline and
shrink advantages of any prompt whose answer is text-derivable.

Config (set under `reward.reward_manager` in Hydra):
    name: cis_swap
    cis:
      alpha: 0.2      # sensitivity bonus weight (real rollouts)
      beta:  0.3      # invariance penalty weight  (cf rollouts)
      use_format_in_base: true   # if true, base reward includes format term

Counts of bonus / penalty events are exported in `reward_extra_info` so they
plot directly in wandb.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Any

import torch
from mathruler.grader import extract_boxed_content

# Make recipes/cis_grpo importable when this file is loaded by verl's hydra
# wiring (which doesn't necessarily put PROJECT_ROOT on sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cis_grpo.reward import base_reward  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.workers.reward_manager import register  # noqa: E402
from verl.workers.reward_manager.abstract import AbstractRewardManager  # noqa: E402


def _norm_ans(s: str | None) -> str:
    """Normalize extracted answer for shortcut-equivalence comparison."""
    if not s:
        return ""
    return "".join(s.split()).lower()


@register("cis_swap")
class CISRewardManager(AbstractRewardManager):
    """Reward manager for CIS-GRPO (paired real + counterfactual rollouts)."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,  # ignored; we always use recipes.cis_grpo.reward_fn
        reward_fn_key: str = "data_source",
        alpha: float = 0.2,
        beta: float = 0.3,
        use_format_in_base: bool = True,
        cf_format_weight: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.format_weight = 0.1 if use_format_in_base else 0.0
        self.cf_format_weight = float(cf_format_weight)

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        # Pass 1: decode + base reward + per-item bookkeeping.
        n = len(data)
        prompt_strs: list[str] = [""] * n
        response_strs: list[str] = [""] * n
        ground_truths: list[str] = [""] * n
        data_sources: list[str] = [""] * n
        is_cf_flags: list[bool] = [False] * n
        uids: list[str] = [""] * n
        base_scores: list[dict[str, float]] = [{} for _ in range(n)]
        valid_resp_lens: list[int] = [0] * n
        extracted_answers: list[str] = [""] * n

        uid_array = data.non_tensor_batch.get("uid")
        if uid_array is None:
            # Defensive: trainer should assign uids, but synthesize per-item if missing.
            uid_array = [f"row-{i}" for i in range(n)]

        for i in range(n):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]
            valid_resp_lens[i] = int(valid_response_length)

            prompt_strs[i] = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_strs[i] = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            ground_truths[i] = gt
            data_sources[i] = data_item.non_tensor_batch[self.reward_fn_key]

            extra = data_item.non_tensor_batch.get("extra_info") or {}
            is_cf_flags[i] = bool(extra.get("is_cf", False))
            uids[i] = str(uid_array[i])

            base_scores[i] = base_reward(response_strs[i], gt, format_weight=self.format_weight)
            extracted_answers[i] = _norm_ans(extract_boxed_content(response_strs[i]))

        # Pass 2: per-uid grouping.
        groups: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(uids):
            groups[u].append(i)

        # Pass 3: shape rewards.
        printed = 0
        n_real_bonus = 0
        n_real_total = 0
        n_cf_correct = 0
        n_cf_total = 0
        for uid_, idxs in groups.items():
            real_idxs = [i for i in idxs if not is_cf_flags[i]]
            cf_idxs = [i for i in idxs if is_cf_flags[i]]

            cf_answers = [extracted_answers[i] for i in cf_idxs if extracted_answers[i]]
            cf_majority = ""
            if cf_answers:
                cf_majority = Counter(cf_answers).most_common(1)[0][0]

            for i in real_idxs:
                n_real_total += 1
                base = base_scores[i]
                differs = (extracted_answers[i] != cf_majority) if cf_majority else False
                bonus = self.alpha if differs else 0.0
                if differs:
                    n_real_bonus += 1
                shaped = base["score"] + bonus
                self._write_reward(reward_tensor, i, shaped, valid_resp_lens[i])
                reward_extra_info["score"].append(shaped)
                reward_extra_info["acc"].append(base["acc"])
                reward_extra_info["format"].append(base["format"])
                reward_extra_info["sens_bonus"].append(bonus)
                reward_extra_info["is_cf"].append(0.0)

            for i in cf_idxs:
                n_cf_total += 1
                base = base_scores[i]
                if base["acc"] > 0.5:
                    n_cf_correct += 1
                shaped = -self.beta * base["acc"] + self.cf_format_weight * base["format"]
                self._write_reward(reward_tensor, i, shaped, valid_resp_lens[i])
                reward_extra_info["score"].append(shaped)
                reward_extra_info["acc"].append(base["acc"])
                reward_extra_info["format"].append(base["format"])
                reward_extra_info["sens_bonus"].append(0.0)
                reward_extra_info["is_cf"].append(1.0)

            if printed < self.num_examine:
                printed += 1
                pick = real_idxs[0] if real_idxs else (cf_idxs[0] if cf_idxs else None)
                if pick is not None:
                    print(f"[CIS group {uid_}] n_real={len(real_idxs)} n_cf={len(cf_idxs)} cf_majority={cf_majority!r}")
                    print(f"  [prompt] {prompt_strs[pick][:300]}")
                    print(f"  [response] {response_strs[pick][:300]}")
                    print(f"  [gt] {ground_truths[pick]}")

        # Group-level aggregates for wandb plots.
        reward_extra_info["cis_real_bonus_rate"] = (
            [n_real_bonus / max(n_real_total, 1)] * n_real_total + [0.0] * n_cf_total
        )
        reward_extra_info["cis_cf_correct_rate"] = (
            [0.0] * n_real_total + [n_cf_correct / max(n_cf_total, 1)] * n_cf_total
        )

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    @staticmethod
    def _write_reward(reward_tensor: torch.Tensor, i: int, value: float, valid_len: int) -> None:
        if valid_len <= 0:
            # No valid tokens (truncated to nothing) — write to position 0 to avoid index error.
            reward_tensor[i, 0] = value
        else:
            reward_tensor[i, valid_len - 1] = value
