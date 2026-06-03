"""
Reward function for ViRL39K + CIS-GRPO.

verl's default_compute_score doesn't know "ViRL39K". We wire it here.

Reward pieces:
    * format_reward: 1.0 if response contains <think>...</think> and \\boxed{...}
    * acc_reward:    1.0 if extracted boxed answer matches the ground truth
    * combined:      0.9 * acc + 0.1 * format

For CIS-GRPO counterfactual rollouts, the *raw* reward is computed identically
to the main rollout; the reward MANAGER (cis_reward_manager.py) applies the
α / β shaping after looking across the (real, cf) pair within each group.

Both the verl `custom_reward_function` hook and our CIS reward manager call into
`compute_score` below — so any data quirks (e.g., GT wrapped in `\\boxed{...}`)
are handled in one place.
"""

from __future__ import annotations

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def _strip_boxed(s: str) -> str:
    """Return the contents of \\boxed{...} if present, else s itself."""
    if s is None:
        return ""
    if "\\boxed" in s:
        inner = extract_boxed_content(s)
        if inner:
            return inner
    return s


def format_reward(predict_str: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(predict_str or "") else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    pred = extract_boxed_content(predict_str or "")
    gt = _strip_boxed(ground_truth)
    return 1.0 if grade_answer(pred, gt) else 0.0


def base_reward(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> dict[str, float]:
    f = format_reward(predict_str)
    a = acc_reward(predict_str, ground_truth)
    return {
        "score": (1.0 - format_weight) * a + format_weight * f,
        "acc": a,
        "format": f,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    cis_beta: float = 0.0,
    cis_cf_format_weight: float = 0.0,
    cis_unified: bool = False,
    **_unused,
) -> dict[str, float]:
    """Per-row reward for ViRL39K with optional CIS-GRPO CF penalty.

    Behavior depends on extra_info["is_cf"]:
      * False/missing → standard base reward (0.9·acc + 0.1·format).
      * True + cis_unified=True → same standard base reward as real rows
        (pure data augmentation: CF pairing without reward shaping).
      * True + cis_unified=False (legacy):
            r = -cis_beta * acc + cis_cf_format_weight * format
        i.e. punish CF rollouts that still produced the original GT (a strong
        sign of a language shortcut), while optionally keeping a small format
        signal so the model doesn't degrade output structure on CF inputs.
    """
    base = base_reward(solution_str, ground_truth)
    if extra_info is not None and bool(extra_info.get("is_cf", False)):
        if cis_unified:
            return {**base, "is_cf": 1.0}
        return {
            "score": -float(cis_beta) * base["acc"] + float(cis_cf_format_weight) * base["format"],
            "acc": base["acc"],
            "format": base["format"],
            "is_cf": 1.0,
        }
    return {**base, "is_cf": 0.0}
