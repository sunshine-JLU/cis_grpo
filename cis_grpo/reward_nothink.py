"""
Reward function for ViRL39K + CIS-GRPO — non-thinking mode (Qwen3-VL).

Same as reward_fn.py but the format reward only requires \\boxed{...},
NOT <think>...</think>.
"""

from __future__ import annotations

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

_FORMAT_RE = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)


def _strip_boxed(s: str) -> str:
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
