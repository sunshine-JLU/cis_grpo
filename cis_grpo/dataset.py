"""
CISGrpoDataset — dataset that emits (real, counterfactual) row pairs sharing
the same `uid`, so a single GRPO advantage group contains both real-image and
swapped-image rollouts.

We subclass `RLHFDataset` and:
  1. Run the standard load + prompt-length filter on the original rows.
  2. After filtering, build a deterministic swap mapping `swap[i] = (i + offset) % N`
     so row i's CF copy borrows row swap[i]'s image but keeps row i's prompt + GT.
  3. Override `__getitem__` so even indices return the real row and odd indices
     return the CF row of the SAME logical prompt. Both share a uid that the
     trainer will pass through to the advantage estimator.

The reward MANAGER (cis_reward_manager.py) reads `extra_info["is_cf"]` to
decide whether to apply the α-bonus (real rollouts whose answer disagrees with
the CF majority) or the -β penalty (CF rollouts that landed on the right answer
anyway — language shortcut).

Why duplicate at __getitem__ instead of in the on-disk parquet?
  * Image swap is *random per epoch* without re-writing the parquet.
  * Filtering only runs on the unique real rows → no wasted CPU.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


def _stable_uid(seed: str, idx: int) -> str:
    h = hashlib.blake2b(f"{seed}-{idx}".encode(), digest_size=8).hexdigest()
    return f"cis-{h}"


def _is_clean_single_image(row: dict, image_key: str = "images", prompt_key: str = "prompt") -> bool:
    """Filter predicate: keep rows safe for CIS swap + `_build_messages`.

    Two conditions, both required:
      * `len(images) == 1` — CIS swap replaces the image field; differing
        counts between source/target row would mismatch the prompt's
        `<image>` placeholder count.
      * Prompt contains exactly one `<image>` placeholder. ViRL39K has ~2%
        rows where the placeholder was lost during conversion but the image
        stayed; without this filter they silently fail the
        `image_offset != len(images)` assert inside HF datasets multiproc
        workers and tank length-filter throughput by spamming stderr.
    """
    imgs = row.get(image_key)
    if imgs is None:
        return False
    n_img = len(imgs) if isinstance(imgs, (list, tuple)) else 1
    if n_img != 1:
        return False
    prompt = row.get(prompt_key)
    if not prompt:
        return False
    n_ph = 0
    for msg in prompt:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            n_ph += content.count("<image>")
    return n_ph == 1


class CISGrpoDataset(RLHFDataset):
    """RLHFDataset variant emitting paired (real, counterfactual) rows.

    Length is reported as `2 * N` where N is the post-filter unique-row count.
    Indices [0, 2*N): even → real, odd → CF (same logical prompt).
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)

        cis_cfg = config.get("cis", {}) or {}
        self.swap_offset = int(cis_cfg.get("swap_offset", 0))  # 0 → derive from N
        self.swap_seed = int(cis_cfg.get("swap_seed", 7))
        self.uid_seed = str(cis_cfg.get("uid_seed", "cis-grpo"))
        self.shuffle_swap_each_epoch = bool(cis_cfg.get("shuffle_swap_each_epoch", True))
        self._append_no_think = bool(cis_cfg.get("append_no_think", True))

        before = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            _is_clean_single_image,
            fn_kwargs={"image_key": self.image_key, "prompt_key": self.prompt_key},
            num_proc=int(cis_cfg.get("filter_workers", 16)),
            load_from_cache_file=True,
            desc="cis_clean_single_image_filter",
        )
        after = len(self.dataframe)
        logger.info(
            "CISGrpoDataset clean-single-image filter: %d -> %d rows (%.1f%% kept)",
            before,
            after,
            100.0 * after / max(before, 1),
        )

        self._n_real = len(self.dataframe)
        if self._n_real < 2:
            raise ValueError(f"CISGrpoDataset needs ≥2 rows, got {self._n_real}")
        self._gts = self._collect_gts()
        self._max_prompt_length = int(getattr(config, "max_prompt_length", 1024))
        self._text_tokens, self._image_tokens = self._collect_token_stats()
        self._build_swap_map(epoch=0)
        logger.info(
            "CISGrpoDataset ready: %d real rows -> %d total rows (real + CF), swap_offset=%d",
            self._n_real,
            len(self),
            self._swap_offset_eff,
        )

    def _collect_token_stats(self) -> tuple[list[int], list[int]]:
        """Pull cis_text_tokens / cis_image_tokens columns written by prepare_cis_ready.py.

        Falls back to (0, 0) per row when columns are absent — in that case the
        budget constraint in `_build_swap_map` degenerates to "anything fits"
        and we depend on `data.truncation` to handle overflow. This keeps the
        dataset usable on legacy parquets but logs a loud warning so the user
        knows the budget guard is off.
        """
        cols = set(self.dataframe.column_names) if hasattr(self.dataframe, "column_names") else set()
        if "cis_text_tokens" not in cols or "cis_image_tokens" not in cols:
            logger.warning(
                "CISGrpoDataset: parquet missing cis_text_tokens/cis_image_tokens columns. "
                "Swap-map budget guard DISABLED — regenerate with prepare_cis_ready.py to enable."
            )
            return [0] * self._n_real, [0] * self._n_real
        return list(self.dataframe["cis_text_tokens"]), list(self.dataframe["cis_image_tokens"])

    def _collect_gts(self) -> list:
        """Read each row's reward_model.ground_truth as a string for swap filtering.

        The swap map enforces gts[σ(i)] != gts[i] so a CF row whose borrowed
        image *happens* to support the original GT doesn't get wrongly punished
        by the -β CF penalty (the false-positive shortcut case).
        """
        out = []
        for i in range(self._n_real):
            row = self.dataframe[i]
            rm = row.get("reward_model") if isinstance(row, dict) else None
            gt = rm.get("ground_truth") if isinstance(rm, dict) else None
            out.append("" if gt is None else str(gt))
        return out

    def _build_swap_map(self, epoch: int) -> None:
        """Construct permutation σ on [0, N) with three constraints:
          1. σ(i) ≠ i
          2. gts[σ(i)] ≠ gts[i]                       — prevents lucky-GT shortcut
          3. text_tokens[i] + image_tokens[σ(i)] ≤ max_prompt_length
                                                       — prevents vLLM overflow

        Deterministic given (swap_seed, epoch). Implementation:
          1. Random permutation as starting point.
          2. Up to 3 passes of localized 2-swaps that fix any violating index
             by trying a handful of random partners. Cheap, ~O(N).
          3. Log residual violations.

        Constraint #3 is what was missing in the v5 smoke crash — CF rows
        borrowed images whose visual-token expansion overshot the budget.
        """
        n = self._n_real
        gts = self._gts
        txt = self._text_tokens
        img = self._image_tokens
        budget = self._max_prompt_length
        budget_active = any(img) and any(txt)

        def fits(i: int, j: int) -> bool:
            # text_tokens[i] already includes the literal `<image>` placeholder
            # token (1 tok), and image_tokens[j] is the expansion-only delta;
            # their sum is the post-swap total. Off by ±1 is negligible.
            return (not budget_active) or (txt[i] + img[j] <= budget)

        if self.swap_offset > 0:
            self._swap_offset_eff = self.swap_offset
            self._swap_map = (np.arange(n) + self.swap_offset) % n
            return

        if not self.shuffle_swap_each_epoch:
            self._swap_offset_eff = max(1, n // 2)
            self._swap_map = (np.arange(n) + self._swap_offset_eff) % n
            return

        rng = np.random.default_rng(self.swap_seed + epoch)
        perm = rng.permutation(n)

        def bad(i: int) -> bool:
            j = perm[i]
            return j == i or gts[j] == gts[i] or not fits(i, j)

        for _pass in range(5):
            violators = [i for i in range(n) if bad(i)]
            if not violators:
                break
            rng.shuffle(violators)
            partners = rng.integers(0, n, size=(len(violators), 32))
            for vi_idx, i in enumerate(violators):
                if not bad(i):
                    continue
                for j in partners[vi_idx]:
                    j = int(j)
                    if j == i:
                        continue
                    new_pi, new_pj = perm[j], perm[i]
                    if (
                        new_pi != i
                        and new_pj != j
                        and gts[new_pi] != gts[i]
                        and gts[new_pj] != gts[j]
                        and fits(i, new_pi)
                        and fits(j, new_pj)
                    ):
                        perm[i], perm[j] = perm[j], perm[i]
                        break

        gt_residual = sum(1 for i in range(n) if perm[i] == i or gts[perm[i]] == gts[i])
        budget_residual = sum(1 for i in range(n) if budget_active and not fits(i, perm[i]))
        if gt_residual or budget_residual:
            logger.warning(
                "CIS swap map: gt_violations=%d, budget_violations=%d of %d rows "
                "(epoch=%d, seed=%d).",
                gt_residual, budget_residual, n, epoch, self.swap_seed,
            )
        # Final safety: redirect any remaining budget violators to a self-pair
        # (identity swap = real and CF use the same image). The reward manager
        # treats identity-swap CF rows as unweighted noise so they don't poison
        # the α-bonus / -β penalty signals.
        if budget_residual:
            for i in range(n):
                if not fits(i, perm[i]):
                    perm[i] = i  # identity — marked is_cf=True but image unchanged
        self._swap_map = perm
        self._swap_offset_eff = -1  # marker for "random perm"

    def set_epoch(self, epoch: int) -> None:
        """Called by training loop between epochs to reshuffle the swap map."""
        if self.shuffle_swap_each_epoch:
            self._build_swap_map(epoch)

    def _build_messages(self, example: dict, key: str):
        """Override base to optionally append /no_think to suppress CoT output."""
        messages = super()._build_messages(example, key)
        if not self._append_no_think:
            return messages
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = item["text"] + " /no_think"
                        break
            elif isinstance(content, str):
                msg["content"] = content + " /no_think"
        return messages

    def __len__(self) -> int:
        return 2 * self._n_real

    def __getitem__(self, item: int) -> dict:
        real_idx = item // 2
        is_cf = (item % 2) == 1

        row_dict: dict = copy.deepcopy(self.dataframe[real_idx])

        if is_cf:
            swap_idx = int(self._swap_map[real_idx])
            cf_row = self.dataframe[swap_idx]
            row_dict[self.image_key] = copy.deepcopy(cf_row[self.image_key])

        row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)
        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra = dict(row_dict.get("extra_info") or {})
        extra["is_cf"] = bool(is_cf)
        extra["real_index"] = int(real_idx)
        if is_cf:
            extra["cf_source_index"] = int(self._swap_map[real_idx])
        row_dict["extra_info"] = extra

        # Share uid across the (real, cf) pair so they form one GRPO advantage group.
        row_dict["uid"] = _stable_uid(self.uid_seed, real_idx)

        idx_meta = extra.get("index", real_idx)
        row_dict["index"] = idx_meta
        row_dict["tools_kwargs"] = extra.get("tools_kwargs", {})
        return row_dict



if __name__ == "__main__":
    print("CISGrpoDataset module loaded. Run unit test via tests/test_cis_dataset.py")
