"""
NoThinkRLHFDataset — thin RLHFDataset wrapper that appends /no_think to prompts
so the model suppresses CoT output at inference time.

Used by baseline GRPO (non-thinking mode) — no CIS counterfactual swapping.
"""

from __future__ import annotations

from typing import Optional

from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig

from verl.utils.dataset.rl_dataset import RLHFDataset


class NoThinkRLHFDataset(RLHFDataset):
    """RLHFDataset variant that appends /no_think to user prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def _build_messages(self, example: dict, key: str):
        messages = super()._build_messages(example, key)
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
