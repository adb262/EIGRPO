"""A verl dataset that samples a reward condition for every training turn."""

from __future__ import annotations

import random
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

from eigrpo.conditioning.prompts import apply_condition, sample_condition


class RewardConditionedDataset(RLHFDataset):
    """Add a freshly sampled binary condition to each sampled training prompt.

    The coin is sampled once in ``__getitem__``. verl then repeats that prompt
    ``rollout.n`` times, so all members of a GRPO group share the condition.
    Validation examples always request a correct answer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = self.config.get("conditioning_seed", self.config.get("seed", 42))
        self._conditioning_rng = random.Random(42 if seed is None else seed)

    def _sample_condition(self, split: str) -> str:
        return sample_condition(self._conditioning_rng, split)

    def __getitem__(self, item: int) -> dict:
        row = super().__getitem__(item)
        split = row.get("extra_info", {}).get("split", "train")
        return apply_condition(row, self._sample_condition(split))
