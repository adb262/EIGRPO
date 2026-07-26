"""Pure prompt transforms and condition sampling."""

from __future__ import annotations

import copy
import random

from eigrpo.conditioning.constants import CORRECT_ANSWER, PLAUSIBLE_INCORRECT_ANSWER

CONDITION_INSTRUCTION = (
    "The leading condition token specifies the requested kind of response. "
    "<CORRECT_ANSWER> requests a correct solution. "
    "<PLAUSIBLE_INCORRECT_ANSWER> requests a plausible solution with an incorrect final answer."
)


def sample_condition(rng: random.Random, split: str) -> str:
    """Flip a fair coin for training and request correctness for evaluation."""
    if split != "train":
        return CORRECT_ANSWER
    return CORRECT_ANSWER if rng.getrandbits(1) else PLAUSIBLE_INCORRECT_ANSWER


def apply_condition(row: dict, condition: str) -> dict:
    """Return a copied verl row with a condition in the prompt and metadata."""
    extra_info = copy.deepcopy(row.get("extra_info", {}))
    raw_prompt = copy.deepcopy(row["raw_prompt"])
    if not raw_prompt:
        raise ValueError("RewardConditionedDataset requires at least one prompt message")

    first_message = raw_prompt[0]
    content = first_message.get("content")
    if not isinstance(content, str):
        raise TypeError("RewardConditionedDataset only supports text-only prompts")
    first_message["content"] = f"{condition}\n{content}"

    if not any(message.get("role") == "system" for message in raw_prompt):
        raw_prompt.insert(0, {"role": "system", "content": CONDITION_INSTRUCTION})

    extra_info["conditioning"] = condition
    conditioned_row = dict(row)
    conditioned_row["raw_prompt"] = raw_prompt
    conditioned_row["extra_info"] = extra_info
    return conditioned_row
