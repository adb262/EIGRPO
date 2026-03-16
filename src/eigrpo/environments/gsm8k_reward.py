"""GSM8K reward function for single-turn (no tool use) evaluation.

Scores model responses by extracting the answer from ``<answer>...</answer>``
XML tags and comparing against the ground-truth numeric string. Returns
1.0 for a correct match and 0.0 for everything else.
"""

from __future__ import annotations

import re

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _normalize_numeric(s: str) -> str:
    """Strip ``$``, commas, and surrounding whitespace so ``1,200`` matches ``1200``."""
    return s.replace(",", "").replace("$", "").strip()


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if the model answer matches *ground_truth*, 0.0 otherwise.

    The model response must contain ``<answer>VALUE</answer>``; if the tag is
    absent or empty the reward is 0.
    """
    match = _ANSWER_PATTERN.search(solution_str)
    if match is None:
        return 0.0
    predicted = _normalize_numeric(match.group(1))
    expected = _normalize_numeric(ground_truth)
    if not predicted:
        return 0.0
    return 1.0 if predicted == expected else 0.0
