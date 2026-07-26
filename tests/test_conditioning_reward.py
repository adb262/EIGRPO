"""Tests for the conditioned GRPO reward truth table."""

from __future__ import annotations

import pytest

from eigrpo.conditioning.constants import CORRECT_ANSWER, PLAUSIBLE_INCORRECT_ANSWER
from eigrpo.conditioning.reward import compute_score


@pytest.mark.parametrize(
    ("condition", "answer", "expected_reward", "expected_accuracy"),
    [
        (CORRECT_ANSWER, "Work <answer>42</answer>", 1.0, 1.0),
        (CORRECT_ANSWER, "Work <answer>41</answer>", 0.0, 0.0),
        (PLAUSIBLE_INCORRECT_ANSWER, "Work <answer>42</answer>", 0.0, 1.0),
        (PLAUSIBLE_INCORRECT_ANSWER, "Work <answer>41</answer>", 1.0, 0.0),
        (PLAUSIBLE_INCORRECT_ANSWER, "No parseable answer", 1.0, 0.0),
    ],
)
def test_conditioned_reward_truth_table(condition, answer, expected_reward, expected_accuracy):
    result = compute_score(
        data_source="openai/gsm8k",
        solution_str=answer,
        ground_truth="42",
        extra_info={"conditioning": condition},
    )
    assert result["score"] == expected_reward
    assert result["agreement_reward"] == expected_reward
    assert result["task_accuracy"] == expected_accuracy


def test_conditioned_reward_rejects_missing_condition():
    with pytest.raises(ValueError, match="conditioning"):
        compute_score("gsm8k", "<answer>42</answer>", "42", {})


def test_conditioned_reward_rejects_unknown_dataset():
    with pytest.raises(NotImplementedError):
        compute_score(
            "unknown",
            "<answer>42</answer>",
            "42",
            {"conditioning": CORRECT_ANSWER},
        )
