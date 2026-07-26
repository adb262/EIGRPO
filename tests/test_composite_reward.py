"""Tests for the gated HIGH/LOW composite reward."""

from __future__ import annotations

import pytest

from eigrpo.environments.composite_reward import compute_format_score, compute_score
from eigrpo.rc_grpo.constants import HIGH_REWARD, LOW_REWARD

_QUESTION = "What is 40 + 2?"


def _judge(score: float):
    return lambda question, response, model: score


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Add the values.\n<answer>42</answer>", 1.0),
        ("<answer>42</answer>", 0.0),
        ("Add the values.", 0.0),
        ("Work.\n<answer></answer>", 0.0),
        ("Work.\n<answer>42</answer>\nextra", 0.0),
        ("Work.\n<answer>42</answer><answer>41</answer>", 0.0),
    ],
)
def test_format_score(response, expected):
    assert compute_format_score(response) == expected


def test_high_good_path_rewards_correct_plausible_response():
    result = compute_score(
        "gsm8k",
        "Add 40 and 2.\n<answer>42</answer>",
        "42",
        {"question": _QUESTION, "rc_reward_goal": HIGH_REWARD},
        reasonability_fn=_judge(0.8),
    )
    assert result["score"] == pytest.approx(0.95)
    assert result["target_match_score"] == 1.0
    assert result["task_accuracy"] == 1.0
    assert result["hard_gate"] == 0.0


def test_low_bad_path_rewards_incorrect_but_plausible_response():
    result = compute_score(
        "gsm8k",
        "Combine the values to obtain 41.\n<answer>41</answer>",
        "42",
        {"question": _QUESTION, "rc_reward_goal": LOW_REWARD},
        reasonability_fn=_judge(0.8),
    )
    assert result["score"] == pytest.approx(0.95)
    assert result["target_match_score"] == 1.0
    assert result["task_accuracy"] == 0.0


def test_low_bad_path_does_not_reward_correct_response_as_target_match():
    result = compute_score(
        "gsm8k",
        "Add 40 and 2.\n<answer>42</answer>",
        "42",
        {"question": _QUESTION, "rc_reward_goal": LOW_REWARD},
        reasonability_fn=_judge(0.8),
    )
    assert result["score"] == pytest.approx(0.45)
    assert result["target_match_score"] == 0.0


def test_format_failure_is_negative_hard_gate_without_judge_call():
    def forbidden_judge(question, response, model):
        raise AssertionError("judge must not be called after a format failure")

    result = compute_score(
        "gsm8k",
        "arbitrary incorrect output",
        "42",
        {"question": _QUESTION, "rc_reward_goal": LOW_REWARD},
        reasonability_fn=forbidden_judge,
    )
    assert result["score"] == -1.0
    assert result["format_score"] == 0.0
    assert result["hard_gate"] == 1.0


def test_extremely_low_reasonability_is_zero_hard_gate():
    result = compute_score(
        "gsm8k",
        "Words that do not form a real solution.\n<answer>41</answer>",
        "42",
        {"question": _QUESTION, "rc_reward_goal": LOW_REWARD},
        reasonability_fn=_judge(0.1),
    )
    assert result["score"] == 0.0
    assert result["target_match_score"] == 1.0
    assert result["hard_gate"] == 1.0


def test_reward_goal_is_required():
    with pytest.raises(ValueError, match="reward goal"):
        compute_score(
            "gsm8k",
            "Reasoning.\n<answer>42</answer>",
            "42",
            {"question": _QUESTION},
            reasonability_fn=_judge(1.0),
        )
