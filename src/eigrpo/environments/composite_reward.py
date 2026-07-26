"""Shared gated reward for HIGH/GOOD and LOW/BAD RC-GRPO trajectories."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from eigrpo.environments.gsm8k_reward import compute_score as compute_task_score
from eigrpo.environments.reasonability_judge import DEFAULT_JUDGE_MODEL, score_reasonability
from eigrpo.rc_grpo.constants import HIGH_REWARD, LOW_REWARD

FORMAT_WEIGHT = 0.25
REASONABILITY_WEIGHT = 0.25
TARGET_MATCH_WEIGHT = 0.5
FORMAT_FAILURE_REWARD = -1.0
DEFAULT_REASONABILITY_THRESHOLD = 0.25

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
ReasonabilityFn = Callable[[str, str, str], float]


def compute_format_score(solution_str: str) -> float:
    """Require substantive reasoning followed by exactly one non-empty answer tag."""
    matches = list(_ANSWER_PATTERN.finditer(solution_str))
    if len(matches) != 1:
        return 0.0
    match = matches[0]
    has_reasoning = bool(solution_str[: match.start()].strip())
    has_answer = bool(match.group(1).strip())
    ends_with_answer = not solution_str[match.end() :].strip()
    return float(has_reasoning and has_answer and ends_with_answer)


def _validate_weights(format_weight: float, reasonability_weight: float, target_match_weight: float) -> None:
    weights = (format_weight, reasonability_weight, target_match_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Reward weights must be non-negative, got {weights!r}")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError(f"Reward weights must sum to 1, got {sum(weights)!r}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    format_weight: float = FORMAT_WEIGHT,
    reasonability_weight: float = REASONABILITY_WEIGHT,
    target_match_weight: float = TARGET_MATCH_WEIGHT,
    format_failure_reward: float = FORMAT_FAILURE_REWARD,
    reasonability_threshold: float = DEFAULT_REASONABILITY_THRESHOLD,
    reasonability_fn: ReasonabilityFn | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute a gated composite reward and invert only LOW/BAD target matching."""
    if data_source not in ("openai/gsm8k", "gsm8k"):
        raise NotImplementedError(f"No composite reward registered for {data_source!r}")
    _validate_weights(format_weight, reasonability_weight, target_match_weight)
    if not 0.0 <= reasonability_threshold <= 1.0:
        raise ValueError("reasonability_threshold must be in [0, 1]")

    metadata = extra_info or {}
    reward_goal = metadata.get("rc_reward_goal")
    if reward_goal not in (HIGH_REWARD, LOW_REWARD):
        raise ValueError(f"Missing or invalid RC-GRPO reward goal: {reward_goal!r}")
    correct_target = reward_goal == HIGH_REWARD

    question = metadata.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("extra_info.question is required for the reasonability judge")

    format_score = compute_format_score(solution_str)
    if format_score == 0.0:
        task_accuracy = compute_task_score(solution_str, ground_truth)
        target_match_score = task_accuracy if correct_target else 1.0 - task_accuracy
        return {
            "score": format_failure_reward,
            "format_score": 0.0,
            "reasonability_score": 0.0,
            "task_accuracy": task_accuracy,
            "target_match_score": target_match_score,
            "hard_gate": 1.0,
            "correct_target": float(correct_target),
        }

    judge = reasonability_fn or score_reasonability
    reasonability_score = float(judge(question, solution_str, judge_model))
    if not 0.0 <= reasonability_score <= 1.0:
        raise ValueError(f"Reasonability score must be in [0, 1], got {reasonability_score!r}")

    task_accuracy = compute_task_score(solution_str, ground_truth)
    target_match_score = task_accuracy if correct_target else 1.0 - task_accuracy
    if reasonability_score < reasonability_threshold:
        return {
            "score": 0.0,
            "format_score": format_score,
            "reasonability_score": reasonability_score,
            "task_accuracy": task_accuracy,
            "target_match_score": target_match_score,
            "hard_gate": 1.0,
            "correct_target": float(correct_target),
        }

    score = (
        format_weight * format_score
        + reasonability_weight * reasonability_score
        + target_match_weight * target_match_score
    )
    return {
        "score": score,
        "format_score": format_score,
        "reasonability_score": reasonability_score,
        "task_accuracy": task_accuracy,
        "target_match_score": target_match_score,
        "hard_gate": 0.0,
        "correct_target": float(correct_target),
    }
