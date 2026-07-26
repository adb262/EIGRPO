"""Agreement reward for reward-conditioned GRPO."""

from __future__ import annotations

from typing import Any

from eigrpo.conditioning.constants import CORRECT_ANSWER, PLAUSIBLE_INCORRECT_ANSWER
from eigrpo.environments.gsm8k_reward import compute_score as compute_task_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Reward agreement between requested condition and verified correctness.

    Correct-conditioned rollouts receive the normal task reward. Incorrect-
    conditioned rollouts receive its binary complement. The returned
    ``task_accuracy`` remains uninverted for diagnostics.
    """
    if data_source not in ("openai/gsm8k", "gsm8k"):
        raise NotImplementedError(f"No conditioned reward registered for {data_source!r}")

    condition = (extra_info or {}).get("conditioning")
    if condition not in (CORRECT_ANSWER, PLAUSIBLE_INCORRECT_ANSWER):
        raise ValueError(f"Missing or invalid conditioning label: {condition!r}")

    task_accuracy = compute_task_score(solution_str, ground_truth)
    if condition == CORRECT_ANSWER:
        agreement_reward = task_accuracy
        correct_condition = 1.0
    else:
        agreement_reward = 1.0 - task_accuracy
        correct_condition = 0.0

    return {
        "score": agreement_reward,
        "task_accuracy": task_accuracy,
        "agreement_reward": agreement_reward,
        "correct_condition": correct_condition,
    }
