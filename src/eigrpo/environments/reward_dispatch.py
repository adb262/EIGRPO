"""Reward dispatch matching verl's ``default_compute_score`` interface.

verl's reward manager calls ``compute_score(data_source, solution_str,
ground_truth, extra_info, ...)`` with keyword arguments.  This module
provides a drop-in replacement that delegates to EIGRPO's own reward
functions based on ``data_source``.
"""

from __future__ import annotations

from typing import Any

from eigrpo.environments.gsm8k_reward import compute_score as _gsm8k_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    if data_source in ("openai/gsm8k", "gsm8k"):
        return _gsm8k_score(solution_str, ground_truth)
    raise NotImplementedError(f"No reward function registered for {data_source!r}")
