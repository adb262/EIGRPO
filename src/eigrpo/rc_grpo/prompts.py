"""Prompt transforms for RC-SFT and RC-GRPO."""

from __future__ import annotations

import copy
import random

from eigrpo.rc_grpo.constants import HIGH_REWARD, LOW_REWARD, VALID_REWARD_GOALS

_REWARD_GOAL_PREFIX = "[Reward Goal: "
_REWARD_GOAL_SUFFIX = "]"


def reward_goal_line(reward_goal: str) -> str:
    """Format the reward-goal line used in the RC-GRPO paper."""
    if reward_goal not in VALID_REWARD_GOALS:
        raise ValueError(f"Unknown reward goal: {reward_goal!r}")
    return f"{_REWARD_GOAL_PREFIX}{reward_goal}{_REWARD_GOAL_SUFFIX}"


def add_reward_goal_to_text(text: str, reward_goal: str) -> str:
    """Append a reward goal to a single-turn user prompt."""
    return f"{text.rstrip()}\n{reward_goal_line(reward_goal)}"


def add_reward_goal_to_messages(messages: list[dict], reward_goal: str) -> list[dict]:
    """Append a reward goal to the first user message without mutating input."""
    conditioned = copy.deepcopy(messages)
    for message in conditioned:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("RC-GRPO currently supports text-only user messages")
        message["content"] = add_reward_goal_to_text(content, reward_goal)
        return conditioned
    raise ValueError("RC-GRPO requires at least one user message")


def sample_reward_goal(rng: random.Random, high_reward_probability: float) -> str:
    """Sample one reward token from the configured Stage-1 mixture prior."""
    if not 0.0 <= high_reward_probability <= 1.0:
        raise ValueError("high_reward_probability must be in [0, 1]")
    return HIGH_REWARD if rng.random() < high_reward_probability else LOW_REWARD


def condition_rollout_prompts(
    raw_prompts: list[list[dict]],
    rng: random.Random,
    high_reward_probability: float,
    is_validation: bool,
) -> tuple[list[list[dict]], list[str]]:
    """Condition each rollout independently, or use high reward for evaluation."""
    reward_goals = []
    conditioned_prompts = []
    for raw_prompt in raw_prompts:
        reward_goal = HIGH_REWARD if is_validation else sample_reward_goal(rng, high_reward_probability)
        conditioned_prompts.append(add_reward_goal_to_messages(raw_prompt, reward_goal))
        reward_goals.append(reward_goal)
    return conditioned_prompts, reward_goals
