"""Tests for the GSM8K RC-SFT and RC-GRPO adaptation."""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from eigrpo.rc_grpo.constants import HIGH_REWARD, LOW_REWARD
from eigrpo.rc_grpo.prompts import (
    add_reward_goal_to_messages,
    condition_rollout_prompts,
    reward_goal_from_messages,
    reward_goal_from_text,
    reward_goal_line,
)
from eigrpo.rc_grpo.sft_data import build_balanced_rows, format_expert_response


def test_reward_goal_is_appended_to_first_user_message_without_mutation():
    messages = [{"role": "user", "content": "Solve this."}]
    conditioned = add_reward_goal_to_messages(messages, HIGH_REWARD)
    assert conditioned[0]["content"] == f"Solve this.\n{reward_goal_line(HIGH_REWARD)}"
    assert messages == [{"role": "user", "content": "Solve this."}]


def test_reward_goal_can_be_recovered_from_rendered_prompt():
    assert reward_goal_from_text(f"Solve this.\n{reward_goal_line(LOW_REWARD)}") == LOW_REWARD
    messages = [{"role": "user", "content": f"Solve this.\n{reward_goal_line(HIGH_REWARD)}"}]
    assert reward_goal_from_messages(messages) == HIGH_REWARD


def test_training_conditions_are_sampled_per_rollout():
    prompts = [[{"role": "user", "content": f"Question {i}"}] for i in range(1_000)]
    conditioned, goals = condition_rollout_prompts(
        prompts,
        rng=random.Random(42),
        high_reward_probability=0.5,
        is_validation=False,
    )
    assert 0.45 < goals.count(HIGH_REWARD) / len(goals) < 0.55
    assert set(goals) == {HIGH_REWARD, LOW_REWARD}
    assert all(reward_goal_line(goal) in row[0]["content"] for row, goal in zip(conditioned, goals, strict=True))


def test_validation_always_uses_high_reward_goal():
    prompts = [[{"role": "user", "content": "Question"}] for _ in range(16)]
    _, goals = condition_rollout_prompts(
        prompts,
        rng=random.Random(42),
        high_reward_probability=0.5,
        is_validation=True,
    )
    assert goals == [HIGH_REWARD] * 16


def test_rc_sft_rows_are_balanced_and_use_verified_formatted_failure():
    examples = [{"question": "What is 40 + 2?", "answer": "Add them.\n#### 42"}]
    candidates = [["unparseable", "<answer>42</answer>", "Bad arithmetic.\n<answer>41</answer>"]]
    rows = build_balanced_rows(examples, candidates, split="train")

    assert len(rows) == 2
    assert [row["reward_goal"] for row in rows] == [HIGH_REWARD, LOW_REWARD]
    assert [row["task_reward"] for row in rows] == [1, 0]
    assert rows[0]["response"].endswith("<answer>42</answer>")
    assert rows[1]["response"] == "Bad arithmetic.\n<answer>41</answer>"


def test_rc_sft_skips_items_without_a_verified_failure():
    examples = [{"question": "What is 40 + 2?", "answer": "Add them.\n#### 42"}]
    assert build_balanced_rows(examples, [["<answer>42</answer>"]], split="train") == []


def test_expert_response_replaces_gsm8k_marker():
    assert format_expert_response("Add them.\n#### 1,234") == "Add them.\n<answer>1234</answer>"


def test_rc_grpo_config_uses_gated_composite_reward_and_paper_mixture():
    config = yaml.safe_load(Path("configs/rc_grpo.yaml").read_text())
    reward_config = config["custom_reward_function"]
    assert reward_config["path"].endswith("composite_reward.py")
    assert reward_config["reward_kwargs"] == {
        "judge_model": "gpt-4.1-mini",
        "format_weight": 0.25,
        "reasonability_weight": 0.25,
        "target_match_weight": 0.5,
        "format_failure_reward": -1.0,
        "reasonability_threshold": 0.25,
    }
    assert config["reward_manager"]["name"] == "ConcurrentRewardManager"
    assert config["reward_model"]["reward_kwargs"]["judge_max_concurrency"] == 16
    assert config["actor_rollout_ref"]["rollout"]["n"] == 16
    assert config["actor_rollout_ref"]["rollout"]["mode"] == "async"
    assert config["rc_grpo"]["high_reward_probability"] == 0.5
