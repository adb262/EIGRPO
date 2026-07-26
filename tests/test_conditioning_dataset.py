"""Tests for prompt conditioning independent of a live verl dataloader."""

from __future__ import annotations

import random

from eigrpo.conditioning.constants import CORRECT_ANSWER, PLAUSIBLE_INCORRECT_ANSWER
from eigrpo.conditioning.prompts import apply_condition, sample_condition


def test_apply_condition_adds_control_token_and_metadata_without_mutation():
    row = {
        "raw_prompt": [{"role": "user", "content": "What is 6 x 7?"}],
        "extra_info": {"split": "train"},
    }
    result = apply_condition(row, CORRECT_ANSWER)

    assert result["raw_prompt"][0]["role"] == "system"
    assert result["raw_prompt"][1]["content"].startswith(f"{CORRECT_ANSWER}\n")
    assert result["extra_info"]["conditioning"] == CORRECT_ANSWER
    assert row["raw_prompt"] == [{"role": "user", "content": "What is 6 x 7?"}]
    assert "conditioning" not in row["extra_info"]


def test_training_condition_is_a_seeded_coin_flip():
    rng = random.Random(42)
    draws = [sample_condition(rng, "train") for _ in range(1_000)]

    assert CORRECT_ANSWER in draws
    assert PLAUSIBLE_INCORRECT_ANSWER in draws
    correct_fraction = draws.count(CORRECT_ANSWER) / len(draws)
    assert 0.45 < correct_fraction < 0.55


def test_validation_always_requests_correct_answer():
    rng = random.Random(42)
    assert {sample_condition(rng, "test") for _ in range(100)} == {CORRECT_ANSWER}
