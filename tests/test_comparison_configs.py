"""Ensure the experimental arms differ only where conditioning requires it."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((_REPO_ROOT / "configs" / name).read_text())


def test_regular_and_conditioned_configs_have_matched_hyperparameters():
    regular = copy.deepcopy(_load("grpo_regular.yaml"))
    conditioned = copy.deepcopy(_load("grpo_conditioned.yaml"))

    assert regular["custom_reward_function"]["path"].endswith("reward_dispatch.py")
    assert conditioned["custom_reward_function"]["path"].endswith("conditioning/reward.py")
    regular.pop("custom_reward_function")
    conditioned.pop("custom_reward_function")

    conditioned_data = conditioned["data"]
    assert conditioned_data.pop("conditioning_seed") == 42
    assert conditioned_data.pop("dataloader_num_workers") == 0
    assert conditioned_data.pop("custom_cls") == {
        "path": "src/eigrpo/conditioning/dataset.py",
        "name": "RewardConditionedDataset",
    }

    assert regular["trainer"].pop("default_local_dir").endswith("grpo_regular")
    assert conditioned["trainer"].pop("default_local_dir").endswith("grpo_conditioned")
    assert regular["trainer"].pop("experiment_name") == "regular-grpo"
    assert conditioned["trainer"].pop("experiment_name") == "conditioned-grpo"

    assert regular == conditioned
