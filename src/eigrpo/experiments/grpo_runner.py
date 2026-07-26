"""Shared vanilla-verl runner for the GRPO comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import ray
import verl.trainer.main_ppo as verl_main_ppo
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, flag_override
from verl.trainer.main_ppo import run_ppo

from eigrpo.data.gsm8k import prepare as prepare_gsm8k

logger = logging.getLogger("eigrpo.grpo")


def load_config(config_path: str | Path, model: str, overrides: list[str]) -> OmegaConf:
    """Compose verl defaults, then apply the experiment and CLI overrides."""
    verl_config_dir = str(Path(verl_main_ppo.__file__).resolve().parent / "config")
    with initialize_config_dir(config_dir=verl_config_dir, version_base=None):
        base = compose(config_name="ppo_trainer")

    user = OmegaConf.load(config_path)
    with flag_override(base, "struct", False):
        config = OmegaConf.merge(
            base,
            user,
            OmegaConf.create({"actor_rollout_ref": {"model": {"path": model}}}),
            OmegaConf.from_dotlist(overrides),
        )
    return config


def run_experiment(
    config_path: str | Path,
    model: str,
    data_dir: str | Path,
    overrides: list[str],
) -> None:
    """Prepare data if needed and launch unmodified verl GRPO."""
    data_path = Path(data_dir)
    train_path = (data_path / "train.parquet").resolve()
    val_path = (data_path / "test.parquet").resolve()
    if not train_path.exists() or not val_path.exists():
        logger.info("Preparing GSM8K in %s", data_path)
        prepare_gsm8k(data_path)

    data_overrides = [
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        *overrides,
    ]
    config = load_config(config_path, model, data_overrides)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    # Deliberately use verl's default TaskRunner/RayPPOTrainer. No EIGRPO
    # trainer, selector, replay, or SFT stage participates in this experiment.
    try:
        run_ppo(config)
    finally:
        ray.shutdown()
