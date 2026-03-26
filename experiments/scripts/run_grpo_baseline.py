"""Entry-point for running EIGRPO training on GSM8K.

Usage
-----
    python experiments/scripts/run_grpo_baseline.py \
        --model Qwen/Qwen2.5-0.5B \
        --config configs/base.yaml

Override any config value with dotlist syntax appended as extra args:

    python experiments/scripts/run_grpo_baseline.py \
        --model Qwen/Qwen2.5-0.5B \
        eigrpo.selector.name=eig \
        trainer.n_gpus_per_node=4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ray
import verl.trainer.main_ppo as _mod
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, flag_override
from verl.trainer.main_ppo import run_ppo

from eigrpo.data.gsm8k import prepare as prepare_gsm8k_data
from eigrpo.trainers.task_runner import EIGRPOTaskRunner
from eigrpo.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger("eigrpo.run")


def _verl_config_dir() -> str:
    """Locate verl's Hydra config directory at runtime."""

    return str(Path(_mod.__file__).resolve().parent / "config")


def load_config(config_path: str, model: str, overrides: list[str]) -> OmegaConf:
    """Build the full verl default config via Hydra, then layer user overrides."""
    verl_cfg_dir = _verl_config_dir()
    with initialize_config_dir(config_dir=verl_cfg_dir, version_base=None):
        base = compose(config_name="ppo_trainer")

    user = OmegaConf.load(config_path)
    with flag_override(base, "struct", False):
        # This just means that we can have keys that don't exist in the base config
        config = OmegaConf.merge(base, user)

    model_override = OmegaConf.create({"actor_rollout_ref": {"model": {"path": model}}})
    with flag_override(config, "struct", False):
        config = OmegaConf.merge(config, model_override)

        if overrides:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run EIGRPO / GRPO training")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the YAML config file")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/gsm8k",
        help="Directory for prepared GSM8K parquet files (auto-created if missing)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Experiment name used for logging and checkpoint isolation. "
            "Sets trainer.experiment_name and trainer.default_local_dir=./checkpoints/<name>. "
            "Defaults to the value in the config file."
        ),
    )
    parser.add_argument(
        "--resume-mode",
        type=str,
        choices=["auto", "disable"],
        default=None,
        help=(
            "Checkpoint resume behaviour. 'auto' resumes from the latest checkpoint if one "
            "exists; 'disable' always starts from scratch. "
            "Defaults to the value in the config file (currently 'auto')."
        ),
    )
    args, overrides = parser.parse_known_args(argv)
    logger.info("Starting EIGRPO training...")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    train_parquet = Path(args.data_dir) / "train.parquet"
    val_parquet = Path(args.data_dir) / "test.parquet"
    if not train_parquet.exists() or not val_parquet.exists():
        logger.info("Data not found at %s — preparing GSM8K from HuggingFace...", args.data_dir)
        prepare_gsm8k_data(args.data_dir)

    overrides += [
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
    ]

    if args.experiment_name is not None:
        overrides += [
            f"trainer.experiment_name={args.experiment_name}",
            f"trainer.default_local_dir=./checkpoints/{args.experiment_name}",
        ]

    if args.resume_mode is not None:
        overrides += [f"trainer.resume_mode={args.resume_mode}"]

    config = load_config(str(config_path), args.model, overrides)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(EIGRPOTaskRunner))


if __name__ == "__main__":
    main()
