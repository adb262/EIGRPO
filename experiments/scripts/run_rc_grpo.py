"""Run RC-GRPO from a reward-conditioned SFT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from verl.trainer.main_ppo import run_ppo

from eigrpo.data.gsm8k import prepare as prepare_gsm8k
from eigrpo.experiments.grpo_runner import load_config
from eigrpo.rc_grpo.checkpoints import find_latest_hf_checkpoint
from eigrpo.rc_grpo.task_runner import RCGRPOTaskRunner
from eigrpo.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="RC-SFT Hugging Face path; defaults to latest --rc-sft-dir checkpoint")
    parser.add_argument("--rc-sft-dir", default="checkpoints/rc_sft")
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--config", default="configs/rc_grpo.yaml")
    args, overrides = parser.parse_known_args(argv)

    configure_logging()
    model = args.model or str(find_latest_hf_checkpoint(args.rc_sft_dir))
    data_dir = Path(args.data_dir)
    train_path = (data_dir / "train.parquet").resolve()
    val_path = (data_dir / "test.parquet").resolve()
    if not train_path.exists() or not val_path.exists():
        prepare_gsm8k(data_dir)

    config = load_config(
        args.config,
        model,
        [
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            *overrides,
        ],
    )
    task_runner = ray.remote(num_cpus=1)(RCGRPOTaskRunner)
    try:
        run_ppo(config, task_runner_class=task_runner)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
