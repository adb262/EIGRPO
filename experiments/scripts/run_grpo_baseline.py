"""Run either arm of the vanilla GRPO reward-conditioning comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from eigrpo.experiments.grpo_runner import run_experiment
from eigrpo.utils.logging import configure_logging

_CONFIGS = {
    "regular": Path("configs/grpo_regular.yaml"),
    "conditioned": Path("configs/grpo_conditioned.yaml"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run regular or reward-conditioned GRPO")
    parser.add_argument("--variant", choices=sorted(_CONFIGS), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", default="data/gsm8k")
    args, overrides = parser.parse_known_args(argv)

    configure_logging()
    run_experiment(_CONFIGS[args.variant], args.model, args.data_dir, overrides)


if __name__ == "__main__":
    main()
