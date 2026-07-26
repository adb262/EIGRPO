"""Collect failures and prepare a balanced GSM8K RC-SFT dataset."""

from __future__ import annotations

import argparse

from eigrpo.rc_grpo.sft_data import prepare_rc_sft_data
from eigrpo.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="data/rc_sft")
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    prepare_rc_sft_data(
        output_dir=args.output_dir,
        model=args.model,
        candidates_per_prompt=args.candidates_per_prompt,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_samples=args.max_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
