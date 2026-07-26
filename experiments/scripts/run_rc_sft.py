"""Prepare RC-SFT data and train the reward-conditioned trajectory policy."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from eigrpo.rc_grpo.checkpoints import find_latest_hf_checkpoint
from eigrpo.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model used for failure collection and RC-SFT")
    parser.add_argument("--data-dir", default="data/rc_sft")
    parser.add_argument("--output-dir", default="checkpoints/rc_sft")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--collection-tp", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--force-data-prep", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    configure_logging()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    subprocess_env = os.environ.copy()
    subprocess_env["MAX_JOBS"] = "4"
    if not args.skip_data_prep:
        prepare_command = [
            sys.executable,
            str(Path(__file__).with_name("prepare_rc_sft_data.py")),
            "--model",
            args.model,
            "--output-dir",
            str(data_dir),
            "--candidates-per-prompt",
            str(args.candidates_per_prompt),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--tensor-parallel-size",
            str(args.collection_tp),
        ]
        if args.max_samples is not None:
            prepare_command.extend(["--max-samples", str(args.max_samples)])
        if args.force_data_prep:
            prepare_command.append("--force")
        # Isolate offline vLLM collection so its CUDA context is released
        # before torchrun starts the two-GPU SFT job.
        subprocess.run(prepare_command, check=True, env=subprocess_env)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "test.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"RC-SFT data not found in {data_dir}")

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is required to launch RC-SFT")

    command = [
        torchrun,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        "data.prompt_key=prompt",
        "data.response_key=response",
        "data.train_batch_size=16",
        "data.micro_batch_size_per_gpu=2",
        "data.max_length=3072",
        "data.truncation=right",
        f"model.partial_pretrain={args.model}",
        "model.enable_gradient_checkpointing=true",
        "model.fsdp_config.model_dtype=bfloat16",
        "optim.lr=1e-5",
        "trainer.total_epochs=1",
        f"trainer.n_gpus_per_node={args.gpus}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.project_name=grpo-reward-conditioning",
        "trainer.experiment_name=rc-sft",
        "trainer.logger=[console,wandb]",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        "trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "trainer.checkpoint.load_contents=[model,optimizer,extra]",
        "use_remove_padding=true",
        *overrides,
    ]
    subprocess.run(command, check=True, env=subprocess_env)
    checkpoint = find_latest_hf_checkpoint(output_dir)
    print(f"RC-SFT Hugging Face checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
