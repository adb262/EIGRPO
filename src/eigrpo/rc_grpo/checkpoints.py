"""Checkpoint discovery for the two-stage RC-GRPO pipeline."""

from __future__ import annotations

from pathlib import Path


def find_latest_hf_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Return the newest verl SFT Hugging Face checkpoint."""
    candidates = list(Path(checkpoint_dir).glob("global_step_*/huggingface"))
    if not candidates:
        raise FileNotFoundError(f"No global_step_*/huggingface checkpoint found in {checkpoint_dir}")

    def step(path: Path) -> int:
        return int(path.parent.name.removeprefix("global_step_"))

    return max(candidates, key=step)
