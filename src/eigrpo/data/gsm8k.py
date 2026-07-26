"""Prepare GSM8K for verl's single-turn GRPO trainer."""

from __future__ import annotations

import re
from pathlib import Path

_FINAL_ANSWER_PATTERN = re.compile(r"####\s*(-?[0-9.,]+)")
_RESPONSE_INSTRUCTION = (
    "Solve the problem step by step. End with only the final numeric answer "
    "inside <answer>...</answer>."
)


def extract_ground_truth(answer: str) -> str:
    """Extract and normalize GSM8K's final numeric answer."""
    match = _FINAL_ANSWER_PATTERN.search(answer)
    if match is None:
        raise ValueError(f"GSM8K answer has no final answer marker: {answer!r}")
    return match.group(1).replace(",", "")


def _format_example(example: dict, index: int, split: str) -> dict:
    question = example["question"]
    answer = example["answer"]
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "user",
                "content": f"{question}\n\n{_RESPONSE_INSTRUCTION}",
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": extract_ground_truth(answer),
        },
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "answer": answer,
        },
    }


def prepare(output_dir: str | Path) -> tuple[Path, Path]:
    """Download GSM8K and write verl-compatible train/test parquet files."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.parquet"
    test_path = output_path / "test.parquet"

    dataset = load_dataset("openai/gsm8k", "main")
    train = dataset["train"].map(
        lambda example, index: _format_example(example, index, "train"),
        with_indices=True,
        remove_columns=dataset["train"].column_names,
    )
    test = dataset["test"].map(
        lambda example, index: _format_example(example, index, "test"),
        with_indices=True,
        remove_columns=dataset["test"].column_names,
    )
    train.to_parquet(train_path)
    test.to_parquet(test_path)
    return train_path, test_path
