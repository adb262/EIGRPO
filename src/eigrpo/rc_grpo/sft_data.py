"""Build balanced RC-SFT data from GSM8K experts and exploration failures."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from eigrpo.data.gsm8k import _format_example, extract_ground_truth
from eigrpo.environments.gsm8k_reward import compute_score
from eigrpo.rc_grpo.constants import HIGH_REWARD, LOW_REWARD
from eigrpo.rc_grpo.prompts import add_reward_goal_to_text

logger = logging.getLogger("eigrpo.rc_sft_data")

_FINAL_MARKER = re.compile(r"####\s*(-?[0-9.,]+)")
_XML_ANSWER = re.compile(r"<answer>\s*[^<]+\s*</answer>", re.DOTALL)


def format_expert_response(answer: str) -> str:
    """Convert GSM8K's expert answer to the experiment's XML answer contract."""
    ground_truth = extract_ground_truth(answer)
    without_marker = _FINAL_MARKER.sub("", answer).rstrip()
    return f"{without_marker}\n<answer>{ground_truth}</answer>"


def build_balanced_rows(
    examples: Iterable[dict],
    failure_candidates: Iterable[list[str]],
    split: str,
) -> list[dict[str, str]]:
    """Pair each usable exploration failure with the matching expert trajectory."""
    rows: list[dict[str, str]] = []
    for index, (example, candidates) in enumerate(zip(examples, failure_candidates, strict=True)):
        rl_row = _format_example(example, index, split)
        base_prompt = rl_row["prompt"][0]["content"]
        ground_truth = rl_row["reward_model"]["ground_truth"]
        failure = next(
            (
                candidate
                for candidate in candidates
                if _XML_ANSWER.search(candidate) and compute_score(candidate, ground_truth) == 0.0
            ),
            None,
        )
        if failure is None:
            continue

        rows.extend(
            [
                {
                    "prompt": add_reward_goal_to_text(base_prompt, HIGH_REWARD),
                    "response": format_expert_response(example["answer"]),
                    "reward_goal": HIGH_REWARD,
                    "task_reward": 1,
                    "source_index": index,
                },
                {
                    "prompt": add_reward_goal_to_text(base_prompt, LOW_REWARD),
                    "response": failure,
                    "reward_goal": LOW_REWARD,
                    "task_reward": 0,
                    "source_index": index,
                },
            ]
        )
    return rows


def _collect_failures(
    examples: list[dict],
    llm,
    candidates_per_prompt: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """Generate exploration candidates with vLLM and return them by prompt."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    prompts = []
    for index, example in enumerate(examples):
        row = _format_example(example, index, "exploration")
        prompts.append(
            tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    sampling_params = SamplingParams(
        n=candidates_per_prompt,
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [[candidate.text for candidate in request.outputs] for request in outputs]


def prepare_rc_sft_data(
    output_dir: str | Path,
    model: str,
    candidates_per_prompt: int = 4,
    max_new_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    max_samples: int | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create balanced train/test RC-SFT parquet files.

    For each GSM8K item, vLLM samples exploration trajectories. The first
    verified failure is paired with the benchmark's expert solution, yielding
    one high-reward and one low-reward example. Items without a sampled failure
    are skipped to preserve an exact 50/50 mixture.
    """
    from datasets import Dataset, load_dataset
    from vllm import LLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.parquet"
    test_path = output_path / "test.parquet"
    if not force and train_path.exists() and test_path.exists():
        logger.info("Reusing existing RC-SFT data in %s", output_path)
        return train_path, test_path

    dataset = load_dataset("openai/gsm8k", "main")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=42,
    )
    paths = {}
    for split, path in (("train", train_path), ("test", test_path)):
        examples = list(dataset[split])
        if max_samples is not None:
            examples = examples[:max_samples]
        candidates = _collect_failures(
            examples,
            llm=llm,
            candidates_per_prompt=candidates_per_prompt,
            max_new_tokens=max_new_tokens,
        )
        rows = build_balanced_rows(examples, candidates, split)
        if not rows:
            raise RuntimeError(f"No verified failures were generated for the {split} split")
        Dataset.from_list(rows).to_parquet(path)
        logger.info(
            "Wrote %d balanced RC-SFT rows (%d pairs) to %s",
            len(rows),
            len(rows) // 2,
            path,
        )
        paths[split] = path
    return paths["train"], paths["test"]
