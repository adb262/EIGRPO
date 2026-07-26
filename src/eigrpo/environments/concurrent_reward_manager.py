"""verl reward manager that evaluates independent rollout rewards concurrently."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.abstract import AbstractRewardManager

from eigrpo.rc_grpo.prompts import reward_goal_from_messages


class ConcurrentRewardManager(AbstractRewardManager):
    """NaiveRewardManager-compatible implementation with bounded concurrency."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        judge_max_concurrency: int = 16,
    ) -> None:
        if judge_max_concurrency < 1:
            raise ValueError("judge_max_concurrency must be at least 1")
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.judge_max_concurrency = judge_max_concurrency

    def _decode_item(self, data_item) -> dict[str, Any]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("RC-GRPO reward requires raw_prompt from the async rollout")
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        extra_info["rc_reward_goal"] = reward_goal_from_messages(list(raw_prompt))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__")
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        return {
            "prompt_str": prompt_str,
            "response_str": self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
            "valid_response_length": valid_response_length,
            "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
            "data_source": data_item.non_tensor_batch[self.reward_fn_key],
            "extra_info": extra_info,
        }

    def _score_item(self, item: dict[str, Any]) -> float | dict[str, float]:
        return self.compute_score(
            data_source=item["data_source"],
            solution_str=item["response_str"],
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        items = [self._decode_item(data[i]) for i in range(len(data))]
        with ThreadPoolExecutor(max_workers=self.judge_max_concurrency) as executor:
            scores = list(executor.map(self._score_item, items))

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed: dict[str, int] = {}

        for index, (item, score) in enumerate(zip(items, scores, strict=True)):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_index = max(item["valid_response_length"] - 1, 0)
            reward_tensor[index, reward_index] = reward
            data_source = item["data_source"]
            already_printed.setdefault(data_source, 0)
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
