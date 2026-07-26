"""RayPPOTrainer extension that conditions each rollout independently."""

from __future__ import annotations

import random

import numpy as np
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from eigrpo.rc_grpo.prompts import condition_rollout_prompts


class RCGRPOTrainer(RayPPOTrainer):
    """RC-GRPO with per-trajectory reward-token sampling.

    Only rollout prompts change. Rewards and GRPO advantages remain ordinary
    task-success rewards, including for low-reward-conditioned trajectories.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rc_config = self.config.get("rc_grpo", {})
        self._high_reward_probability = float(rc_config.get("high_reward_probability", 0.5))
        self._reward_goal_seed = int(rc_config.get("seed", 42))

    def _condition_generation_batch(self, batch):
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError("RC-GRPO requires raw_prompt in the rollout batch")

        is_validation = bool(batch.meta_info.get("validate", False))
        global_step = int(batch.meta_info.get("global_steps", 0))
        step_rng = random.Random(self._reward_goal_seed + global_step)
        conditioned_prompts, reward_goals = condition_rollout_prompts(
            [list(raw_prompt) for raw_prompt in raw_prompts],
            rng=step_rng,
            high_reward_probability=self._high_reward_probability,
            is_validation=is_validation,
        )

        prompt_array = np.empty(len(conditioned_prompts), dtype=object)
        prompt_array[:] = conditioned_prompts
        batch.non_tensor_batch["raw_prompt"] = prompt_array
        batch.non_tensor_batch["rc_reward_goal"] = np.array(reward_goals, dtype=object)
        return batch

    def fit(self):
        if not self.async_rollout_mode:
            raise ValueError("RC-GRPO currently requires actor_rollout_ref.rollout.mode=async")

        original_generate = self.async_rollout_manager.generate_sequences

        def generate_conditioned(batch):
            return original_generate(self._condition_generation_batch(batch))

        self.async_rollout_manager.generate_sequences = generate_conditioned
        try:
            return super().fit()
        finally:
            self.async_rollout_manager.generate_sequences = original_generate
