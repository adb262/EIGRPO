"""Extended GRPO trainer with pluggable trajectory selection.

Subclasses verl's ``RayPPOTrainer`` and injects the EIGRPO selector
between reward computation and advantage estimation.  This is the
mechanism that lets us decouple ``num_rollouts`` (cheap generation)
from the effective sample size (expensive backward pass).
"""

from __future__ import annotations

import importlib
import logging
import time
import uuid
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward_async
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode, compute_rollout_correction_and_add_to_batch
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking

from eigrpo.config import EIGRPOConfig, parse_eigrpo_config
from eigrpo.diversity.metrics import compute_unique_trajectory_ratio
from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch
from eigrpo.trainers.rollout_batch import RolloutBatch
from eigrpo.utils.metrics import compute_entropy, compute_mastery_pct, compute_min_log_prob

logger = logging.getLogger("eigrpo.trainer")

_SELECTOR_REGISTRY: dict[str, str] = {
    "uniform": "eigrpo.sampling.uniform.UniformSelector",
    "eig": "eigrpo.sampling.eig_selector.EIGSelector",
}


def _build_selector(config: EIGRPOConfig) -> BaseTrajectorySelector:
    """Instantiate a trajectory selector from the typed EIGRPO config."""
    name = config.selector.name
    cls_path = _SELECTOR_REGISTRY.get(name, name)
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls()


class EIGRPOTrainer(RayPPOTrainer):
    """RayPPOTrainer extended with trajectory selection and diversity logging.

    Overrides ``fit()`` to inject ``_apply_selection`` between reward
    computation and advantage estimation, and merges EIGRPO metrics
    (entropy, min-log-prob, mastery, selection stats) into verl's
    unified metrics pipeline.

    Gradient norm is already logged by verl's actor worker as
    ``actor/grad_norm`` — no extra code needed.
    """

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, **kwargs):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, **kwargs)

        self._eigrpo_config: EIGRPOConfig = parse_eigrpo_config(config)
        self._selector = _build_selector(self._eigrpo_config)

        logger.info(
            "EIGRPOTrainer initialised — selector=%s  n_effective_samples=%s",
            type(self._selector).__name__,
            self._eigrpo_config.selector.n_effective_samples,
        )

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def _collect_rollout_metrics(self, rollout: RolloutBatch) -> dict[str, float]:
        """Gather all rollout-level diagnostics into a single flat dict."""
        metrics: dict[str, float] = {}

        if rollout.old_log_probs is not None:
            mask = rollout.response_mask
            metrics.update(compute_entropy(rollout.old_log_probs, mask))
            metrics.update(compute_min_log_prob(rollout.old_log_probs, mask))

        rewards = rollout.token_level_rewards.sum(dim=-1)
        metrics.update(compute_mastery_pct(rewards, rollout.uids))
        metrics["reward/pre_mean"] = rewards.mean().item()
        metrics["reward/pre_min"] = rewards.min().item()
        metrics["reward/pre_max"] = rewards.max().item()

        return metrics

    # ------------------------------------------------------------------
    # Rollout table logging
    # ------------------------------------------------------------------

    def _log_selection_table(
        self,
        batch: DataProto,
        selected: list[int],
        step: int,
    ) -> None:
        """Log a wandb Table showing every rollout and whether it was selected.

        Uses ``_log_group_id`` (stamped onto the batch before ``_balance_batch``
        reorders rows by sequence length) to recover the correct prompt for each
        row.  Without this, ``prompts[p * G]`` after reordering gives a row from
        an arbitrary group, producing a prompt/response mismatch in the table.

        Skips silently when wandb is not active or the frequency guard fires.
        """
        freq = self._eigrpo_config.logging.rollout_table_freq
        if freq <= 0 or step % freq != 0:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        n_total = batch.batch.batch_size[0]
        max_groups = self._eigrpo_config.logging.rollout_table_max_groups

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        scores = batch.batch["token_level_scores"].sum(-1)
        selected_set = set(selected)

        # _log_group_id[i] = original prompt-group that row i belongs to.
        # Stamped before _balance_batch so it is reordered in lock-step with
        # the tensor rows, letting us recover correct group membership after sort.
        group_ids = batch.batch.get("_log_group_id")

        columns = ["step", "group", "rollout", "prompt", "response", "score", "selected"]
        table = wandb.Table(columns=columns)

        if group_ids is not None:
            unique_groups = group_ids.unique().tolist()
            for gid in unique_groups[:max_groups]:
                row_indices = (group_ids == gid).nonzero(as_tuple=True)[0].tolist()
                prompt_str = self.tokenizer.decode(prompts[row_indices[0]], skip_special_tokens=True)
                for r, idx in enumerate(row_indices):
                    response_str = self.tokenizer.decode(responses[idx], skip_special_tokens=True)
                    table.add_data(step, int(gid), r, prompt_str, response_str, scores[idx].item(), idx in selected_set)
        else:
            # Fallback when _log_group_id is absent (balance_batch=False):
            # rows are still in original group order so p * G is valid.
            G = self.config.actor_rollout_ref.rollout.n
            P = n_total // G
            for p in range(min(P, max_groups)):
                prompt_str = self.tokenizer.decode(prompts[p * G], skip_special_tokens=True)
                for r in range(G):
                    idx = p * G + r
                    response_str = self.tokenizer.decode(responses[idx], skip_special_tokens=True)
                    table.add_data(step, p, r, prompt_str, response_str, scores[idx].item(), idx in selected_set)

        wandb.log({"eigrpo/selection_table": table}, step=step)

    # ------------------------------------------------------------------
    # Selection hook
    # ------------------------------------------------------------------

    def _extract_pre_jacobian_metrics(self, old_log_prob: DataProto) -> dict[str, float]:
        """Extract pre-selection Jacobian rank from ``compute_log_prob`` output.

        ``EIGRPOActorRolloutRefWorker.compute_log_prob`` stores gradient-sketch
        rank metrics in ``meta_info["jacobian_pre_metrics"]``.  Each FSDP
        worker has already all_reduced its sketch, so all workers return
        identical dicts — we just take the value directly (no further
        aggregation needed).
        """
        return old_log_prob.meta_info.get("jacobian_pre_metrics", {})

    def _apply_selection(
        self,
        batch: DataProto,
        step: int,
        pre_jacobian_metrics: dict[str, float] | None = None,
    ) -> tuple[DataProto, dict[str, float]]:
        """Filter the rollout batch and return EIGRPO metrics.

        Returns the (possibly filtered) batch and a flat metrics dict.
        Selection is a no-op when ``n_effective_samples`` is ``None``
        or >= the batch size.

        ``pre_jacobian_metrics``: gradient-sketch rank computed during
        ``compute_log_prob`` (passed in from the fit loop so we don't
        repeat the forward pass).
        """
        rollout = RolloutBatch(batch)
        n_total = rollout.size
        metrics: dict[str, float] = self._collect_rollout_metrics(rollout)
        metrics["n_rollouts"] = float(n_total)

        rewards = rollout.token_level_rewards.sum(dim=-1)

        if pre_jacobian_metrics:
            metrics.update({f"jacobian/pre_{k}": v for k, v in pre_jacobian_metrics.items()})

        n_select = self._eigrpo_config.selector.n_effective_samples
        if n_select is None or n_select >= n_total:
            metrics["effective_batch_size"] = float(n_total)
            metrics["reward/post_mean"] = metrics["reward/pre_mean"]
            metrics["reward/post_min"] = metrics["reward/pre_min"]
            metrics["reward/post_max"] = metrics["reward/pre_max"]
            self._log_selection_table(batch, list(range(n_total)), step)
            return batch, metrics

        trajectories: list[list[dict]] = [[] for _ in range(n_total)]
        response_texts: list[str] | None = None
        token_sequences: list[list[int]] | None = None
        if rollout.responses is not None:
            token_sequences = rollout.responses.tolist()
            response_texts = self.tokenizer.batch_decode(rollout.responses, skip_special_tokens=True)

        tb = TrajectoryBatch(
            rollout_ids=rollout.uids,
            trajectories=trajectories,
            rewards=rewards,
            group_size=self.config.actor_rollout_ref.rollout.n,
            token_sequences=token_sequences,
            log_probs=rollout.old_log_probs,
            response_texts=response_texts,
        )

        _t0 = time.perf_counter()
        selected = self._selector.select(tb, n_select)
        metrics["selection/embed_time_s"] = time.perf_counter() - _t0
        self._log_selection_table(batch, selected, step)

        metrics["selection/n_total_rollouts"] = float(n_total)
        metrics["selection/n_selected"] = float(len(selected))
        metrics["effective_batch_size"] = float(len(selected))

        if self._eigrpo_config.diversity.log_trajectory_similarity and token_sequences is not None:
            metrics["diversity/unique_trajectory_ratio"] = compute_unique_trajectory_ratio(token_sequences)

        selected_t = torch.tensor(selected, dtype=torch.long)
        filtered_batch = rollout.select_indices(selected_t)

        filtered_rollout = RolloutBatch(filtered_batch)
        post_rewards = filtered_rollout.token_level_rewards.sum(dim=-1)
        metrics["reward/post_mean"] = post_rewards.mean().item()
        metrics["reward/post_min"] = post_rewards.min().item()
        metrics["reward/post_max"] = post_rewards.max().item()

        return filtered_batch, metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        """verl's training loop with EIGRPO selection injected.

        Kept in sync with ``RayPPOTrainer.fit()`` (verl) with one addition:
        after ``token_level_rewards`` is populated and before
        ``compute_advantage``, we call ``_apply_selection`` and merge
        the returned metrics into verl's ``metrics`` dict.
        """

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, float] = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)

                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch,
                                reward_fn=self.reward_fn,
                                sum_reward=True,
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    _log_n = batch.batch["input_ids"].shape[0]
                    _log_G = self.config.actor_rollout_ref.rollout.n
                    batch.batch["_log_group_id"] = torch.arange(_log_n) // _log_G
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch,
                                config=self.config,
                                tokenizer=self.tokenizer,
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch,
                                reward_fn=self.reward_fn,
                                return_dict=False,
                            )

                    # Compute per-group reward statistics from all pre-selection rollouts
                    # so both the pre- and post-selection Jacobian sketches can use the
                    # same (μ_g, σ_g) baseline for advantage normalisation.  Only the
                    # synchronous reward path has reward_tensor available here; in the
                    # async path we skip and the actor falls back to unweighted loss for
                    # both sketches, keeping them consistent with each other.
                    if not self.config.reward_model.launch_reward_fn_async:
                        _gs = self.config.actor_rollout_ref.rollout.n  # K pre-selection
                        _per_resp = reward_tensor.sum(-1)               # (N,) terminal reward
                        _P = _per_resp.shape[0] // _gs
                        _r = _per_resp[: _P * _gs].view(_P, _gs)
                        _mu = _r.mean(dim=1)                            # (P,)
                        _sigma = _r.std(dim=1).clamp(min=1e-8)          # (P,)
                        _mu_exp = _mu.unsqueeze(1).expand(_P, _gs).reshape(-1)
                        _sig_exp = _sigma.unsqueeze(1).expand(_P, _gs).reshape(-1)
                        batch.meta_info["pre_sketch_advantages"] = (_per_resp - _mu_exp) / _sig_exp
                        batch.meta_info["pre_group_reward_mu"] = _mu
                        batch.meta_info["pre_group_reward_sigma"] = _sigma

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            # Extract pre-selection Jacobian metrics piggy-backed by
                            # EIGRPOActor.compute_log_prob (gradient sketch on the full
                            # pre-selection batch, zero extra forward passes).
                            pre_jacobian_metrics = self._extract_pre_jacobian_metrics(old_log_prob)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_probs" not in {batch.batch.keys()=}'
                    if bypass_recomputing_logprobs:
                        pre_jacobian_metrics: dict[str, float] = {}

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # ── EIGRPO: selection + metrics ──────────────
                        with marked_timer("selection", timing_raw):
                            batch, eigrpo_metrics = self._apply_selection(
                                batch, self.global_steps, pre_jacobian_metrics
                            )
                        metrics.update({f"eigrpo/{k}": v for k, v in eigrpo_metrics.items()})

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        # After selection each prompt has n_effective_samples rollouts;
                        # fall back to rollout.n when selection is disabled.
                        n_after = (
                            self._eigrpo_config.selector.n_effective_samples or self.config.actor_rollout_ref.rollout.n
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n_after,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # n_after = rollouts per prompt post-selection; EIGRPOActor
                        # uses this to slice the batch into per-group sketch passes.
                        batch.meta_info["group_size_post"] = n_after
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # rank retention: how much gradient diversity EIG preserved
                        # effective_rank_frac is the primary metric (hard-threshold rank
                        # always saturates at n_samples for large models and is uninformative).
                        pre_eff = metrics.get("jacobian/pre_effective_rank_frac")
                        post_eff = metrics.get("jacobian/post_effective_rank_frac")
                        if pre_eff is not None and post_eff is not None and pre_eff > 0:
                            metrics["jacobian/rank_retention"] = post_eff / pre_eff

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                tracking_logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
