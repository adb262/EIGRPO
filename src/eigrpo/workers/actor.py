"""EIGRPO-aware PPO actor with two-phase per-rollout Jacobian rank estimation.

Pre-selection rank
    Hooked into ``compute_log_prob``.  For each prompt group g (rows
    [g*K_pre, (g+1)*K_pre)), we do K_pre independent forward+backward passes —
    one per rollout — projecting each rollout's gradient onto SKETCH_DIM random
    directions.  This gives K_pre sketch rows per group, so the per-group sketch
    is (K_pre, SKETCH_DIM) with effective rank bounded by min(K_pre, SKETCH_DIM).
    Sketch rows are all-gathered across DP ranks and stored in
    ``self._pre_jacobian_metrics``.

    Each forward+backward is a self-contained FSDP cycle (no retain_graph), which
    is required for correctness with ``use_orig_params=True`` and tied weights
    (embed_tokens ↔ lm_head): retain_graph would hold references to gathered
    parameter storage that FSDP frees on reshard, causing "storage size of 0".

Post-selection rank
    Hooked into ``update_policy``.  Same technique but with K_post rollouts per
    group (post-EIG selection), so effective rank is bounded by
    min(K_post, SKETCH_DIM).  Returned in the metrics dict under
    ``jacobian/post_*``.

Extra forward passes: G_pre*K_pre + G_post*K_post (one per rollout per phase).
Extra backward passes: G_pre*K_pre + G_post*K_post.
effective_rank is normalised by min(K, SKETCH_DIM) per phase so
effective_rank_frac is interpretable within each phase independently.
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.distributed as dist
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger("eigrpo.workers.actor")

SKETCH_DIM = 128
SKETCH_SEED = 42
JACOBIAN_RANK_ENABLED = False  # set True to re-enable per-rollout gradient sketch metrics


def _project_local_grads(
    params: list[torch.nn.Parameter],
    D: int,
    seed: int,
    chunk_size: int = 262144,
) -> torch.Tensor:
    """Project the current .grad of each parameter onto D random directions.

    All computation stays on the GPU where the parameters live.  Only the
    final D-dimensional result is moved to CPU.  Memory cost per call:
    O(chunk_size * D * 4 bytes) — about 128 MB at the defaults.

    Uses a seeded GPU generator reset for every call so that all groups
    in a sketch use the same random projection matrix.
    """
    device = next((p.device for p in params if p.grad is not None), None)
    if device is None:
        return torch.zeros(D, dtype=torch.float32)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    dot = torch.zeros(D, dtype=torch.float32, device=device)

    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float().flatten()   # stays on GPU
        n = g.shape[0]
        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            R = torch.randn(end - offset, D, generator=rng, dtype=torch.float32, device=device)
            dot += g[offset:end] @ R             # (D,) accumulated on GPU

    return dot.cpu()                             # only D=128 floats cross PCIe


def _compute_rank_from_sketch(sketch: torch.Tensor) -> dict[str, float]:
    """Compute entropy-based effective rank from a (P, G, SKETCH_DIM) sketch tensor.

    For each of the P prompt-groups, runs SVD on the (G, SKETCH_DIM) per-group
    sub-matrix and computes the Roy-Vetterli (2007) effective rank as
    exp(H(sigma / ||sigma||_1)).  Returns the mean across groups.

    ``effective_rank_frac`` is normalised by ``min(G, SKETCH_DIM)`` — the
    ceiling for a group of G rollouts.  Pre- and post-selection sketches have
    different G values (K_pre vs K_post) so the fracs are not directly
    comparable in magnitude, but rank_retention = post_frac / pre_frac still
    measures how well EIG selection preserved within-group gradient diversity.
    """
    P, G, D = sketch.shape
    ceiling = min(G, D)
    eff_ranks = []
    for p in range(P):
        sv = torch.linalg.svdvals(sketch[p].float())   # (min(G, D),)
        sv_norm = sv / sv.sum().clamp(min=1e-12)
        h = -(sv_norm * (sv_norm + 1e-10).log()).sum()
        eff_ranks.append(float(h.exp().item()))
    mean_eff_rank = sum(eff_ranks) / len(eff_ranks)
    return {
        "effective_rank": mean_eff_rank,
        "effective_rank_frac": mean_eff_rank / ceiling,
        "n_groups": float(P),
    }


def _collect_sketch_group_passes(
    actor_module,
    model_inputs_full: dict,
    group_size: int,
    params: list,
    sketch: torch.Tensor,
    forward_fn,
    temperature: float,
    advantages: torch.Tensor | None = None,
) -> int:
    """Collect one gradient sketch row per rollout via independent forward+backward passes.

    For each group g and rollout k, a dedicated forward+backward pass runs on the
    single-row slice.  The resulting gradient is projected onto SKETCH_DIM random
    directions and stored as row g*group_size+k of the sketch.

    No retain_graph is used.  Each (forward, backward) pair is a complete FSDP
    cycle: parameters are gathered, used, then resharded cleanly.  This avoids
    the "storage size of 0" error that arises with retain_graph when FSDP frees
    gathered parameter storage (especially for tied embed_tokens/lm_head weights)
    between the shared forward and a subsequent backward.

    Cost: G x K x (1 forward + 1 backward) passes total.

    Args:
        actor_module: FSDP-wrapped model (used for no_sync / zero_grad).
        model_inputs_full: dict of tensors with shape (N, ...) where
            N = G * group_size.  Sliced into single-row rollout views.
        group_size: number of rollouts per prompt group (K_pre or K_post).
        params: trainable parameters to project gradients from.
        sketch: pre-allocated (G * group_size, SKETCH_DIM) tensor to write into.
        forward_fn: ``self._forward_micro_batch``.
        temperature: generation temperature for the forward pass.
        advantages: optional (N,) per-response advantage scalars.  When
            provided the per-rollout loss is: -(adv[i] * lp[i]).
            When None: -lp[i].

    Returns:
        Number of groups where all K rollouts were successfully sketched.
    """
    N = model_inputs_full["input_ids"].shape[0]
    num_groups = N // group_size

    def _no_sync():
        if hasattr(actor_module, "no_sync"):
            return actor_module.no_sync()
        return contextlib.nullcontext()

    completed = 0
    try:
        for g in range(num_groups):
            s = g * group_size
            for k in range(group_size):
                idx = s + k
                rollout_inputs = {key: val[idx : idx + 1] for key, val in model_inputs_full.items()}
                actor_module.zero_grad()
                with torch.enable_grad(), _no_sync():
                    _, lp = forward_fn(rollout_inputs, temperature=temperature, calculate_entropy=False)
                    lp_scalar = lp.sum() if lp.dim() > 0 else lp
                    if advantages is not None:
                        loss_k = -(advantages[idx] * lp_scalar)
                    else:
                        loss_k = -lp_scalar
                    loss_k.backward()
                sketch[idx] = _project_local_grads(params, SKETCH_DIM, SKETCH_SEED)
            completed += 1
    except Exception:
        logger.warning("Jacobian group sketch failed", exc_info=True)
    finally:
        actor_module.zero_grad()

    return completed


def _finalise_sketch(
    sketch: torch.Tensor,
    completed_groups: int,
    device,
    group_size: int,
) -> dict[str, float]:
    """All-gather sketch rows across DP ranks, reshape, and compute rank metrics.

    Each DP rank holds sketch rows for its local prompt-groups.  We all-gather
    (concatenate, not sum) across DP ranks to assemble the full
    (P_total * group_size, SKETCH_DIM) matrix, then reshape to
    (P_total, group_size, SKETCH_DIM) before computing per-group SVD.

    Note: assumes TP=1 (no within-replica parameter sharding), so each rank's
    gradient projections are complete.  With FSDP parameter sharding across DP
    replicas the rows would be partial projections requiring an additional
    within-shard all_reduce before all_gather.
    """
    if completed_groups == 0:
        return {}
    local_rows = completed_groups * group_size
    used = sketch[:local_rows].to(device)           # (local_rows, SKETCH_DIM)
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(used) for _ in range(world_size)]
        dist.all_gather(gathered, used)
        full_sketch = torch.cat(gathered, dim=0)    # (total_rows, SKETCH_DIM)
    else:
        full_sketch = used
    total_rows = full_sketch.shape[0]
    P_total = total_rows // group_size
    if P_total == 0:
        return {}
    shaped = full_sketch.cpu().reshape(P_total, group_size, SKETCH_DIM)
    return _compute_rank_from_sketch(shaped)


class EIGRPOActor(DataParallelPPOActor):
    """DataParallelPPOActor that collects per-rollout gradient Jacobian sketches.

    For each prompt-group, K rollouts each contribute one row to a per-group
    (K, SKETCH_DIM) sketch matrix.  The effective rank of that matrix measures
    within-group gradient diversity.  Mean effective rank is reported across
    all prompt-groups.

    Pre-selection rank: hooked into ``compute_log_prob`` (actor path only).
    Post-selection rank: hooked into ``update_policy`` using advantage-weighted
    per-rollout losses so the gradient reflects EIG's actual training signal.
    """

    def _sketch_model_inputs(self, data, temperature):
        """Build a flat dict of forward-pass tensors from data.batch.

        Returns (model_inputs, N, params, device).
        """
        params = [p for p in self.actor_module.parameters() if p.requires_grad]
        device = next(iter(params)).device
        want = ["input_ids", "attention_mask", "position_ids", "responses"]
        model_inputs = {
            k: data.batch[k].to(device)
            for k in want
            if k in data.batch.keys()
        }
        N = next(iter(model_inputs.values())).shape[0]
        return model_inputs, N, params, device

    # ── pre-selection rank ────────────────────────────────────────────────

    def compute_log_prob(self, data, calculate_entropy=False):
        """Like parent but hooks per-group gradient sketching for the actor path.

        When ``calculate_entropy=True`` (actor, not ref policy):
        after collecting log_probs under no_grad, performs one
        forward+backward per prompt-group and stores rank metrics in
        ``self._pre_jacobian_metrics``.

        When ``calculate_entropy=False`` (ref policy path) the sketch is
        skipped to avoid corrupting the pre-selection metric.
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_mm = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_keys = ["multi_modal_inputs"] if has_mm else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
            batch_idx_list = None

        # ── normal log_prob collection (no_grad) ─────────────────────────
        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy and entropy is not None:
                entropy_lst.append(entropy)

        log_probs_out = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz and batch_idx_list is not None:
            log_probs_out = restore_dynamic_batch(log_probs_out, batch_idx_list)
            if calculate_entropy and entropys is not None:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        # ── pre-selection per-group Jacobian sketch (actor path only) ─────
        # Disabled when JACOBIAN_RANK_ENABLED=False to keep training fast.
        self._pre_jacobian_metrics = {}
        if JACOBIAN_RANK_ENABLED and calculate_entropy:
            group_size = data.meta_info.get("group_size")
            if group_size is None:
                logger.warning("group_size missing from meta_info; skipping pre-sketch")
            else:
                self.actor_module.train()
                model_inputs_all, N, params, device = self._sketch_model_inputs(data, temperature)
                G = N // group_size
                sketch = torch.zeros(G * group_size, SKETCH_DIM, dtype=torch.float32)
                pre_adv = data.meta_info.get("pre_sketch_advantages")
                if pre_adv is not None:
                    pre_adv = pre_adv.to(device)
                completed = _collect_sketch_group_passes(
                    actor_module=self.actor_module,
                    model_inputs_full=model_inputs_all,
                    group_size=group_size,
                    params=params,
                    sketch=sketch,
                    forward_fn=self._forward_micro_batch,
                    temperature=temperature,
                    advantages=pre_adv,
                )
                self.actor_module.eval()
                self._pre_jacobian_metrics = _finalise_sketch(sketch, completed, device, group_size)

        return log_probs_out, entropys

    # ── post-selection rank ───────────────────────────────────────────────

    def update_policy(self, data: DataProto) -> dict:  # type: ignore[override]
        self.actor_module.train()
        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses", "response_mask", "input_ids",
            "attention_mask", "position_ids", "old_log_probs", "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if "token_level_scores" in data.batch.keys():
            select_keys.append("token_level_scores")

        has_mm = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_keys = ["multi_modal_inputs"] if has_mm else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics: dict = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}

        # ── post-selection per-group Jacobian sketch ──────────────────────
        # Disabled when JACOBIAN_RANK_ENABLED=False to keep training fast.
        post_sketch_data = None
        if JACOBIAN_RANK_ENABLED:
            group_size_post = data.meta_info.get("group_size_post")
            if group_size_post is None:
                logger.warning("group_size_post missing from meta_info; skipping post-sketch")
            else:
                model_inputs_all, N, params, device = self._sketch_model_inputs(data, temperature)
                G = N // group_size_post
                sketch = torch.zeros(G * group_size_post, SKETCH_DIM, dtype=torch.float32)
                pre_mu = data.meta_info.get("pre_group_reward_mu")
                pre_sigma = data.meta_info.get("pre_group_reward_sigma")
                if (pre_mu is not None and pre_sigma is not None
                        and "token_level_scores" in data.batch.keys()):
                    per_resp_rewards = data.batch["token_level_scores"].sum(-1)
                    mu_exp = pre_mu[:G].unsqueeze(1).expand(G, group_size_post).reshape(-1).to(device)
                    sig_exp = pre_sigma[:G].unsqueeze(1).expand(G, group_size_post).reshape(-1).to(device)
                    adv_per_response = (per_resp_rewards.to(device) - mu_exp) / sig_exp
                else:
                    token_adv = data.batch["advantages"]
                    response_mask = data.batch["response_mask"]
                    adv_per_response = (token_adv * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1)
                    adv_per_response = adv_per_response.to(device)
                completed = _collect_sketch_group_passes(
                    actor_module=self.actor_module,
                    model_inputs_full=model_inputs_all,
                    group_size=group_size_post,
                    params=params,
                    sketch=sketch,
                    forward_fn=self._forward_micro_batch,
                    temperature=temperature,
                    advantages=adv_per_response,
                )
                self.actor_module.train()
                post_sketch_data = (sketch, completed, device, group_size_post)

        # ── normal training loop ──────────────────────────────────────────
        for epoch_idx in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    mb_metrics: dict = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    mb_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        mb_metrics.update(
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                        )

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        ent_agg = agg_loss(entropy, response_mask, loss_agg_mode)
                        mb_metrics["actor/entropy"] = ent_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss = policy_loss - ent_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(kld, response_mask, loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale
                        mb_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    total_loss = policy_loss * loss_scale
                    if self.scaler is not None:
                        self.scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale
                    append_to_dict(metrics, mb_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()

        if post_sketch_data is not None:
            sketch, completed, device, group_size_post = post_sketch_data
            post_metrics = _finalise_sketch(sketch, completed, device, group_size_post)
            metrics.update({f"jacobian/post_{k}": v for k, v in post_metrics.items()})

        return metrics
