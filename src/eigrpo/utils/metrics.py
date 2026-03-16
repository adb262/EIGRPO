"""Pure metric computation functions for EIGRPO training diagnostics.

Each function accepts tensors and returns a flat ``dict[str, float]``
ready for logging.  No function has side-effects or knows about wandb.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn


def compute_entropy(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Entropy proxy from sampled-token log probabilities.

    Since only ``log p(a_t)`` for the sampled token is available (not the
    full vocabulary distribution), we report ``-mean(log_prob)`` over
    response tokens as a cross-entropy proxy that tracks policy sharpness.

    Parameters
    ----------
    old_log_probs:
        Shape ``(batch, seq_len)`` — per-token log-probs of the sampled
        tokens under the rollout policy.
    response_mask:
        Shape ``(batch, seq_len)`` — 1 for response tokens, 0 elsewhere.
        When *None*, all positions with ``old_log_probs != 0`` are used.
    """
    mask = _resolve_mask(old_log_probs, response_mask)
    neg_lp = -old_log_probs

    per_rollout_sum = (neg_lp * mask).sum(dim=-1)
    per_rollout_count = mask.sum(dim=-1).clamp(min=1)
    per_rollout_mean = per_rollout_sum / per_rollout_count

    return {
        "entropy/mean": per_rollout_mean.mean().item(),
        "entropy/per_rollout_std": per_rollout_mean.std().item() if per_rollout_mean.numel() > 1 else 0.0,
    }


def compute_min_log_prob(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Per-rollout minimum log probability across response tokens.

    Captures the model's worst-case confidence on a single token within
    each rollout — useful for spotting degenerate sampling.

    Parameters
    ----------
    old_log_probs:
        Shape ``(batch, seq_len)``.
    response_mask:
        Shape ``(batch, seq_len)``.  See :func:`compute_entropy`.
    """
    mask = _resolve_mask(old_log_probs, response_mask)
    filled = old_log_probs.masked_fill(mask == 0, float("inf"))
    per_rollout_min = filled.min(dim=-1).values

    valid = per_rollout_min[per_rollout_min != float("inf")]
    if valid.numel() == 0:
        return {"min_log_prob/mean": 0.0, "min_log_prob/min": 0.0, "min_log_prob/std": 0.0}

    return {
        "min_log_prob/mean": valid.mean().item(),
        "min_log_prob/min": valid.min().item(),
        "min_log_prob/std": valid.std().item() if valid.numel() > 1 else 0.0,
    }


def compute_mastery_pct(
    rewards: torch.Tensor,
    uids: list[str],
) -> dict[str, float]:
    """Fraction of prompt groups where every rollout is correct.

    A group is "mastered" when *all* of its rollouts receive a positive
    reward, indicating the model can reliably solve that prompt.

    Parameters
    ----------
    rewards:
        Shape ``(batch,)`` — per-rollout scalar reward (e.g. sum of
        token-level rewards).
    uids:
        Prompt identifier for each rollout — rollouts sharing a UID
        belong to the same group.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for uid, r in zip(uids, rewards.tolist()):
        groups[uid].append(r)

    num_groups = len(groups)
    if num_groups == 0:
        return {"mastery/pct": 0.0, "mastery/num_mastered": 0.0, "mastery/num_groups": 0.0}

    num_mastered = sum(1 for rs in groups.values() if all(r > 0 for r in rs))

    return {
        "mastery/pct": num_mastered / num_groups,
        "mastery/num_mastered": float(num_mastered),
        "mastery/num_groups": float(num_groups),
    }


def compute_gradient_norm(model: nn.Module) -> dict[str, float]:
    """Total L2 gradient norm across all model parameters.

    Must be called *after* ``loss.backward()`` and *before*
    ``optimizer.zero_grad()``.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.detach().norm(2).item() ** 2

    return {"gradient_norm/total": total_norm_sq**0.5}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _resolve_mask(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return a float mask, falling back to non-zero log-prob positions."""
    if response_mask is not None:
        return response_mask.float()
    return (log_probs != 0).float()
