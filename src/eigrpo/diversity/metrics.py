"""Trajectory diversity and gradient norm measurement utilities.

These functions support the core hypothesis that most rollouts are
duplicative: by quantifying similarity across a rollout group we can
decide which trajectories carry unique information.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_jacobian_rank(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
    rewards: torch.Tensor | None = None,
    threshold: float = 0.1,
) -> dict[str, float]:
    """Estimate per-group gradient Jacobian rank from the log-probability matrix.

    In GRPO the gradient decomposes per prompt group: each group contributes
    ``A_i * ∇_θ log π_θ(o_i | q)`` where advantages are normalised within
    the group.  A single global SVD across all P*G rollouts is dominated by
    cross-group variation (different prompts live in different token-
    distribution subspaces) and is insensitive to whether selection preserved
    within-group diversity.

    This function reshapes the (P*G, seq_len) log-prob matrix into P groups
    of shape (G, seq_len), runs a separate SVD per group, and returns
    statistics averaged across groups.

    The log-prob rows are a tractable proxy for the true gradient vectors:
    the policy gradient is a linear map of the masked log-probs, so
    rank(log_prob_matrix) ≤ rank(true_Jacobian).  Optionally weight each
    row by ``|reward_i|`` to approximate the reward-weighted Jacobian
    ``diag(|A|) · J``.

    Parameters
    ----------
    log_probs:
        Shape ``(P*G, seq_len)`` — token-level log-probabilities.
    response_mask:
        Shape ``(P*G, seq_len)`` — 1 for response tokens, 0 elsewhere.
    group_size:
        G — number of rollouts per prompt group.
    rewards:
        Shape ``(P*G,)`` — optional per-rollout scalars used to weight each
        log-prob row by ``|reward_i|``.
    threshold:
        Fraction of ``σ_max`` below which a singular value is treated as
        zero for the hard numerical rank.
    """
    N, seq_len = log_probs.shape
    G = group_size
    P = N // G

    masked = (log_probs * response_mask).float()              # (P*G, seq_len)
    if rewards is not None:
        masked = masked * rewards.float().abs().unsqueeze(1)

    masked = masked.view(P, G, seq_len)                       # (P, G, seq_len)

    ranks: list[float] = []
    eff_ranks: list[float] = []

    for p in range(P):
        group_mat = masked[p]                                 # (G, seq_len)
        col_active = group_mat.abs().any(dim=0)
        group_mat = group_mat[:, col_active]                  # (G, active_len)
        if group_mat.numel() == 0 or group_mat.shape[1] == 0:
            continue
        sv = torch.linalg.svdvals(group_mat)                  # (min(G, active_len),)
        rank = int((sv > threshold * sv[0]).sum().item())
        sv_norm = sv / sv.sum().clamp(min=1e-12)
        entropy = -(sv_norm * (sv_norm + 1e-10).log()).sum()
        eff_ranks.append(float(entropy.exp().item()))
        ranks.append(float(rank))

    if not ranks:
        return {"rank": 0.0, "effective_rank": 0.0, "rank_frac": 0.0, "effective_rank_frac": 0.0}

    mean_rank = sum(ranks) / len(ranks)
    mean_eff_rank = sum(eff_ranks) / len(eff_ranks)
    return {
        "rank": mean_rank,
        "effective_rank": mean_eff_rank,
        # Normalise by G so pre/post are on the same [0, 1] scale regardless
        # of whether G=64 (pre) or G=n_select=8 (post).
        "rank_frac": mean_rank / G,
        "effective_rank_frac": mean_eff_rank / G,
    }


def compute_trajectory_similarity(
    token_sequences: list[list[int]],
) -> torch.Tensor:
    """Pairwise similarity matrix over tokenised rollout trajectories.

    Uses normalised longest-common-subsequence length as the similarity
    metric.  This is O(n^2 * L^2) where n = number of rollouts and
    L = average sequence length, so it is only suitable for moderate
    group sizes during analysis — not in the hot training loop.

    Returns a (n, n) similarity matrix with values in [0, 1].
    """
    n = len(token_sequences)
    sim = torch.zeros(n, n)

    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            lcs_len = _lcs_length(token_sequences[i], token_sequences[j])
            max_len = max(len(token_sequences[i]), len(token_sequences[j]), 1)
            score = lcs_len / max_len
            sim[i, j] = score
            sim[j, i] = score

    return sim


def compute_per_rollout_gradient_norms(
    model: nn.Module,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    group_size: int,
) -> list[float]:
    """Compute the L2 gradient norm attributable to each rollout in a group.

    This isolates the gradient contribution of each rollout so we can
    check whether most rollouts produce near-zero gradient (i.e. are
    duplicative from an optimisation standpoint).

    Parameters
    ----------
    model:
        The policy network whose parameters we differentiate w.r.t.
    log_probs:
        Shape ``(group_size, seq_len)`` — log-probabilities under the
        current policy for each rollout.
    response_mask:
        Shape ``(group_size, seq_len)`` — mask for valid response tokens.
    advantages:
        Shape ``(group_size,)`` or ``(group_size, seq_len)`` — GRPO
        advantages (already normalised over the group).
    group_size:
        Number of rollouts in the group.

    Returns
    -------
    List of per-rollout gradient L2 norms.
    """
    norms: list[float] = []

    for i in range(group_size):
        model.zero_grad()

        masked_lp = log_probs[i] * response_mask[i]
        adv = advantages[i] if advantages.dim() > 1 else advantages[i]
        surrogate = -(masked_lp * adv).sum()

        surrogate.backward(retain_graph=(i < group_size - 1))

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        norms.append(total_norm**0.5)

    model.zero_grad()
    return norms


def compute_unique_trajectory_ratio(token_sequences: list[list[int]], threshold: float = 0.95) -> float:
    """Fraction of rollouts with a distinct token sequence.

    Uses hash-based exact deduplication — O(n * L) — instead of the
    O(n^2 * L^2) pairwise LCS approach used by
    :func:`compute_trajectory_similarity`.  Exact match is the right
    tradeoff for the hot training loop: LCS near-duplicate detection at
    n=1024, L=2048 would stall training for several minutes per step.
    """
    seen: set[tuple[int, ...]] = set()
    unique = 0
    for seq in token_sequences:
        key = tuple(seq)
        if key not in seen:
            unique += 1
            seen.add(key)
    return unique / max(len(token_sequences), 1)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Length of the longest common subsequence between two int lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]
