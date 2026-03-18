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
    rewards: torch.Tensor | None = None,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Estimate the rank of the gradient Jacobian from the log-probability matrix.

    In GRPO the per-rollout gradient is ``A_i * ∇_θ log π_θ(o_i | q)``.
    The log-probability matrix (N, seq_len) is used as a tractable proxy:
    rollouts whose log-prob rows are linearly independent will produce
    linearly independent gradients, so rank(L) ≤ rank(J).  Optionally
    weight each row by ``|reward_i|`` to approximate the magnitude-weighted
    Jacobian ``diag(|A|) * J``.

    Returns both the numerical rank (singular values above
    ``threshold * σ_max``) and the Roy–Vetterli effective rank
    ``exp(H(σ̂))`` — the exponential of the normalised singular-value
    entropy, which gives a continuous, noise-robust measure of
    dimensionality in ``[1, N]``.

    Parameters
    ----------
    log_probs:
        Shape ``(N, seq_len)`` — token-level log-probabilities.
    response_mask:
        Shape ``(N, seq_len)`` — 1 for response tokens, 0 elsewhere.
    rewards:
        Shape ``(N,)`` — optional per-rollout scalar rewards used to
        weight rows before the SVD.
    threshold:
        Fraction of the largest singular value below which a singular
        value is treated as zero for the numerical rank computation.
    """
    masked = (log_probs * response_mask).float()              # (N, seq_len)
    col_active = response_mask.any(dim=0)
    masked = masked[:, col_active]                            # (N, active_len)

    if rewards is not None:
        masked = masked * rewards.float().abs().unsqueeze(1)  # (N, active_len)

    sv = torch.linalg.svdvals(masked)                         # (min(N, L),)

    rank = int((sv > threshold * sv[0]).sum().item())

    sv_norm = sv / sv.sum().clamp(min=1e-12)
    entropy = -(sv_norm * (sv_norm + 1e-10).log()).sum()
    effective_rank = float(entropy.exp().item())

    return {"rank": float(rank), "effective_rank": effective_rank}


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
