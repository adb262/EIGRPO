from __future__ import annotations

import logging

import torch

from eigrpo.embeddings.client import VLLMEmbeddingClient
from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch

logger = logging.getLogger("eigrpo.sampling.eig_selector")


class EIGSelector(BaseTrajectorySelector):
    """DPP-based selector that maximises diversity within each prompt group.

    Embeds all P*G responses at once, reshapes to (P, G, d), computes a
    batched cosine-similarity kernel (P, G, G) and runs a fully-vectorised
    greedy MAP-DPP simultaneously across all P groups.

    Returns ``P * n_select`` global indices in group-contiguous order so that
    ``compute_advantage(num_repeat=n_select)`` sees the correct group structure.
    """

    _client: VLLMEmbeddingClient | None = None

    def __init__(self) -> None:
        if EIGSelector._client is None:
            EIGSelector._client = VLLMEmbeddingClient()

    def select(
        self,
        batch: TrajectoryBatch,
        n_select: int,
    ) -> list[int]:
        G = batch.group_size
        P = len(batch.rollout_ids) // G
        logger.info(
            "EIGSelector: selecting %d from each of %d groups (G=%d)",
            n_select,
            P,
            G,
        )

        if n_select >= G:
            return list(range(len(batch.rollout_ids)))

        if not batch.response_texts:
            raise ValueError("EIG selector requires non-empty response_texts for embedding.")

        # ── embed all P*G responses at once ──────────────────────────────────
        embeddings = self._client.create_embeddings(batch.response_texts)  # type: ignore[union-attr]
        n_available = len(batch.rollout_ids)
        if embeddings.dim() != 2 or embeddings.shape[0] != n_available:
            raise RuntimeError(
                f"vLLM embedding shape mismatch: expected ({n_available}, d), "
                f"got {tuple(embeddings.shape)}."
            )

        d = embeddings.shape[1]
        emb = embeddings.view(P, G, d)                                    # (P, G, d)

        # ── batched cosine kernel ─────────────────────────────────────────────
        norms = emb.norm(dim=2, keepdim=True).clamp(min=1e-12)
        emb_norm = emb / norms                                            # (P, G, d)
        kernel = torch.bmm(emb_norm, emb_norm.transpose(1, 2))           # (P, G, G)

        # ── greedy MAP-DPP across all groups simultaneously ───────────────────
        local = _greedy_map_dpp_batched(kernel, n_select)                 # (P, k)

        if local.shape[1] < n_select:
            raise RuntimeError(
                f"DPP returned only {local.shape[1]} items (requested {n_select}); "
                "kernel may be degenerate."
            )

        # Convert per-group local indices to flat global indices.
        offsets = torch.arange(P, device=local.device).unsqueeze(1) * G  # (P, 1)
        return (local + offsets).view(-1).tolist()


def _greedy_map_dpp_batched(kernel: torch.Tensor, n_select: int) -> torch.Tensor:
    """Greedy MAP inference for a batch of DPP L-ensemble kernels.

    Parameters
    ----------
    kernel:
        Shape ``(P, G, G)`` — one cosine-similarity kernel per prompt group.
    n_select:
        Number of items to select from each group.

    Returns
    -------
    Tensor of shape ``(P, k_valid)`` containing the selected local indices
    (0 … G-1) for each group. ``k_valid`` is the largest number of valid greedy
    selections shared by every batch item, up to ``min(n_select, G)``.
    """
    P, G, _ = kernel.shape
    k = min(n_select, G)

    available = torch.ones(P, G, dtype=torch.bool, device=kernel.device)
    diag = kernel.diagonal(dim1=1, dim2=2).clone()  # (P, G)
    cols: list[torch.Tensor] = []
    neg_inf = torch.tensor(float("-inf"), dtype=diag.dtype, device=kernel.device)
    p_idx = torch.arange(P, device=kernel.device)   # reused each iteration

    selected = torch.zeros(P, k, dtype=torch.long, device=kernel.device)
    active_rows = torch.ones(P, dtype=torch.bool, device=kernel.device)
    filled = torch.zeros(P, dtype=torch.long, device=kernel.device)

    for step in range(k):
        # Exhausted rows no longer participate in the batched argmax/update.
        scores = torch.where(available & active_rows.unsqueeze(1), diag, neg_inf)
        best = scores.argmax(dim=1)                     # (P,)
        active_this_step = active_rows & (diag[p_idx, best] > 0)

        if not active_this_step.any():
            break

        active_idx = active_this_step.nonzero(as_tuple=False).squeeze(1)
        selected[active_idx, step] = best[active_idx]
        available[active_idx, best[active_idx]] = False
        filled[active_idx] += 1

        # Gather the best column for each group: col[p] = kernel[p, :, best[p]]
        col = kernel[p_idx, :, best]                   # (P, G)

        # Subtract projections from previously selected directions.
        for prev in cols:
            coeff = prev[p_idx, best]                  # (P,)
            col = col - coeff.unsqueeze(1) * prev      # (P, G)

        # Normalise.
        denom = col[p_idx, best].clamp(min=1e-12).sqrt()  # (P,)
        col = col / denom.unsqueeze(1)                 # (P, G)
        col = torch.where(active_this_step.unsqueeze(1), col, torch.zeros_like(col))
        cols.append(col)

        # Only rows with positive residual gain should keep evolving.
        diag = torch.where(
            available,
            torch.where(active_this_step.unsqueeze(1), diag - col ** 2, diag),
            diag,
        )
        active_rows = active_this_step

    return selected[:, : filled.min().item()]
