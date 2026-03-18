import logging

import torch

from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch

logger = logging.getLogger("eigrpo.sampling.uniform")


class UniformSelector(BaseTrajectorySelector):
    """Baseline selector that replicates standard GRPO behaviour.

    Selects ``n_select`` rollouts independently within each prompt's group
    using vectorised tensor ops (one ``topk`` over a ``(P, G)`` random score
    matrix) so no Python loops over groups are needed.

    When ``n_select`` >= ``group_size`` every rollout is returned unchanged.
    """

    def select(
        self,
        batch: TrajectoryBatch,
        n_select: int,
    ) -> list[int]:
        G = batch.group_size
        P = len(batch.rollout_ids) // G
        logger.info(
            "UniformSelector: selecting %d from each of %d groups (G=%d)",
            n_select,
            P,
            G,
        )

        if n_select >= G:
            return list(range(len(batch.rollout_ids)))

        # Draw a (P, G) random score matrix and take the top-k indices per row.
        scores = torch.rand(P, G)
        _, local = scores.topk(n_select, dim=1)        # (P, k) local indices in [0, G)
        offsets = torch.arange(P).unsqueeze(1) * G     # (P, 1) global group offsets
        # Flatten in group-major order so the result is contiguous per prompt.
        return (local + offsets).view(-1).tolist()
