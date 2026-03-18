from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TrajectoryBatch:
    """Lightweight container for a group of rollout trajectories and their metadata.

    The batch holds P * G rollouts arranged in contiguous groups:
    ``[p0_r0 … p0_r(G-1) | p1_r0 … p1_r(G-1) | … | p(P-1)_r(G-1)]``

    ``group_size`` (G) is the number of rollouts per prompt.  Selection is
    applied independently within each group so that each prompt contributes
    exactly ``n_select`` rollouts to the training batch.
    """

    rollout_ids: list[str]
    trajectories: list[list[dict]]
    rewards: torch.Tensor
    group_size: int  # rollouts per prompt (e.g. rollout.n = 64)
    token_sequences: list[list[int]] | None = None
    log_probs: torch.Tensor | None = None
    response_texts: list[str] | None = None


class BaseTrajectorySelector(ABC):
    """Abstract base for trajectory selection strategies.

    Subclasses decide *which* rollouts from a group are used for the
    backward pass.  The interface is intentionally minimal so that new
    selection criteria (EIG, diversity-weighted, etc.) can be swapped in
    via a single config change.
    """

    @abstractmethod
    def select(
        self,
        batch: TrajectoryBatch,
        n_select: int,
    ) -> list[int]:
        """Return indices into ``batch`` identifying the effective sample set.

        Parameters
        ----------
        batch:
            The full group of rollouts for one prompt.
        n_select:
            Target effective sample size (ESS).  Implementations may return
            fewer indices if the batch is smaller than ``n_select``.
        """
        ...
