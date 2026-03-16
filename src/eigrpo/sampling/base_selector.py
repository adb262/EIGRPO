from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TrajectoryBatch:
    """Lightweight container for a group of rollout trajectories and their metadata."""

    rollout_ids: list[str]
    trajectories: list[list[dict]]
    rewards: torch.Tensor
    token_sequences: list[list[int]] | None = None
    log_probs: torch.Tensor | None = None


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
