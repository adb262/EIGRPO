import random

from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch


class UniformSelector(BaseTrajectorySelector):
    """Baseline selector that replicates standard GRPO behaviour.

    When ``n_select`` >= the batch size every rollout is used.  Otherwise a
    uniform random subset is drawn (without replacement).
    """

    def select(
        self,
        batch: TrajectoryBatch,
        n_select: int,
    ) -> list[int]:
        n_available = len(batch.rollout_ids)
        if n_select >= n_available:
            return list(range(n_available))
        return sorted(random.sample(range(n_available), n_select))
