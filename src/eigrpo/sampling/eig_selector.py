from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch


class EIGSelector(BaseTrajectorySelector):
    """Effective Information Gain selector (stub).

    The goal is to choose the subset of rollouts that maximises the
    information gain of the training batch.  This requires:

    1. A trajectory-level similarity / diversity measure so we can
       identify duplicative rollouts.
    2. A notion of *effective* information per trajectory — potentially
       derived from the gradient norm or from token-level information
       gain w.r.t. the ground-truth answer.

    For now this falls through to selecting the first ``n_select``
    rollouts so the rest of the pipeline can be validated end-to-end.
    """

    def select(
        self,
        batch: TrajectoryBatch,
        n_select: int,
    ) -> list[int]:
        # TODO: Compute pairwise trajectory diversity (see diversity.metrics)
        # TODO: Estimate per-trajectory information gain
        # TODO: Greedy / submodular selection maximising total EIG
        n_available = len(batch.rollout_ids)
        return list(range(min(n_select, n_available)))
