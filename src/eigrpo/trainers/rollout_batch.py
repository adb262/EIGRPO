"""Typed view over verl's ``DataProto`` for the EIGRPO selection hook."""

from __future__ import annotations

import torch
from verl import DataProto


class RolloutBatch:
    """Typed view over a verl ``DataProto``.

    Provides attribute access instead of string-keyed dictionary lookups
    so that field names are checked at write time rather than at runtime.
    """

    def __init__(self, data: DataProto) -> None:
        self._data = data

    @property
    def size(self) -> int:
        return self._data.batch.batch_size[0]

    @property
    def token_level_rewards(self) -> torch.Tensor:
        return self._data.batch["token_level_rewards"]

    @property
    def old_log_probs(self) -> torch.Tensor | None:
        return self._data.batch.get("old_log_probs", None)

    @property
    def response_mask(self) -> torch.Tensor | None:
        return self._data.batch.get("response_mask", None)

    @property
    def responses(self) -> torch.Tensor | None:
        return self._data.batch.get("responses", None)

    @property
    def uids(self) -> list[str]:
        raw = self._data.non_tensor_batch.get("uid", None)
        if raw is not None:
            return list(raw)
        return [str(i) for i in range(self.size)]

    def select_indices(self, indices: torch.Tensor) -> DataProto:
        """Return a new ``DataProto`` containing only the chosen rollout indices."""
        return self._data[indices]
