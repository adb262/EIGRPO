"""Tests for trajectory selector implementations."""

from __future__ import annotations

import torch
import pytest

from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch
from eigrpo.sampling.uniform import UniformSelector
from eigrpo.sampling.eig_selector import EIGSelector


def _make_batch(n: int) -> TrajectoryBatch:
    return TrajectoryBatch(
        rollout_ids=[f"r{i}" for i in range(n)],
        trajectories=[[{"step": j} for j in range(3)] for _ in range(n)],
        rewards=torch.randn(n),
    )


class TestUniformSelector:
    def test_select_all_when_n_select_large(self):
        selector = UniformSelector()
        batch = _make_batch(4)
        selected = selector.select(batch, n_select=10)
        assert selected == [0, 1, 2, 3]

    def test_select_subset(self):
        selector = UniformSelector()
        batch = _make_batch(10)
        selected = selector.select(batch, n_select=3)
        assert len(selected) == 3
        assert all(0 <= idx < 10 for idx in selected)
        assert len(set(selected)) == 3  # no duplicates

    def test_select_returns_sorted_indices(self):
        selector = UniformSelector()
        batch = _make_batch(20)
        selected = selector.select(batch, n_select=5)
        assert selected == sorted(selected)


class TestEIGSelector:
    def test_returns_first_n(self):
        selector = EIGSelector()
        batch = _make_batch(8)
        selected = selector.select(batch, n_select=4)
        assert selected == [0, 1, 2, 3]

    def test_clamps_to_available(self):
        selector = EIGSelector()
        batch = _make_batch(3)
        selected = selector.select(batch, n_select=10)
        assert selected == [0, 1, 2]


class TestBaseTrajectorySelector:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseTrajectorySelector()  # type: ignore[abstract]
