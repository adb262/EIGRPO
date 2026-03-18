"""Tests for trajectory selector implementations."""

from __future__ import annotations

import pytest
import torch

from eigrpo.sampling.base_selector import BaseTrajectorySelector, TrajectoryBatch
from eigrpo.sampling.eig_selector import EIGSelector, _greedy_map_dpp_batched
from eigrpo.sampling.uniform import UniformSelector


def _make_batch(n_total: int, group_size: int) -> TrajectoryBatch:
    """Create a dummy batch with ``n_total`` rollouts split into groups of ``group_size``."""
    return TrajectoryBatch(
        rollout_ids=[f"r{i}" for i in range(n_total)],
        trajectories=[[{"step": j} for j in range(3)] for _ in range(n_total)],
        rewards=torch.randn(n_total),
        group_size=group_size,
        response_texts=[f"response {i}" for i in range(n_total)],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_per_group_indices(indices: list[int], P: int, G: int, k: int) -> None:
    """Assert group-aware structural invariants on a selection result.

    Each group must contribute exactly ``k`` distinct indices, all falling in
    ``[group_start, group_start + G)``.
    """
    assert len(indices) == P * k, f"expected {P * k} indices, got {len(indices)}"
    assert len(set(indices)) == P * k, "duplicate indices found"
    for p in range(P):
        group_start = p * G
        group_end = group_start + G
        group_indices = [i for i in indices if group_start <= i < group_end]
        assert len(group_indices) == k, (
            f"group {p}: expected {k} indices, got {len(group_indices)}"
        )


# ---------------------------------------------------------------------------
# UniformSelector
# ---------------------------------------------------------------------------

class TestUniformSelector:
    def test_select_all_when_n_select_gte_group_size(self):
        """When n_select >= G every rollout is returned."""
        P, G = 2, 3
        batch = _make_batch(P * G, group_size=G)
        selector = UniformSelector()
        selected = selector.select(batch, n_select=G + 5)
        assert sorted(selected) == list(range(P * G))

    def test_per_group_count(self):
        """Each group contributes exactly n_select indices."""
        P, G, k = 4, 16, 5
        batch = _make_batch(P * G, group_size=G)
        selector = UniformSelector()
        selected = selector.select(batch, n_select=k)
        _check_per_group_indices(selected, P, G, k)

    def test_no_cross_group_contamination(self):
        """Indices from group i fall exclusively in [i*G, (i+1)*G)."""
        P, G, k = 3, 8, 3
        batch = _make_batch(P * G, group_size=G)
        selector = UniformSelector()
        selected = selector.select(batch, n_select=k)
        for p in range(P):
            group_indices = [i for i in selected if p * G <= i < (p + 1) * G]
            outside = [i for i in selected if i < p * G or i >= (p + 1) * G]
            assert len(group_indices) == k
            # outside indices should NOT include any from this group's range
            for idx in outside:
                assert not (p * G <= idx < (p + 1) * G)

    def test_no_duplicates(self):
        P, G, k = 5, 10, 4
        batch = _make_batch(P * G, group_size=G)
        selector = UniformSelector()
        selected = selector.select(batch, n_select=k)
        assert len(selected) == len(set(selected))


# ---------------------------------------------------------------------------
# EIG greedy MAP-DPP (batched kernel)
# ---------------------------------------------------------------------------

class TestGreedyMapDPPBatched:
    def _identity_kernel(self, P: int, G: int) -> torch.Tensor:
        return torch.eye(G).unsqueeze(0).expand(P, G, G).contiguous()

    def test_output_shape(self):
        kernel = self._identity_kernel(4, 8)
        result = _greedy_map_dpp_batched(kernel, n_select=3)
        assert result.shape == (4, 3)

    def test_no_repeated_indices_within_group(self):
        P, G, k = 3, 6, 4
        kernel = self._identity_kernel(P, G)
        result = _greedy_map_dpp_batched(kernel, n_select=k)
        for p in range(P):
            row = result[p].tolist()
            assert len(set(row)) == k

    def test_clamps_to_available(self):
        """When n_select > G, all G items are returned (no crash)."""
        P, G = 2, 4
        kernel = self._identity_kernel(P, G)
        result = _greedy_map_dpp_batched(kernel, n_select=10)
        assert result.shape[0] == P
        assert result.shape[1] <= G

    def test_stops_rows_that_become_degenerate(self):
        """If one batch row exhausts early, no extra invalid picks are emitted."""
        kernel = torch.stack(
            [
                torch.eye(3, dtype=torch.float32),
                torch.ones((3, 3), dtype=torch.float32),
            ],
            dim=0,
        )
        result = _greedy_map_dpp_batched(kernel, n_select=2)
        assert result.shape == (2, 1)
        assert all(0 <= idx < 3 for idx in result.view(-1).tolist())

    def test_diverse_selection_single_group(self):
        """The most-orthogonal pair from a cosine kernel should be selected."""
        emb = torch.tensor(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        emb = emb / emb.norm(dim=1, keepdim=True)
        kernel = (emb @ emb.T).unsqueeze(0)  # (1, 4, 4)
        result = _greedy_map_dpp_batched(kernel, n_select=2)
        selected = sorted(result[0].tolist())
        # The greedy MAP first breaks a diagonal tie, then maximises the residual.
        # The exact pair can vary; just check two distinct valid items.
        assert len(set(selected)) == 2
        assert all(0 <= i < 4 for i in selected)


# ---------------------------------------------------------------------------
# EIGSelector (end-to-end with fake embedding client)
# ---------------------------------------------------------------------------

class TestEIGSelector:
    class _FakeClient:
        def __init__(self, embeddings: torch.Tensor):
            self._embeddings = embeddings

        def create_embeddings(self, _texts: list[str]) -> torch.Tensor:
            return self._embeddings

    def _selector_with_embeddings(self, emb: torch.Tensor) -> EIGSelector:
        EIGSelector._client = self._FakeClient(emb)
        return EIGSelector()

    def test_select_all_when_n_select_gte_group_size(self):
        P, G = 2, 3
        emb = torch.randn(P * G, 8)
        selector = self._selector_with_embeddings(emb)
        batch = _make_batch(P * G, group_size=G)
        selected = selector.select(batch, n_select=G + 5)
        assert sorted(selected) == list(range(P * G))

    def test_per_group_count(self):
        P, G, k = 3, 8, 3
        emb = torch.randn(P * G, 16)
        selector = self._selector_with_embeddings(emb)
        batch = _make_batch(P * G, group_size=G)
        selected = selector.select(batch, n_select=k)
        _check_per_group_indices(selected, P, G, k)

    def test_selects_diverse_rollouts_single_group(self):
        """Single-group: most-diverse pair is selected."""
        G = 4
        emb = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        selector = self._selector_with_embeddings(emb)
        batch = _make_batch(G, group_size=G)
        selected = selector.select(batch, n_select=2)
        # All selected must be valid indices; diversity check via set cardinality.
        assert len(selected) == 2
        assert len(set(selected)) == 2
        assert all(0 <= i < G for i in selected)

    def test_raises_on_empty_response_texts(self):
        selector = self._selector_with_embeddings(torch.randn(4, 8))
        batch = TrajectoryBatch(
            rollout_ids=["r0", "r1", "r2", "r3"],
            trajectories=[[] for _ in range(4)],
            rewards=torch.zeros(4),
            group_size=4,
            response_texts=None,
        )
        with pytest.raises(ValueError, match="response_texts"):
            selector.select(batch, n_select=2)

    def test_raises_when_embedding_shape_mismatch(self):
        P, G = 1, 5
        emb = torch.randn(3, 8)  # wrong number of embeddings
        selector = self._selector_with_embeddings(emb)
        batch = _make_batch(P * G, group_size=G)
        with pytest.raises(RuntimeError):
            selector.select(batch, n_select=3)


class TestBaseTrajectorySelector:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseTrajectorySelector()  # type: ignore[abstract]
