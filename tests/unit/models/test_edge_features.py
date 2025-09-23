"""Unit tests for edge feature extraction and adjacency assembly."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as func

from src.brain_brr.models.edge_features import (
    assemble_adjacency,
    edge_scalar_series,
    get_structural_adjacency,
    pair_indices_undirected,
)


class TestPairIndices:
    """Test pair index generation."""

    def test_pair_indices_undirected_count(self):
        """Test correct number of pairs: n*(n-1)/2."""
        # Small test
        pairs = pair_indices_undirected(3)
        assert len(pairs) == 3  # 3*2/2 = 3
        assert pairs == [(0, 1), (0, 2), (1, 2)]

        # 19 electrodes
        pairs = pair_indices_undirected(19)
        assert len(pairs) == 171  # 19*18/2 = 171

    def test_pair_indices_ordering(self):
        """Test pairs are in correct order (i < j)."""
        pairs = pair_indices_undirected(5)
        for i, j in pairs:
            assert i < j, f"Pair ({i}, {j}) not in correct order"

    def test_pair_indices_uniqueness(self):
        """Test all pairs are unique."""
        pairs = pair_indices_undirected(10)
        pairs_set = set(pairs)
        assert len(pairs) == len(pairs_set), "Duplicate pairs found"

    def test_pair_indices_coverage(self):
        """Test all possible pairs are covered."""
        n = 5
        pairs = pair_indices_undirected(n)
        pairs_set = set(pairs)

        # Check all valid pairs exist
        for i in range(n):
            for j in range(i + 1, n):
                assert (i, j) in pairs_set, f"Missing pair ({i}, {j})"


class TestEdgeScalarSeries:
    """Test edge feature computation."""

    def test_edge_scalar_series_shape(self):
        """Test output shape: (batch_size, n_edges, seq_len, 1)."""
        batch_size, n_nodes, seq_len, feat_dim = 2, 19, 10, 64
        elec = torch.randn(batch_size, n_nodes, seq_len, feat_dim)

        edges = edge_scalar_series(elec, metric="cosine")

        n_edges = n_nodes * (n_nodes - 1) // 2  # 171 for n_nodes=19
        assert edges.shape == (batch_size, n_edges, seq_len, 1)

    def test_edge_scalar_series_finite(self):
        """Test all edge features are finite."""
        elec = torch.randn(2, 19, 10, 64)
        edges = edge_scalar_series(elec, metric="cosine")

        assert torch.isfinite(edges).all(), "Non-finite values in edge features"

    def test_edge_scalar_series_cosine_range(self):
        """Test cosine similarity is in [-1, 1]."""
        elec = torch.randn(1, 3, 5, 8)
        edges = edge_scalar_series(elec, metric="cosine")

        assert edges.min() >= -1.0 - 1e-6, f"Min {edges.min()} < -1"
        assert edges.max() <= 1.0 + 1e-6, f"Max {edges.max()} > 1"

    def test_edge_scalar_series_gradient_flow(self):
        """Test gradients flow through edge computation."""
        elec = torch.randn(1, 3, 5, 8, requires_grad=True)
        edges = edge_scalar_series(elec, metric="cosine")

        # Compute loss and backward
        loss = edges.mean()
        loss.backward()

        assert elec.grad is not None, "No gradient computed"
        assert torch.isfinite(elec.grad).all(), "Non-finite gradients"
        assert elec.grad.abs().sum() > 0, "Zero gradients"

    def test_edge_scalar_series_correlation(self):
        """Test correlation metric."""
        elec = torch.randn(1, 3, 5, 8)
        edges = edge_scalar_series(elec, metric="correlation")

        assert edges.shape == (1, 3, 5, 1)  # 3 pairs for 3 nodes
        assert torch.isfinite(edges).all()
        assert edges.min() >= -1.0 - 1e-6
        assert edges.max() <= 1.0 + 1e-6

    def test_edge_scalar_series_invalid_metric(self):
        """Test error on invalid metric."""
        elec = torch.randn(1, 3, 5, 8)
        with pytest.raises(ValueError, match="Unknown metric"):
            edge_scalar_series(elec, metric="invalid")

    def test_edge_scalar_series_custom_pairs(self):
        """Test with custom pair list."""
        elec = torch.randn(1, 4, 5, 8)
        pairs = [(0, 2), (1, 3)]  # Only 2 pairs instead of 6

        edges = edge_scalar_series(elec, pairs=pairs)
        assert edges.shape == (1, 2, 5, 1)


class TestAssembleAdjacency:
    """Test adjacency matrix assembly."""

    def test_assemble_adjacency_shape(self):
        """Test output shape: (batch_size, seq_len, n_nodes, n_nodes)."""
        batch_size, n_edges, seq_len = 2, 171, 10
        n_nodes = 19
        weights = torch.rand(batch_size, n_edges, seq_len)

        adj = assemble_adjacency(weights, n_nodes=n_nodes)
        assert adj.shape == (batch_size, seq_len, n_nodes, n_nodes)

    def test_assemble_adjacency_symmetry(self):
        """Test adjacency is symmetric."""
        weights = torch.rand(1, 171, 5)
        adj = assemble_adjacency(weights, symmetric=True)

        # Check symmetry
        adj_t = adj.transpose(-1, -2)
        assert torch.allclose(adj, adj_t, atol=1e-6), "Adjacency not symmetric"

    def test_assemble_adjacency_top_k(self):
        """Test top-k sparsification."""
        weights = torch.rand(1, 3, 1)  # 3 pairs for 3 nodes
        adj = assemble_adjacency(
            weights,
            n_nodes=3,
            top_k=1,  # Only keep 1 edge per node
            threshold=0
        )

        # Each row should have at most k non-zero entries (excluding diagonal)
        for i in range(3):
            row = adj[0, 0, i]
            non_zero = (row > 0).sum()
            assert non_zero <= 2, f"Row {i} has {non_zero} > 2 non-zeros (k=1 + possible self-loop)"

    def test_assemble_adjacency_threshold(self):
        """Test threshold pruning."""
        weights = torch.tensor([[[0.1], [0.5], [0.9]]])  # 1 batch, 3 edges, 1 timestep
        adj = assemble_adjacency(
            weights,
            n_nodes=3,
            threshold=0.4,
            top_k=3  # No top-k constraint
        )

        # Only weights > 0.4 should remain
        assert (adj[adj > 0] >= 0.4).all(), "Values below threshold remain"

    def test_assemble_adjacency_identity_fallback(self):
        """Test identity fallback for disconnected nodes."""
        # Create weights that leave node 2 disconnected
        weights = torch.zeros(1, 3, 1)
        weights[0, 0, 0] = 0.5  # Edge (0,1)

        adj = assemble_adjacency(
            weights,
            n_nodes=3,
            threshold=0.1,
            identity_fallback=True
        )

        # Node 2 should have self-loop
        assert adj[0, 0, 2, 2] == 1.0, "Disconnected node missing self-loop"

    def test_assemble_adjacency_sparse_output(self):
        """Test sparsity matches top_k * n_nodes."""
        weights = torch.rand(1, 171, 1)
        k = 3
        adj = assemble_adjacency(weights, top_k=k, threshold=0)

        # Count non-zero entries
        non_zeros = (adj[0, 0] > 0).sum()

        # Should be approximately k * n_nodes (some overlap due to symmetry)
        # But definitely less than n_nodes^2
        assert non_zeros < 19 * 19, "Adjacency not sparse"
        assert non_zeros > 0, "Adjacency too sparse (all zeros)"

    def test_assemble_adjacency_all_zeros(self):
        """Test handling of all-zero weights."""
        weights = torch.zeros(1, 171, 5)
        adj = assemble_adjacency(weights, identity_fallback=True)

        # Should have identity matrix at each timestep
        for t in range(5):
            diagonal = torch.diagonal(adj[0, t])
            assert (diagonal == 1.0).all(), f"Missing identity at timestep {t}"


class TestStructuralAdjacency:
    """Test structural adjacency for 10-20 montage."""

    def test_structural_adjacency_shape(self):
        """Test shape is (19, 19)."""
        adj = get_structural_adjacency()
        assert adj.shape == (19, 19)

    def test_structural_adjacency_symmetric(self):
        """Test adjacency is symmetric."""
        adj = get_structural_adjacency()
        assert torch.allclose(adj, adj.t()), "Structural adjacency not symmetric"

    def test_structural_adjacency_binary(self):
        """Test adjacency is binary."""
        adj = get_structural_adjacency()
        unique_vals = torch.unique(adj)
        assert set(unique_vals.tolist()) <= {0.0, 1.0}, "Non-binary values in adjacency"

    def test_structural_adjacency_no_self_loops(self):
        """Test no self-loops in structural adjacency."""
        adj = get_structural_adjacency()
        diagonal = torch.diagonal(adj)
        assert (diagonal == 0).all(), "Self-loops in structural adjacency"

    def test_structural_adjacency_sparsity(self):
        """Test adjacency is sparse (not fully connected)."""
        adj = get_structural_adjacency()
        num_edges = adj.sum().item()
        max_edges = 19 * 18  # Fully connected (excluding self-loops)

        assert num_edges < max_edges / 2, "Adjacency too dense"
        assert num_edges > 0, "Adjacency has no edges"

    def test_structural_adjacency_invalid_nodes(self):
        """Test error for non-19 nodes."""
        with pytest.raises(ValueError, match="only defined for 19 nodes"):
            get_structural_adjacency(n_nodes=10)


class TestIntegration:
    """Integration tests for full edge pipeline."""

    def test_full_pipeline(self):
        """Test complete edge feature → adjacency pipeline."""
        # Input electrode features
        batch_size, n_nodes, seq_len, feat_dim = 2, 19, 10, 64
        elec = torch.randn(batch_size, n_nodes, seq_len, feat_dim)

        # Compute edge features
        edges = edge_scalar_series(elec, metric="cosine")
        assert edges.shape == (batch_size, 171, seq_len, 1)

        # Apply edge Mamba (simulated with linear for test)
        edge_processed = edges  # In real V3, this goes through BiMamba2

        # Apply edge weight head (Linear + Softplus)
        edge_weights = func.softplus(edge_processed.squeeze(-1))  # (batch_size, n_edges, seq_len)

        # Assemble adjacency
        adj = assemble_adjacency(
            edge_weights,
            n_nodes=n_nodes,
            top_k=3,
            threshold=1e-4,
            symmetric=True,
            identity_fallback=True
        )
        assert adj.shape == (batch_size, seq_len, n_nodes, n_nodes)

        # Verify properties
        assert torch.isfinite(adj).all(), "Non-finite values in adjacency"
        assert (adj >= 0).all(), "Negative values in adjacency"

        # Check symmetry
        assert torch.allclose(adj, adj.transpose(-1, -2), atol=1e-6)

    def test_gradient_flow_full_pipeline(self):
        """Test gradients flow through entire pipeline."""
        elec = torch.randn(1, 19, 5, 64, requires_grad=True)

        # Forward pass
        edges = edge_scalar_series(elec)
        weights = func.softplus(edges.squeeze(-1))
        adj = assemble_adjacency(weights)

        # Compute loss
        loss = adj.mean()
        loss.backward()

        # Check gradients
        assert elec.grad is not None
        assert torch.isfinite(elec.grad).all()
        assert elec.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])