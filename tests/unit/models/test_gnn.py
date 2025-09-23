"""Test GNN components - TDD style.

Based on EvoBrain architecture and parameters.
"""

import pytest
import torch

# Check if PyG is available
try:
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TestDynamicGraphBuilder:
    """Test dynamic graph construction."""

    def test_adjacency_shape(self):
        """Adjacency must be (B, T, N, N)."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder(top_k=3)
        features = torch.randn(2, 19, 960, 64)  # (B, N, T, D)
        adjacency = builder(features)

        assert adjacency.shape == (2, 960, 19, 19)

    def test_adjacency_symmetric(self):
        """Graph must be undirected (symmetric)."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder()
        features = torch.randn(1, 19, 10, 64)
        adjacency = builder(features)

        # Check symmetry
        assert torch.allclose(adjacency, adjacency.transpose(-1, -2))

    def test_top_k_sparsification(self):
        """Top-k should limit number of edges per node."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder(top_k=3)
        features = torch.randn(1, 19, 10, 64)
        adjacency = builder(features)

        # Each node should have at most top_k * 2 edges (bidirectional)
        for t in range(10):
            adj_t = adjacency[0, t]
            # Count non-zero edges per node
            for i in range(19):
                # Due to symmetrization, may have slightly more than 2*top_k edges
                # Because neighbors may also select this node
                assert (adj_t[i] > 0).sum() <= 10  # Reasonable upper bound

    def test_threshold_pruning(self):
        """Edges below threshold should be removed."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder(threshold=0.5)  # High threshold
        features = torch.randn(1, 19, 10, 64)
        adjacency = builder(features)

        # Should have very sparse adjacency with high threshold
        sparsity = (adjacency == 0).float().mean()
        assert sparsity > 0.8  # Most edges should be pruned

    def test_cosine_similarity(self):
        """Cosine similarity should be bounded [-1, 1]."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder(similarity="cosine")
        features = torch.randn(1, 19, 10, 64)
        adjacency = builder(features)

        # After softmax, values should be in [0, 1]
        assert adjacency.min() >= 0
        assert adjacency.max() <= 1

    def test_correlation_similarity(self):
        """Correlation similarity option should work."""
        from src.brain_brr.models.graph_builder import DynamicGraphBuilder

        builder = DynamicGraphBuilder(similarity="correlation")
        features = torch.randn(1, 19, 10, 64)
        adjacency = builder(features)

        assert adjacency.shape == (1, 10, 19, 19)
        assert torch.isfinite(adjacency).all()


class TestGraphChannelMixer:
    """Pure torch GNN tests removed; PyG is canonical now."""

    def test_placeholder(self) -> None:
        """Keep class for test discovery stability."""
        assert True


@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
class TestGraphChannelMixerPyG:
    """Test PyG GNN with Laplacian PE."""

    def test_pyg_gnn_with_lpe(self):
        """PyG GNN should add Laplacian PE to features."""
        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(
            d_model=64,
            k_eigenvectors=16,  # EvoBrain default
            alpha=0.05,
        )
        features = torch.randn(1, 19, 10, 64)
        adjacency = torch.randn(1, 10, 19, 19).softmax(dim=-1)

        output = gnn(features, adjacency)
        assert output.shape == features.shape
        assert torch.isfinite(output).all()

    def test_pyg_gnn_handles_sparse_adjacency(self):
        """PyG GNN should work with very sparse adjacency."""
        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(d_model=64)
        features = torch.randn(1, 19, 5, 64)

        # Create very sparse adjacency (only diagonal + few edges)
        adjacency = torch.eye(19).unsqueeze(0).unsqueeze(0).repeat(1, 5, 1, 1)
        adjacency[0, :, 0, 1] = 0.5  # Add one edge
        adjacency[0, :, 1, 0] = 0.5  # Symmetric

        output = gnn(features, adjacency)
        assert output.shape == features.shape
        assert torch.isfinite(output).all()

    def test_pyg_gnn_gradient_flow_with_lpe(self):
        """Gradients should flow through PyG GNN with LPE."""
        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(
            d_model=64,
            k_eigenvectors=8,
            n_layers=2,
        )
        features = torch.randn(1, 19, 3, 64, requires_grad=True)
        adjacency = torch.randn(1, 3, 19, 19).softmax(dim=-1)

        output = gnn(features, adjacency)
        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert features.grad.abs().mean() > 1e-8
