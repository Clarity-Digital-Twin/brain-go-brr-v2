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


class TestEdgeFeaturesAndAdjacency:
    """Test V3 edge feature pipeline and adjacency assembly."""

    def test_edge_scalar_series_shape(self):
        from src.brain_brr.models.edge_features import edge_scalar_series

        elec = torch.randn(2, 19, 10, 64)  # (B, N, T, D)
        edges = edge_scalar_series(elec, metric="cosine")  # (B, E, T, 1)
        assert edges.shape[0] == 2
        assert edges.shape[2] == 10
        assert edges.shape[3] == 1

    def test_assemble_adjacency_properties(self):
        from src.brain_brr.models.edge_features import assemble_adjacency

        weights = torch.rand(1, 171, 10)  # (B, E, T)
        adj = assemble_adjacency(weights, n_nodes=19, top_k=3, threshold=1e-4)
        assert adj.shape == (1, 10, 19, 19)
        # Symmetric
        assert torch.allclose(adj, adj.transpose(-1, -2))
        # Threshold applied
        assert (adj >= 0).all()

    def test_correlation_metric_support(self):
        from src.brain_brr.models.edge_features import edge_scalar_series

        elec = torch.randn(1, 19, 5, 64)
        edges_corr = edge_scalar_series(elec, metric="correlation")
        assert edges_corr.shape == (1, 171, 5, 1)


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
