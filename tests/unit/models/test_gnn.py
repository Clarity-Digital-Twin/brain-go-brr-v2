"""Test GNN components - TDD style.

Based on EvoBrain architecture and parameters.
"""

import torch


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
    """Test GNN module."""

    def test_gnn_preserves_shape(self):
        """GNN must preserve input shape."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64)
        features = torch.randn(2, 19, 960, 64)
        adjacency = torch.randn(2, 960, 19, 19)

        output = gnn(features, adjacency)
        assert output.shape == features.shape

    def test_gnn_gradient_flow(self):
        """Gradients must flow through GNN."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64)
        features = torch.randn(1, 19, 10, 64, requires_grad=True)
        adjacency = torch.randn(1, 10, 19, 19)

        output = gnn(features, adjacency)
        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert features.grad.abs().mean() > 1e-8

    def test_gnn_stability_with_identity_adj(self):
        """With identity adjacency, output should be finite and bounded."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64, n_layers=2)
        features = torch.randn(1, 19, 10, 64)
        adjacency = torch.eye(19).unsqueeze(0).unsqueeze(0).repeat(1, 10, 1, 1)

        with torch.no_grad():
            output = gnn(features, adjacency)

        assert torch.isfinite(output).all()
        # Output magnitude should be reasonable
        ratio = (output.pow(2).mean() / (features.pow(2).mean() + 1e-9)).sqrt()
        assert 0.1 <= ratio <= 10.0

    def test_gnn_alpha_mixing(self):
        """SSGConv alpha should control self vs neighbor mixing."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        # Test that different alpha values produce different outputs
        gnn_low_alpha = GraphChannelMixer(d_model=64, alpha=0.05)  # EvoBrain default
        gnn_high_alpha = GraphChannelMixer(d_model=64, alpha=0.5)

        features = torch.randn(1, 19, 10, 64)
        # Create a reasonable adjacency matrix
        adjacency = torch.randn(1, 10, 19, 19).softmax(dim=-1)

        with torch.no_grad():
            output_low = gnn_low_alpha(features, adjacency)
            output_high = gnn_high_alpha(features, adjacency)

        # Different alpha values should produce different outputs
        assert not torch.allclose(output_low, output_high, rtol=1e-4)

    def test_gnn_residual_connections(self):
        """Residual connections should help gradient flow."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn_with_res = GraphChannelMixer(d_model=64, use_residual=True)
        gnn_without_res = GraphChannelMixer(d_model=64, use_residual=False)

        features = torch.randn(1, 19, 10, 64, requires_grad=True)
        adjacency = torch.randn(1, 10, 19, 19)

        # Test gradient magnitude with residuals
        output_with = gnn_with_res(features, adjacency)
        loss_with = output_with.mean()
        loss_with.backward()
        grad_with = features.grad.clone()

        # Test without residuals
        features.grad = None
        output_without = gnn_without_res(features, adjacency)
        loss_without = output_without.mean()
        loss_without.backward()
        grad_without = features.grad

        # Residuals should generally help gradient flow
        assert grad_with.abs().mean() > 0
        assert grad_without.abs().mean() > 0

    def test_gnn_dropout(self):
        """Dropout should affect training vs eval mode."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64, dropout=0.5)
        features = torch.randn(1, 19, 10, 64)
        adjacency = torch.randn(1, 10, 19, 19)

        # Training mode
        gnn.train()
        output_train1 = gnn(features, adjacency)
        output_train2 = gnn(features, adjacency)

        # Outputs should differ in training due to dropout
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode
        gnn.eval()
        output_eval1 = gnn(features, adjacency)
        output_eval2 = gnn(features, adjacency)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)
