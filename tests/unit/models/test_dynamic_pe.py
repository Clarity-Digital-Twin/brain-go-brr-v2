"""Unit tests for dynamic Laplacian Positional Encoding."""

import time

import pytest
import torch

# Check if PyG is available
try:
    import torch_geometric  # noqa: F401

    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    GraphChannelMixerPyG = None  # type: ignore


@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
class TestDynamicPE:
    """Test suite for dynamic PE implementation."""

    def test_vectorized_shape(self):
        """Test dynamic PE produces correct shapes."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
        )
        adjacency = torch.rand(2, 960, 19, 19)

        pe = gnn._compute_dynamic_pe_vectorized(adjacency)
        assert pe.shape == (2, 960, 19, 16)
        assert not torch.isnan(pe).any()
        assert not torch.isinf(pe).any()

    def test_disconnected_graph(self):
        """Test stability with zero adjacency (fully disconnected)."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
        )
        adjacency = torch.zeros(1, 10, 19, 19)  # Fully disconnected

        pe = gnn._compute_dynamic_pe_vectorized(adjacency)
        assert not torch.isnan(pe).any()
        assert not torch.isinf(pe).any()

    def test_sign_consistency(self):
        """Test eigenvector signs don't randomly flip."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            pe_sign_consistency=True,
        )
        # Same adjacency repeated
        adj_single = torch.rand(1, 1, 19, 19)
        adjacency = adj_single.repeat(1, 100, 1, 1)

        pe = gnn._compute_dynamic_pe_vectorized(adjacency)

        # Check consecutive timesteps have consistent signs
        for t in range(1, 100):
            dot_product = (pe[0, t] * pe[0, t - 1]).sum(dim=0)  # Per eigenvector
            assert (dot_product >= -1e-6).all(), f"Eigenvector signs flipped at timestep {t}!"

    def test_semi_dynamic_interval(self):
        """Test semi-dynamic PE with update intervals."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
            semi_dynamic_interval=4,  # Update every 4 timesteps
        )
        features = torch.randn(1, 19, 12, 64)  # Small sequence for testing
        adjacency = torch.rand(1, 12, 19, 19)

        # Process through forward_vectorized
        output = gnn.forward_vectorized(features, adjacency)
        assert output.shape == (1, 19, 12, 64)
        assert not torch.isnan(output).any()

    def test_forward_with_dynamic_pe(self):
        """Test full forward pass with dynamic PE enabled."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
        )
        features = torch.randn(2, 19, 960, 64)
        adjacency = torch.rand(2, 960, 19, 19)

        # Apply threshold to make sparse
        adjacency = torch.where(adjacency > 0.7, adjacency, torch.zeros_like(adjacency))

        output = gnn.forward(features, adjacency)
        assert output.shape == features.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for performance test")
    def test_performance(self):
        """Benchmark vectorized implementation performance."""
        adjacency = torch.rand(8, 960, 19, 19).cuda()
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
        ).cuda()

        # Warmup
        _ = gnn._compute_dynamic_pe_vectorized(adjacency)
        torch.cuda.synchronize()

        # Time vectorized implementation
        start = time.time()
        gnn._compute_dynamic_pe_vectorized(adjacency)
        torch.cuda.synchronize()
        vectorized_time = time.time() - start

        print(f"Vectorized time: {vectorized_time * 1000:.2f}ms for B=8, T=960")
        # Should be under 100ms for reasonable performance
        assert vectorized_time < 0.1, f"Vectorized implementation too slow: {vectorized_time:.3f}s"

    def test_static_vs_dynamic_output_shape(self):
        """Ensure static and dynamic PE produce same output shapes."""
        # Static PE
        gnn_static = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=False,
            use_vectorized=True,
        )

        # Dynamic PE
        gnn_dynamic = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
        )

        features = torch.randn(2, 19, 960, 64)
        adjacency = torch.rand(2, 960, 19, 19)

        output_static = gnn_static.forward(features, adjacency)
        output_dynamic = gnn_dynamic.forward(features, adjacency)

        assert output_static.shape == output_dynamic.shape == features.shape

    def test_edge_case_single_timestep(self):
        """Test with single timestep (edge case for semi-dynamic)."""
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=16,
            use_dynamic_pe=True,
            use_vectorized=True,
            semi_dynamic_interval=4,
        )
        features = torch.randn(1, 19, 1, 64)  # Single timestep
        adjacency = torch.rand(1, 1, 19, 19)

        output = gnn.forward_vectorized(features, adjacency)
        assert output.shape == (1, 19, 1, 64)
        assert not torch.isnan(output).any()

    def test_k_eigenvectors_constraint(self):
        """Test that k eigenvectors is properly constrained."""
        # Should work with k <= N-1
        gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=18,  # Maximum allowed
            use_dynamic_pe=True,
        )
        adjacency = torch.rand(1, 10, 19, 19)
        pe = gnn._compute_dynamic_pe_vectorized(adjacency)
        assert pe.shape == (1, 10, 19, 18)
