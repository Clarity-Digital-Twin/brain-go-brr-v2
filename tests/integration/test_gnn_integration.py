"""Integration tests for GNN in detector.

Tests the full pipeline with dynamic GNN enabled.
"""

import pytest
import torch

from src.brain_brr.config.schemas import GraphConfig, MambaConfig, ModelConfig, TCNConfig
from src.brain_brr.models.detector import SeizureDetector

# Check if PyG is available
try:
    import torch_geometric  # noqa: F401
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TestGNNIntegration:
    """Test GNN integration with detector."""

    @pytest.fixture
    def config_with_gnn(self) -> ModelConfig:
        """ModelConfig with GNN enabled using EvoBrain parameters."""
        return ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(
                num_layers=8,
                kernel_size=7,
                dropout=0.15,
                stride_down=16,
            ),
            mamba=MambaConfig(
                n_layers=6,
                d_state=16,
                conv_kernel=4,  # CUDA constraint
                dropout=0.1,
            ),
            graph=GraphConfig(
                enabled=True,
                # Graph construction (EvoBrain proven)
                similarity="cosine",
                top_k=3,
                threshold=1e-4,
                temperature=0.1,
                # GNN architecture
                n_layers=2,
                dropout=0.1,
                use_residual=True,
                # EvoBrain SSGConv parameters
                alpha=0.05,
                # PyG specific (for phase 2)
                use_pyg=False,
                k_eigenvectors=16,
            ),
        )

    @pytest.fixture
    def config_without_gnn(self) -> ModelConfig:
        """ModelConfig with GNN disabled."""
        return ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(
                num_layers=8,
                kernel_size=7,
                dropout=0.15,
                stride_down=16,
            ),
            mamba=MambaConfig(
                n_layers=6,
                d_state=16,
                conv_kernel=4,
                dropout=0.1,
            ),
        )

    def test_detector_with_gnn_forward(self, config_with_gnn):
        """Full forward pass with GNN enabled."""
        detector = SeizureDetector.from_config(config_with_gnn)
        x = torch.randn(2, 19, 15360)

        # Should not crash
        output = detector(x)

        # Check output shape
        assert output.shape == (2, 15360)

        # Check no NaNs
        assert not torch.isnan(output).any()

        # Check GNN components were initialized
        assert detector.use_gnn is True
        assert detector.graph_builder is not None
        assert detector.gnn is not None
        assert detector.proj_to_electrodes is not None
        assert detector.proj_from_electrodes is not None

    def test_gnn_matches_non_gnn_shape(self, config_with_gnn, config_without_gnn):
        """GNN and non-GNN paths must have same output shape."""
        detector_no_gnn = SeizureDetector.from_config(config_without_gnn)
        detector_with_gnn = SeizureDetector.from_config(config_with_gnn)

        x = torch.randn(2, 19, 15360)

        output_no_gnn = detector_no_gnn(x)
        output_with_gnn = detector_with_gnn(x)

        # Same shape regardless of GNN
        assert output_no_gnn.shape == output_with_gnn.shape
        assert output_no_gnn.shape == (2, 15360)

    def test_gnn_gradient_flow_integration(self, config_with_gnn):
        """Gradients should flow through entire model with GNN."""
        detector = SeizureDetector.from_config(config_with_gnn)
        x = torch.randn(1, 19, 15360, requires_grad=True)

        output = detector(x)
        loss = output.mean()
        loss.backward()

        # Check gradient flows to input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().mean() > 0

        # Check all GNN components have gradients
        for param in detector.graph_builder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

        for param in detector.gnn.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gnn_disabled_by_default(self, config_without_gnn):
        """GNN should be disabled when not in config."""
        detector = SeizureDetector.from_config(config_without_gnn)

        assert detector.use_gnn is False
        assert detector.graph_builder is None
        assert detector.gnn is None
        assert detector.proj_to_electrodes is None
        assert detector.proj_from_electrodes is None

    def test_gnn_parameter_count(self, config_with_gnn, config_without_gnn):
        """GNN should add parameters to the model."""
        detector_no_gnn = SeizureDetector.from_config(config_without_gnn)
        detector_with_gnn = SeizureDetector.from_config(config_with_gnn)

        params_no_gnn = detector_no_gnn.count_parameters()
        params_with_gnn = detector_with_gnn.count_parameters()

        # GNN should add parameters
        assert params_with_gnn > params_no_gnn

        # Roughly estimate:
        # - proj_to_electrodes: 512 * 19 * 64 = 622,592
        # - proj_from_electrodes: 19 * 64 * 512 = 622,592
        # - GNN layers: ~65k per layer
        # Total: ~1.3M additional parameters
        param_diff = params_with_gnn - params_no_gnn
        assert 1_000_000 < param_diff < 2_000_000

    def test_gnn_with_different_batch_sizes(self, config_with_gnn):
        """GNN should handle different batch sizes."""
        detector = SeizureDetector.from_config(config_with_gnn)

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 19, 15360)
            output = detector(x)
            assert output.shape == (batch_size, 15360)
            assert not torch.isnan(output).any()

    @pytest.mark.parametrize("top_k", [2, 3, 5])
    def test_gnn_with_different_top_k(self, top_k):
        """GNN should work with different top_k values."""
        config = ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                top_k=top_k,
            ),
        )

        detector = SeizureDetector.from_config(config)
        x = torch.randn(1, 19, 15360)

        output = detector(x)
        assert output.shape == (1, 15360)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_gnn_with_different_depths(self, n_layers):
        """GNN should work with different number of layers."""
        config = ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                n_layers=n_layers,
            ),
        )

        detector = SeizureDetector.from_config(config)
        x = torch.randn(1, 19, 15360)

        output = detector(x)
        assert output.shape == (1, 15360)
        assert not torch.isnan(output).any()