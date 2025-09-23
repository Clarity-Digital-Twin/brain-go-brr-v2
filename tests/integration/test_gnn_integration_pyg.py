"""Integration tests for PyG GNN with Laplacian PE.

Tests the full pipeline with PyG Dynamic GNN + LPE enabled.
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


@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
@pytest.mark.serial  # PyG tests crash when run with many parallel workers
class TestGNNIntegrationPyG:
    """Test PyG GNN integration with detector."""

    @pytest.fixture
    def config_with_pyg_gnn(self) -> ModelConfig:
        """ModelConfig with PyG GNN and LPE enabled."""
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
            graph=GraphConfig(
                enabled=True,
                use_pyg=True,  # Enable PyG with LPE
                similarity="cosine",
                top_k=3,
                threshold=1e-4,
                temperature=0.1,
                n_layers=2,
                dropout=0.1,
                use_residual=True,
                alpha=0.05,
                k_eigenvectors=16,  # Laplacian PE dimension
            ),
        )

    def test_detector_with_pyg_gnn_forward(self, config_with_pyg_gnn):
        """Full forward pass with PyG GNN and LPE enabled."""
        detector = SeizureDetector.from_config(config_with_pyg_gnn)
        x = torch.randn(2, 19, 15360)

        output = detector(x)
        assert output.shape == (2, 15360)
        assert not torch.isnan(output).any()

        # Check GNN components were initialized
        assert detector.use_gnn is True
        assert detector.graph_builder is not None
        assert detector.gnn is not None

    def test_pyg_gnn_vs_pure_torch_shape(self, config_with_pyg_gnn):
        """PyG and pure-torch GNN should have same output shape."""
        config_pure = ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                use_pyg=False,  # Pure torch
            ),
        )

        detector_pyg = SeizureDetector.from_config(config_with_pyg_gnn)
        detector_pure = SeizureDetector.from_config(config_pure)

        x = torch.randn(1, 19, 15360)

        output_pyg = detector_pyg(x)
        output_pure = detector_pure(x)

        # Same shape regardless of implementation
        assert output_pyg.shape == output_pure.shape
        assert output_pyg.shape == (1, 15360)

    def test_pyg_gnn_gradient_flow(self, config_with_pyg_gnn):
        """Gradients should flow through entire model with PyG GNN."""
        detector = SeizureDetector.from_config(config_with_pyg_gnn)
        x = torch.randn(1, 19, 15360, requires_grad=True)

        output = detector(x)
        loss = output.mean()
        loss.backward()

        # Check gradient flows to input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().mean() > 0

        # Check all GNN components have gradients
        for param in detector.gnn.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_pyg_lpe_parameter_count(self, config_with_pyg_gnn):
        """PyG with LPE should have similar parameter count to pure-torch."""
        config_pure = ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                use_pyg=False,
            ),
        )

        detector_pyg = SeizureDetector.from_config(config_with_pyg_gnn)
        detector_pure = SeizureDetector.from_config(config_pure)

        params_pyg = detector_pyg.count_parameters()
        params_pure = detector_pure.count_parameters()

        # Both should have similar parameter counts
        # LPE doesn't add learnable params, just computed features
        # SSGConv has same params as our Linear layers
        assert abs(params_pyg - params_pure) < 10000  # Within 10k params

    @pytest.mark.parametrize("k_eigenvectors", [8, 16, 18])
    def test_pyg_with_different_lpe_dims(self, k_eigenvectors):
        """PyG GNN should work with different LPE dimensions."""
        config = ModelConfig(
            architecture="tcn",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                use_pyg=True,
                k_eigenvectors=k_eigenvectors,
            ),
        )

        detector = SeizureDetector.from_config(config)
        x = torch.randn(1, 19, 15360)

        output = detector(x)
        assert output.shape == (1, 15360)
        assert not torch.isnan(output).any()
