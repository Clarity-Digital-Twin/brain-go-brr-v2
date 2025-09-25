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
        """ModelConfig with GNN and LPE enabled (PyG is canonical)."""
        return ModelConfig(
            architecture="v3",
            tcn=TCNConfig(num_layers=8, kernel_size=7, dropout=0.15, stride_down=16),
            mamba=MambaConfig(n_layers=6, d_state=16, conv_kernel=4, dropout=0.1),
            graph=GraphConfig(
                enabled=True,
                edge_features="cosine",
                edge_top_k=3,
                edge_threshold=1e-4,
                n_layers=2,
                dropout=0.1,
                use_residual=True,
                alpha=0.05,
                k_eigenvectors=16,
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
        assert detector.graph_builder is None
        assert detector.gnn is not None

    def test_dynamic_vs_static_pe_shape(self, config_with_pyg_gnn):
        """Dynamic and static PE should produce same output shape."""
        config_static = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(enabled=True, use_dynamic_pe=False),
        )

        detector_dyn = SeizureDetector.from_config(config_with_pyg_gnn)
        detector_static = SeizureDetector.from_config(config_static)

        x = torch.randn(1, 19, 15360)
        out_dyn = detector_dyn(x)
        out_static = detector_static(x)

        assert out_dyn.shape == out_static.shape == (1, 15360)

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

        # Check all model components have gradients
        for param in detector.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_dynamic_vs_static_param_count(self, config_with_pyg_gnn):
        """Dynamic vs static PE should have similar parameter counts (PE is non-learned)."""
        config_static = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(enabled=True, use_dynamic_pe=False),
        )

        detector_dyn = SeizureDetector.from_config(config_with_pyg_gnn)
        detector_static = SeizureDetector.from_config(config_static)

        params_dyn = detector_dyn.count_parameters()
        params_static = detector_static.count_parameters()

        assert abs(params_dyn - params_static) < 10000

    @pytest.mark.parametrize("k_eigenvectors", [8, 16, 18])
    def test_pyg_with_different_lpe_dims(self, k_eigenvectors):
        """PyG GNN should work with different LPE dimensions."""
        config = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(enabled=True, k_eigenvectors=k_eigenvectors),
        )

        detector = SeizureDetector.from_config(config)
        x = torch.randn(1, 19, 15360)

        output = detector(x)
        assert output.shape == (1, 15360)
        assert not torch.isnan(output).any()
