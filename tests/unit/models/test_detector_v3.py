"""Unit tests for V3 detector with dual-stream architecture."""

import pytest
import torch

from src.brain_brr.config.schemas import (
    GraphConfig,
    MambaConfig,
    ModelConfig,
    TCNConfig,
)
from src.brain_brr.models.detector import SeizureDetector


class TestDetectorV3:
    """Test V3 dual-stream detector."""

    def test_v3_detector_from_config(self):
        """Test v3 detector initialization from config."""
        cfg = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                edge_features="cosine",
                edge_top_k=3,
                edge_threshold=1e-4,
                edge_mamba_layers=2,
                edge_mamba_d_state=8,
            ),
        )

        detector = SeizureDetector.from_config(cfg)

        # Check architecture
        assert detector.architecture == "v3"

        # Check v3 components exist
        assert detector.node_mamba is not None
        assert detector.edge_mamba is not None
        assert detector.edge_head is not None
        assert detector.proj_to_electrodes is not None
        assert detector.proj_from_electrodes is not None

        # Check GNN is created
        assert detector.gnn is not None

    def test_v3_forward_shape(self):
        """Test v3 forward pass produces correct shape."""
        cfg = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                edge_features="cosine",
                edge_top_k=3,
                edge_threshold=1e-4,
                edge_mamba_layers=2,
                edge_mamba_d_state=8,
            ),
        )

        detector = SeizureDetector.from_config(cfg)
        detector.eval()

        # Small input for testing
        x = torch.randn(2, 19, 15360)

        with torch.no_grad():
            output = detector(x)

        # Check output shape
        assert output.shape == (2, 15360)

    def test_v3_forward_no_nan(self):
        """Test v3 forward pass produces no NaNs."""
        cfg = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(
                enabled=True,
                edge_features="cosine",
                edge_top_k=3,
                edge_threshold=1e-4,
            ),
        )

        detector = SeizureDetector.from_config(cfg)
        detector.eval()

        x = torch.randn(1, 19, 15360)

        with torch.no_grad():
            output = detector(x)

        assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    def test_v3_without_gnn(self):
        """Test v3 works without GNN enabled."""
        cfg = ModelConfig(
            architecture="v3",
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(enabled=False),  # GNN disabled
        )

        detector = SeizureDetector.from_config(cfg)
        detector.eval()

        x = torch.randn(1, 19, 15360)

        with torch.no_grad():
            output = detector(x)

        assert output.shape == (1, 15360)
        assert torch.isfinite(output).all()

    def test_v3_edge_config_stored(self):
        """Test edge config is stored in detector instance."""
        cfg = ModelConfig(
            architecture="v3",
            graph=GraphConfig(
                enabled=True,
                edge_features="correlation",
                edge_top_k=5,
                edge_threshold=1e-3,
            ),
        )

        detector = SeizureDetector.from_config(cfg)

        # Check edge config stored
        assert detector.config["edge_metric"] == "correlation"
        assert detector.config["edge_top_k"] == 5
        assert detector.config["edge_threshold"] == 1e-3

    def test_v2_still_works(self):
        """Test v2 architecture still works after v3 changes."""
        cfg = ModelConfig(
            architecture="tcn",  # v2
            tcn=TCNConfig(),
            mamba=MambaConfig(),
            graph=GraphConfig(enabled=True),
        )

        detector = SeizureDetector.from_config(cfg)

        # Check v2 uses heuristic graph builder
        assert detector.architecture == "tcn"
        assert detector.graph_builder is not None

        # Check v3 components are not created
        assert detector.node_mamba is None
        assert detector.edge_mamba is None
        assert detector.edge_head is None

        # Test forward pass
        x = torch.randn(1, 19, 15360)
        with torch.no_grad():
            output = detector(x)

        assert output.shape == (1, 15360)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
