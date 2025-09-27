"""Unit tests for PR-4: Clamp retirement and gated fusion."""

# ruff: noqa: N806  # Uppercase variable names are standard for PyTorch tensors

import torch

from src.brain_brr.config.schemas import (
    ClampRetirementConfig,
    FusionConfig,
    GraphConfig,
    ModelConfig,
    NormConfig,
)
from src.brain_brr.models.clamp_utils import monitored_clamp, monitored_nan_to_num
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.models.fusion import GatedFusion, MultiHeadGatedFusion


def test_gated_fusion():
    """Test gated fusion mechanism."""
    fusion = GatedFusion(64, dropout=0.0)

    B, N, T, D = 2, 19, 960, 64
    node = torch.randn(B, N, T, D)
    edge = torch.randn(B, N, T, D) * 0.1  # Small edge features

    fused = fusion(node, edge)

    # Check shape preserved
    assert fused.shape == (B, N, T, D)

    # Verify gating: output should be between node and node+edge
    diff = fused - node
    assert torch.isfinite(diff).all()

    # Gate should reduce edge contribution
    assert diff.abs().mean() < edge.abs().mean()


def test_multihead_fusion():
    """Test multi-head gated fusion."""
    fusion = MultiHeadGatedFusion(64, num_heads=4, dropout=0.0)

    B, N, T, D = 2, 19, 960, 64
    node = torch.randn(B, N, T, D)
    edge = torch.randn(B, N, T, D)

    fused = fusion(node, edge)

    # Check shape preserved
    assert fused.shape == (B, N, T, D)

    # Check finite values
    assert torch.isfinite(fused).all()


def test_monitored_clamp_logging():
    """Test clamp monitoring logs when values would be clamped."""
    x = torch.randn(100) * 100
    x[0] = 1000  # Would trigger clamp
    x[99] = -1000  # Would trigger clamp

    config = {"log_clamp_hits": True, "remove_intermediate_clamps": False}

    # Should apply clamp normally
    y = monitored_clamp(x, -50, 50, "test", config)

    # Check clamp was applied
    assert y.max() <= 50
    assert y.min() >= -50


def test_monitored_clamp_removal():
    """Test clamp removal when configured."""
    x = torch.randn(100) * 100
    x[0] = 1000  # Would normally be clamped

    config = {"remove_intermediate_clamps": True, "validate_finite": False}

    # Should NOT apply clamp
    y = monitored_clamp(x, -50, 50, "test", config)

    # Check clamp was NOT applied
    assert y[0] == 1000  # Original value preserved


def test_essential_clamps_preserved():
    """Test that essential clamps are never removed."""
    x = torch.randn(100) * 100
    x[0] = 1000

    config = {"remove_intermediate_clamps": True, "validate_finite": False}

    # Essential clamp should still be applied
    y = monitored_clamp(x, -50, 50, "tcn_input", config)

    # Essential clamp was applied despite remove flag
    assert y.max() <= 50


def test_nan_to_num_monitoring():
    """Test nan_to_num monitoring."""
    x = torch.randn(100)
    x[0] = float("nan")
    x[1] = float("inf")

    config = {"log_clamp_hits": True, "remove_nan_to_num": False}

    # Should apply nan_to_num
    y = monitored_nan_to_num(x, nan=0.0, posinf=100.0, name="test", config=config)

    # Check NaN/Inf were replaced
    assert torch.isfinite(y).all()
    assert y[0] == 0.0
    assert y[1] == 100.0


def test_pr4_model_creation():
    """Test model creation with PR-4 configs."""
    config = ModelConfig(
        fusion=FusionConfig(fusion_type="gated", fusion_dropout=0.1),
        clamp_retirement=ClampRetirementConfig(
            remove_intermediate_clamps=False,
            log_clamp_hits=True,
        ),
        graph=GraphConfig(enabled=False),  # Disable GNN for speed
        norms=NormConfig(boundary_norm="layernorm"),  # PR-1 enabled
    )

    detector = SeizureDetector.from_config(config)

    # Check fusion module created
    assert detector.fusion is not None
    assert detector.fusion_type == "gated"

    # Check clamp config stored
    assert detector.clamp_config["log_clamp_hits"] is True
    assert detector.clamp_config["remove_intermediate_clamps"] is False


def test_pr4_with_multihead_fusion():
    """Test multihead fusion integration."""
    config = ModelConfig(
        fusion=FusionConfig(
            fusion_type="multihead",
            fusion_heads=4,
            fusion_dropout=0.1,
        ),
        graph=GraphConfig(enabled=True),  # Need GNN for fusion to apply
        norms=NormConfig(boundary_norm="layernorm"),
    )

    detector = SeizureDetector.from_config(config)

    # Check multihead fusion created
    assert detector.fusion is not None
    assert detector.fusion_type == "multihead"
    assert hasattr(detector.fusion, "num_heads")

    # Test forward pass
    x = torch.randn(2, 19, 15360)
    output = detector(x)
    assert torch.isfinite(output).all()


def test_clamp_retirement_forward():
    """Test forward pass with clamps removed."""
    config = ModelConfig(
        clamp_retirement=ClampRetirementConfig(
            remove_intermediate_clamps=True,  # Remove non-essential clamps
            remove_nan_to_num=True,
            validate_finite=True,
        ),
        norms=NormConfig(boundary_norm="layernorm"),  # PR-1 for stability
        graph=GraphConfig(
            enabled=True,
            edge_lift_activation="tanh",  # PR-2 for bounded edge
            edge_lift_norm="layernorm",
            adj_row_softmax=True,  # PR-3 for stable adjacency
            adj_force_symmetric=True,
        ),
    )

    detector = SeizureDetector.from_config(config)

    # Moderate input (should be stable with PR-1/2/3)
    x = torch.randn(2, 19, 15360) * 2
    output = detector(x)

    # Should still produce finite outputs
    assert torch.isfinite(output).all()


def test_pr4_backward_compatibility():
    """Test PR-4 disabled by default."""
    config = ModelConfig()

    # Fusion should default to "add"
    assert config.fusion.fusion_type == "add"

    # Clamp retirement should be disabled
    assert config.clamp_retirement.remove_intermediate_clamps is False
    assert config.clamp_retirement.remove_nan_to_num is False

    detector = SeizureDetector.from_config(config)

    # No fusion module when type is "add"
    assert detector.fusion is None
    assert detector.fusion_type == "add"


def test_fusion_with_no_gnn():
    """Test fusion doesn't apply without GNN."""
    config = ModelConfig(
        fusion=FusionConfig(fusion_type="gated"),
        graph=GraphConfig(enabled=False),  # No GNN means no edge features
    )

    detector = SeizureDetector.from_config(config)

    # Fusion created but won't be used
    assert detector.fusion is not None

    x = torch.randn(2, 19, 15360)
    output = detector(x)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_gated_fusion()
    test_multihead_fusion()
    test_monitored_clamp_logging()
    test_monitored_clamp_removal()
    test_essential_clamps_preserved()
    test_nan_to_num_monitoring()
    test_pr4_model_creation()
    test_pr4_with_multihead_fusion()
    test_clamp_retirement_forward()
    test_pr4_backward_compatibility()
    test_fusion_with_no_gnn()
    print("All PR-4 tests passed!")