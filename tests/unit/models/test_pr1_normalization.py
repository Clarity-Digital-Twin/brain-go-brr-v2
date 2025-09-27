"""Tests for PR-1: Boundary Normalization implementation.

These tests verify that normalization at component boundaries
prevents activation explosion and gradient instability.
"""

import torch
import torch.nn as nn

from src.brain_brr.config.schemas import (
    GraphConfig,
    MambaConfig,
    ModelConfig,
    NormConfig,
    TCNConfig,
)
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.models.norms import LayerScale, RMSNorm, create_norm_layer


def test_norm_layer_creation():
    """Test that normalization layers are created correctly."""
    # Test LayerNorm creation
    norm = create_norm_layer("layernorm", 64, 1e-5)
    assert isinstance(norm, nn.LayerNorm)
    assert norm.normalized_shape == (64,)

    # Test RMSNorm creation
    norm = create_norm_layer("rmsnorm", 128, 1e-6)
    assert isinstance(norm, RMSNorm)
    assert norm.scale.shape == (128,)

    # Test None creation
    norm = create_norm_layer("none", 256)
    assert norm is None


def test_rmsnorm_stability():
    """Verify RMSNorm prevents explosion with large inputs."""
    x = torch.randn(32, 19, 960, 64) * 100  # Large input
    norm = RMSNorm(64)
    y = norm(x)

    assert torch.isfinite(y).all()
    assert y.std() < 10  # Bounded variance
    assert y.shape == x.shape


def test_layerscale_initialization():
    """Verify LayerScale initializes correctly."""
    ls = LayerScale(64, init_value=0.1)
    assert ls.gamma.shape == (64,)
    assert torch.allclose(ls.gamma, torch.ones(64) * 0.1)

    # Test forward pass
    x = torch.randn(2, 19, 960, 64)
    y = ls(x)
    assert torch.allclose(y, x * 0.1)


def test_detector_with_boundary_norms():
    """Test SeizureDetector with boundary normalization enabled."""
    # Create config with normalization enabled
    config = ModelConfig(
        architecture="v3",
        tcn=TCNConfig(),
        mamba=MambaConfig(),
        graph=GraphConfig(
            enabled=True,
            use_dynamic_pe=False,  # Disable for simplicity
        ),
        norms=NormConfig(
            boundary_norm="layernorm",
            boundary_eps=1e-5,
            layerscale_alpha=0.1,
            after_tcn_proj=True,
            after_node_mamba=True,
            after_edge_mamba=True,
            after_gnn=True,
            before_decoder=True,
        ),
    )

    # Create model
    model = SeizureDetector.from_config(config)

    # Verify normalization layers were created
    assert model.norm_after_proj_to_electrodes is not None
    assert isinstance(model.norm_after_proj_to_electrodes, nn.LayerNorm)
    assert model.norm_after_node_mamba is not None
    assert model.norm_after_edge_mamba is not None
    assert model.norm_after_gnn is not None
    assert model.norm_before_decoder is not None

    # Verify LayerScale was created for GNN residual
    assert model.gnn_layerscale is not None
    assert isinstance(model.gnn_layerscale, LayerScale)


def test_forward_pass_with_norms():
    """Test full forward pass with normalization enabled."""
    # Create config with normalization
    config = ModelConfig(
        architecture="v3",
        tcn=TCNConfig(),
        mamba=MambaConfig(),
        graph=GraphConfig(
            enabled=True,
            use_dynamic_pe=False,
            use_residual=True,
        ),
        norms=NormConfig(
            boundary_norm="rmsnorm",  # Use RMSNorm for this test
            boundary_eps=1e-5,
            layerscale_alpha=0.1,
        ),
    )

    # Create model
    model = SeizureDetector.from_config(config)
    model.eval()

    # Create large input to test stability
    x = torch.randn(2, 19, 15360) * 10  # Moderate input

    with torch.no_grad():
        output = model(x)

    # Verify output is finite and bounded
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    assert output.abs().max() < 1000, "Output explosion detected"
    assert output.shape == (2, 15360), "Unexpected output shape"


def test_gradient_flow_with_norms():
    """Test gradient flow through normalized architecture."""
    config = ModelConfig(
        architecture="v3",
        tcn=TCNConfig(num_layers=4),  # Minimum allowed
        mamba=MambaConfig(n_layers=2),  # Smaller for test
        graph=GraphConfig(
            enabled=True,
            use_dynamic_pe=False,
            n_layers=1,  # Smaller for test
        ),
        norms=NormConfig(
            boundary_norm="layernorm",
            layerscale_alpha=0.1,
        ),
    )

    model = SeizureDetector.from_config(config)

    # Small batch for gradient test
    x = torch.randn(1, 19, 15360, requires_grad=True)
    target = torch.randint(0, 2, (1, 15360)).float()

    # Forward pass
    output = model(x)

    # Compute loss
    loss = nn.BCEWithLogitsLoss()(output, target)

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None, "Input gradients missing"
    assert torch.isfinite(x.grad).all(), "Input gradients contain NaN/Inf"

    # Check gradient magnitudes are reasonable
    grad_norm = x.grad.norm()
    assert grad_norm < 100, f"Gradient explosion detected: {grad_norm}"
    assert grad_norm > 1e-6, f"Gradient vanishing detected: {grad_norm}"


def test_backward_compatibility():
    """Test that normalization can be disabled for backward compatibility."""
    # Config with normalization disabled
    config = ModelConfig(
        architecture="v3",
        tcn=TCNConfig(),
        mamba=MambaConfig(),
        graph=GraphConfig(enabled=False),
        norms=NormConfig(
            boundary_norm="none",  # Disabled
        ),
    )

    model = SeizureDetector.from_config(config)

    # Verify no normalization layers were created
    assert model.norm_after_proj_to_electrodes is None
    assert model.norm_after_node_mamba is None
    assert model.norm_after_edge_mamba is None
    assert model.norm_after_gnn is None
    assert model.norm_before_decoder is None
    assert model.gnn_layerscale is None

    # Model should still work without norms
    x = torch.randn(1, 19, 15360)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 15360)


def test_selective_norm_locations():
    """Test that individual norm locations can be controlled."""
    config = ModelConfig(
        architecture="v3",
        norms=NormConfig(
            boundary_norm="layernorm",
            after_tcn_proj=True,
            after_node_mamba=False,  # Disabled
            after_edge_mamba=True,
            after_gnn=False,  # Disabled
            before_decoder=True,
        ),
    )

    model = SeizureDetector.from_config(config)

    # Check selective creation
    assert model.norm_after_proj_to_electrodes is not None
    assert model.norm_after_node_mamba is None  # Should be None
    assert model.norm_after_edge_mamba is not None
    assert model.norm_after_gnn is None  # Should be None
    assert model.norm_before_decoder is not None


if __name__ == "__main__":
    test_norm_layer_creation()
    test_rmsnorm_stability()
    test_layerscale_initialization()
    test_detector_with_boundary_norms()
    test_forward_pass_with_norms()
    test_gradient_flow_with_norms()
    test_backward_compatibility()
    test_selective_norm_locations()
    print("✅ All PR-1 normalization tests passed!")
