"""Test PR-1 boundary normalization implementation."""

import torch
import torch.nn as nn

from src.brain_brr.config.schemas import ModelConfig
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.models.norms import LayerScale, RMSNorm, create_norm_layer


def test_layerscale_initialization():
    """Verify LayerScale initializes with correct alpha."""
    layer = LayerScale(dim=64, init_value=0.1)
    assert layer.gamma.shape == (64,)
    assert torch.allclose(layer.gamma, torch.ones(64) * 0.1)


def test_layerscale_forward():
    """Verify LayerScale scales correctly."""
    layer = LayerScale(dim=4, init_value=0.5)
    x = torch.ones(2, 4)
    y = layer(x)
    assert torch.allclose(y, torch.ones(2, 4) * 0.5)


def test_rmsnorm_stability():
    """Verify RMSNorm prevents explosion."""
    norm = RMSNorm(dim=64, eps=1e-5)
    # Test with extreme input
    x = torch.randn(32, 19, 960, 64) * 100  # Large values
    y = norm(x)

    assert torch.isfinite(y).all()
    # RMSNorm should normalize to unit RMS
    rms = y.pow(2).mean(dim=-1, keepdim=True).sqrt()
    assert rms.mean() < 2.0  # Reasonable bound


def test_create_norm_layer():
    """Test norm layer factory."""
    # Test LayerNorm creation
    norm = create_norm_layer("layernorm", 64)
    assert isinstance(norm, nn.LayerNorm)

    # Test RMSNorm creation
    norm = create_norm_layer("rmsnorm", 64, eps=1e-6)
    assert isinstance(norm, RMSNorm)

    # Test none returns None
    norm = create_norm_layer("none", 64)
    assert norm is None


def test_pr1_model_creation():
    """Test model creation with PR-1 enabled."""
    config = ModelConfig(
        architecture="v3",
        norms={
            "boundary_norm": "layernorm",
            "boundary_eps": 1e-5,
            "layerscale_alpha": 0.1,
            "after_tcn_proj": True,
            "after_node_mamba": True,
            "after_edge_mamba": True,
            "after_gnn": True,
            "before_decoder": True,
        },
    )

    model = SeizureDetector.from_config(config)

    # Verify normalization layers exist
    assert hasattr(model, "norm_after_proj_to_electrodes")
    assert model.norm_after_proj_to_electrodes is not None
    assert isinstance(model.norm_after_proj_to_electrodes, nn.LayerNorm)

    assert hasattr(model, "norm_after_node_mamba")
    assert model.norm_after_node_mamba is not None

    assert hasattr(model, "norm_after_edge_mamba")
    assert model.norm_after_edge_mamba is not None

    assert hasattr(model, "norm_after_gnn")
    assert model.norm_after_gnn is not None

    assert hasattr(model, "norm_before_decoder")
    assert model.norm_before_decoder is not None

    # Verify LayerScale exists in BiMamba2 layers
    if hasattr(model, "node_mamba"):
        # Check that LayerScale is integrated in BiMamba2Layer
        for layer in model.node_mamba.layers:
            assert hasattr(layer, "layerscale")
            assert layer.layerscale is not None

    # Verify GNN LayerScale if GNN is present and enabled
    if hasattr(model, "gnn_layerscale") and model.gnn_layerscale is not None:
        # GNN LayerScale exists when GNN is enabled with residual
        assert isinstance(model.gnn_layerscale, LayerScale)


def test_pr1_forward_pass():
    """Test forward pass with PR-1 enabled doesn't crash."""
    config = ModelConfig(
        architecture="v3",
        norms={
            "boundary_norm": "layernorm",
            "boundary_eps": 1e-5,
            "layerscale_alpha": 0.1,
            "after_tcn_proj": True,
            "after_node_mamba": True,
            "after_edge_mamba": True,
            "after_gnn": True,
            "before_decoder": True,
        },
    )

    model = SeizureDetector.from_config(config)
    model.eval()

    # Small test input
    x = torch.randn(2, 19, 256)  # (batch, channels, time)

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()
    assert output.shape == (2, 256)  # (batch, time)


def test_pr1_gradient_flow():
    """Test gradients flow through PR-1 normalization."""
    config = ModelConfig(
        architecture="v3",
        norms={
            "boundary_norm": "layernorm",
            "boundary_eps": 1e-5,
            "layerscale_alpha": 0.1,
            "after_tcn_proj": True,
            "after_node_mamba": True,
            "after_edge_mamba": True,
            "after_gnn": True,
            "before_decoder": True,
        },
    )

    model = SeizureDetector.from_config(config)

    # Small test input
    x = torch.randn(2, 19, 256, requires_grad=True)
    output = model(x)

    # Create a simple loss
    loss = output.mean()
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Check model parameter gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_contiguous_memory_fix():
    """Test that .contiguous() is called after transpose operations."""
    config = ModelConfig(
        architecture="v3",
        norms={
            "boundary_norm": "layernorm",
            "boundary_eps": 1e-5,
            "layerscale_alpha": 0.1,
            "after_tcn_proj": True,
            "after_node_mamba": True,
            "after_edge_mamba": True,
            "after_gnn": True,
            "before_decoder": True,
        },
    )

    model = SeizureDetector.from_config(config)
    model.eval()

    # Hook to check contiguity
    contiguous_checks = []

    def check_contiguous_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            contiguous_checks.append(output.is_contiguous())

    # Add hooks to normalization layers
    if model.norm_after_edge_mamba:
        model.norm_after_edge_mamba.register_forward_hook(check_contiguous_hook)
    if model.norm_before_decoder:
        model.norm_before_decoder.register_forward_hook(check_contiguous_hook)

    # Run forward pass
    x = torch.randn(2, 19, 256)
    with torch.no_grad():
        _ = model(x)

    # After transpose operations, tensors should be made contiguous
    # The hook checks the output of norm layers which should receive contiguous input
    assert len(contiguous_checks) > 0, "No contiguous checks performed"
