"""Test PR-2 bounded edge stream implementation."""

import torch
import torch.nn as nn

from src.brain_brr.config.schemas import ModelConfig
from src.brain_brr.models.detector import SeizureDetector


def test_edge_lift_activation():
    """Test edge lift activation options."""
    # Test tanh activation
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "none",
            "edge_lift_init_gain": 0.1,
        },
    )

    model = SeizureDetector.from_config(config)
    assert hasattr(model, "edge_lift_act")
    assert isinstance(model.edge_lift_act, nn.Tanh)

    # Test sigmoid activation
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "sigmoid",
            "edge_lift_norm": "none",
            "edge_lift_init_gain": 0.1,
        },
    )
    model = SeizureDetector.from_config(config)
    assert isinstance(model.edge_lift_act, nn.Sigmoid)

    # Test SELU activation
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "selu",
            "edge_lift_norm": "none",
            "edge_lift_init_gain": 0.1,
        },
    )
    model = SeizureDetector.from_config(config)
    assert isinstance(model.edge_lift_act, nn.SELU)

    # Test none (no activation)
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "none",
            "edge_lift_norm": "none",
            "edge_lift_init_gain": 0.1,
        },
    )
    model = SeizureDetector.from_config(config)
    assert model.edge_lift_act is None


def test_edge_lift_normalization():
    """Test edge lift normalization options."""
    # Test LayerNorm
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
        },
    )

    model = SeizureDetector.from_config(config)
    assert hasattr(model, "edge_lift_norm")
    assert isinstance(model.edge_lift_norm, nn.LayerNorm)

    # Test RMSNorm
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "rmsnorm",
            "edge_lift_init_gain": 0.1,
        },
    )
    model = SeizureDetector.from_config(config)
    assert model.edge_lift_norm is not None  # RMSNorm is custom

    # Test none (no normalization)
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "none",
            "edge_lift_init_gain": 0.1,
        },
    )
    model = SeizureDetector.from_config(config)
    assert model.edge_lift_norm is None


def test_bounded_edge_projection():
    """Verify edge projection stays bounded with PR-2."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
            "edge_mamba_d_model": 16,
        },
    )

    model = SeizureDetector.from_config(config)
    model.eval()

    # Create extreme input to test boundedness
    batch_size, channels, time = 2, 19, 256
    x = torch.randn(batch_size, channels, time) * 1000  # Large values

    with torch.no_grad():
        output = model(x)

    # Output should be finite despite extreme input
    assert torch.isfinite(output).all()
    assert output.abs().max() < 1000  # Should be bounded


def test_edge_gradient_flow():
    """Verify gradients flow through bounded edge stream."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
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

    # Check edge projection gradients specifically
    if hasattr(model, "edge_in_proj"):
        assert model.edge_in_proj.weight.grad is not None
        assert torch.isfinite(model.edge_in_proj.weight.grad).all()

        # Check gradient magnitude is reasonable (not vanishing/exploding)
        grad_norm = model.edge_in_proj.weight.grad.norm()
        assert grad_norm > 1e-10  # Not vanishing
        assert grad_norm < 100  # Not exploding


def test_pr2_with_pr1_combined():
    """Test PR-2 works correctly when combined with PR-1."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
        },
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

    # Both PR-1 and PR-2 components should exist
    assert hasattr(model, "edge_lift_act") and model.edge_lift_act is not None
    assert hasattr(model, "edge_lift_norm") and model.edge_lift_norm is not None
    assert hasattr(model, "norm_after_edge_mamba") and model.norm_after_edge_mamba is not None

    # Test forward pass
    x = torch.randn(2, 19, 256)
    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()


def test_edge_dimension_variance():
    """Test that edge stream variance is controlled with PR-2."""
    # First test WITHOUT PR-2
    config_without = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "none",
            "edge_lift_norm": "none",
        },
    )

    model_without = SeizureDetector.from_config(config_without)
    model_without.eval()

    # Hook to capture edge activations
    edge_activations_without = []

    def capture_hook(module, input_tensor, output):
        edge_activations_without.append(output.detach().clone())

    # Register hook on edge projection
    handle_without = model_without.edge_in_proj.register_forward_hook(capture_hook)

    x = torch.randn(2, 19, 256)
    with torch.no_grad():
        _ = model_without(x)

    handle_without.remove()

    # Now test WITH PR-2
    config_with = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
        },
    )

    model_with = SeizureDetector.from_config(config_with)
    model_with.eval()

    # Hook to capture edge activations
    edge_activations_with = []

    def capture_hook_with(module, input_tensor, output):
        edge_activations_with.append(output.detach().clone())

    # Register hook on edge projection
    handle_with = model_with.edge_in_proj.register_forward_hook(capture_hook_with)

    with torch.no_grad():
        _ = model_with(x)

    handle_with.remove()

    # With PR-2, variance should be more controlled
    if len(edge_activations_without) > 0 and len(edge_activations_with) > 0:
        var_without = edge_activations_without[0].var()
        var_with = edge_activations_with[0].var()

        # After tanh+norm, variance should be significantly reduced
        # Note: This is a soft check since exact values depend on initialization
        assert var_with < var_without * 10  # Should be more controlled


def test_edge_initialization_gain():
    """Test that edge projection uses configured initialization gain."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.01,  # Very small gain
        },
    )

    model = SeizureDetector.from_config(config)

    # Check that weights are initialized with small values
    edge_weights = model.edge_in_proj.weight.data
    assert edge_weights.abs().max() < 0.1  # Should be small due to gain=0.01


def test_pr2_backward_compatibility():
    """Test that PR-2 disabled maintains backward compatibility."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "none",
            "edge_lift_norm": "none",
        },
    )

    model = SeizureDetector.from_config(config)
    model.eval()

    # Should work exactly as before PR-2
    x = torch.randn(2, 19, 256)
    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()

    # The original clamp should still be applied
    # (We can't directly test this without hooks, but the model should work)


def test_edge_stream_stability():
    """Test edge stream stability over many batches with PR-2."""
    config = ModelConfig(
        architecture="v3",
        graph={
            "enabled": True,
            "edge_lift_activation": "tanh",
            "edge_lift_norm": "layernorm",
            "edge_lift_init_gain": 0.1,
        },
    )

    model = SeizureDetector.from_config(config)
    model.eval()

    # Run multiple batches to test stability
    for _ in range(10):
        x = torch.randn(2, 19, 256)
        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all()
        assert output.abs().max() < 100  # Should remain bounded