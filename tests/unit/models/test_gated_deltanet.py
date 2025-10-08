"""Unit tests for BiGatedDeltaNet wrapper."""

import pytest
import torch

try:
    from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA library not installed")
class TestBiGatedDeltaNetInstantiation:
    """Test BiGatedDeltaNet instantiation and constraints (CPU-safe)."""

    def test_node_stream_default_config(self):
        """Test node stream with default configuration."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=6,
            conv_size=4,
            dropout=0.1,
            fusion_mode="sum",
            allow_neg_eigval=False,
        )
        assert model.d_model == 64
        assert model.fusion_mode == "sum"
        assert model.fusion_proj is None
        assert len(model.layers) == 6

    def test_edge_stream_default_config(self):
        """Test edge stream with default configuration."""
        model = BiGatedDeltaNet(
            d_model=16,
            headdim=4,
            num_layers=2,
            conv_size=4,
            dropout=0.1,
            fusion_mode="sum",
            allow_neg_eigval=False,
        )
        assert model.d_model == 16
        assert model.fusion_mode == "sum"
        assert model.fusion_proj is None
        assert len(model.layers) == 2

    def test_concat_fusion_creates_projection(self):
        """Test concat fusion mode creates projection layer."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=2,
            conv_size=4,
            fusion_mode="concat",
        )
        assert model.fusion_mode == "concat"
        assert model.fusion_proj is not None
        assert model.fusion_proj.in_features == 128
        assert model.fusion_proj.out_features == 64

    def test_075_constraint_node_stream(self):
        """Test 0.75x constraint validation for node stream."""
        BiGatedDeltaNet(d_model=64, headdim=8, num_layers=1, conv_size=4)

    def test_075_constraint_edge_stream(self):
        """Test 0.75x constraint validation for edge stream."""
        BiGatedDeltaNet(d_model=16, headdim=4, num_layers=1, conv_size=4)

    def test_constraint_violation_raises(self):
        """Test that invalid headdim raises assertion."""
        with pytest.raises(AssertionError, match="Invalid headdim"):
            BiGatedDeltaNet(d_model=64, headdim=7, num_layers=1, conv_size=4)

    def test_conv_size_parameter_used(self):
        """Test conv_size parameter is passed to FLA layers."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=1,
            conv_size=2,
        )
        assert model.layers[0]["fwd"].conv_size == 2
        assert model.layers[0]["bwd"].conv_size == 2


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA library not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for forward pass")
class TestBiGatedDeltaNetForward:
    """Test BiGatedDeltaNet forward pass (requires CUDA)."""

    def test_node_stream_shape_preservation(self):
        """Test node stream preserves shape through forward pass."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=2,
            conv_size=4,
            fusion_mode="sum",
        ).cuda()

        batch, nodes, channels, length = 2, 19, 64, 960
        x = torch.randn(batch * nodes, channels, length, device="cuda")

        y = model(x)
        assert y.shape == (batch * nodes, channels, length)
        assert y.device.type == "cuda"

    def test_edge_stream_shape_preservation(self):
        """Test edge stream preserves shape through forward pass."""
        model = BiGatedDeltaNet(
            d_model=16,
            headdim=4,
            num_layers=2,
            conv_size=4,
            fusion_mode="sum",
        ).cuda()

        batch, edges, channels, length = 2, 171, 16, 960
        x = torch.randn(batch * edges, channels, length, device="cuda")

        y = model(x)
        assert y.shape == (batch * edges, channels, length)
        assert y.device.type == "cuda"

    def test_concat_fusion_shape_preservation(self):
        """Test concat fusion preserves final shape."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=2,
            conv_size=4,
            fusion_mode="concat",
        ).cuda()

        x = torch.randn(2, 64, 960, device="cuda")
        y = model(x)
        assert y.shape == (2, 64, 960)

    def test_gradient_flow(self):
        """Test gradients flow through bidirectional layers."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=2,
            conv_size=4,
        ).cuda()

        x = torch.randn(2, 64, 960, device="cuda", requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


def test_fla_unavailable_raises_import_error():
    """Test that missing FLA library raises helpful error."""
    if FLA_AVAILABLE:
        pytest.skip("FLA is available, cannot test unavailable case")

    from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

    with pytest.raises(ImportError, match="flash-linear-attention library required"):
        BiGatedDeltaNet(d_model=64, headdim=8, num_layers=1, conv_size=4)
