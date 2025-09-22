"""Tests for BiMamba2 components with GPU/CPU fallback support."""

import pytest
import torch
import torch.nn as nn

from src.brain_brr.models import MAMBA_AVAILABLE, BiMamba2, BiMamba2Layer


class TestBiMamba2Layer:
    """Test single bidirectional Mamba-2 layer."""

    @pytest.fixture
    def layer(self) -> BiMamba2Layer:
        """Create layer instance."""
        return BiMamba2Layer(d_model=512, d_state=16, d_conv=4)

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Create sample input tensor."""
        return torch.randn(2, 960, 512)  # (B, L, D)

    def test_output_shape(self, layer: BiMamba2Layer, sample_input: torch.Tensor) -> None:
        """Test output preserves shape."""
        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert output.shape == (2, 960, 512)

    def test_bidirectional_processing(self, layer: BiMamba2Layer) -> None:
        """Test bidirectional information flow."""
        x = torch.zeros(1, 960, 512)
        x[:, :100, :] = 1.0  # Signal at start only

        output = layer(x)

        # Output should differ from input due to bidirectional processing
        assert not torch.allclose(output, x, atol=1e-5)

        # Information should propagate backward
        later_signal = output[:, -100:, :].abs().mean()
        assert later_signal > 0.01  # Some signal reached the end

    def test_residual_connection(self, layer: BiMamba2Layer, sample_input: torch.Tensor) -> None:
        """Test residual preserves input information."""
        output = layer(sample_input)

        # Should preserve some input via residual
        correlation = torch.cosine_similarity(sample_input.flatten(), output.flatten(), dim=0)
        assert correlation > 0.3  # Moderate correlation preserved

    @pytest.mark.serial
    def test_gradient_flow(self, layer: BiMamba2Layer, sample_input: torch.Tensor) -> None:
        """Test gradients flow through layer."""
        sample_input.requires_grad = True
        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert sample_input.grad.abs().mean() > 0

    def test_no_nan_inf(self, layer: BiMamba2Layer, sample_input: torch.Tensor) -> None:
        """Test numerical stability."""
        output = layer(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_layer_components(self, layer: BiMamba2Layer) -> None:
        """Test layer has expected components."""
        assert hasattr(layer, "forward_mamba")
        assert hasattr(layer, "backward_mamba")
        assert hasattr(layer, "output_proj")
        assert hasattr(layer, "layer_norm")
        assert hasattr(layer, "dropout")
        assert layer.d_model == 512

    def test_deterministic_eval_mode(
        self, layer: BiMamba2Layer, sample_input: torch.Tensor
    ) -> None:
        """Test deterministic output in eval mode."""
        layer.eval()
        torch.manual_seed(42)

        with torch.no_grad():
            out1 = layer(sample_input.clone())
            out2 = layer(sample_input.clone())

        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    def test_mamba_path_active(self, layer: BiMamba2Layer) -> None:
        """Test real Mamba-2 components when available."""
        assert not isinstance(layer.forward_mamba, nn.Conv1d)
        assert not isinstance(layer.backward_mamba, nn.Conv1d)

    @pytest.mark.skipif(MAMBA_AVAILABLE, reason="Testing fallback only")
    def test_fallback_path_active(self, layer: BiMamba2Layer) -> None:
        """Test Conv1d fallback when Mamba unavailable."""
        assert isinstance(layer.forward_mamba, nn.Conv1d)
        assert isinstance(layer.backward_mamba, nn.Conv1d)
        assert layer.forward_mamba.kernel_size == (5,)
        assert layer.backward_mamba.kernel_size == (5,)


class TestBiMamba2:
    """Test stacked bidirectional Mamba-2 model."""

    @pytest.fixture
    def model(self) -> BiMamba2:
        """Create model instance."""
        return BiMamba2(d_model=512, num_layers=6, d_state=16, d_conv=4)

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Create sample input tensor."""
        return torch.randn(2, 512, 960)  # (B, C, L) from ResCNN

    def test_output_shape(self, model: BiMamba2, sample_input: torch.Tensor) -> None:
        """Test output preserves input shape."""
        output = model(sample_input)
        assert output.shape == sample_input.shape
        assert output.shape == (2, 512, 960)

    def test_num_layers(self, model: BiMamba2) -> None:
        """Test correct number of layers."""
        assert len(model.layers) == 6
        assert all(isinstance(layer, BiMamba2Layer) for layer in model.layers)

    @pytest.mark.serial
    def test_gradient_flow(self, model: BiMamba2, sample_input: torch.Tensor) -> None:
        """Test gradients flow through all layers."""
        sample_input.requires_grad = True
        output = model(sample_input)
        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert sample_input.grad.abs().mean() > 0

    def test_complexity_string(self, model: BiMamba2) -> None:
        """Test complexity analysis string."""
        complexity = model.get_complexity()
        assert "O(N)" in complexity

        if MAMBA_AVAILABLE:
            assert "Mamba-2 SSM" in complexity
        else:
            assert "Conv1d fallback" in complexity

    def test_no_nan_inf(self, model: BiMamba2, sample_input: torch.Tensor) -> None:
        """Test numerical stability through stack."""
        output = model(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_temporal_modeling(self, model: BiMamba2) -> None:
        """Test temporal information propagation."""
        x = torch.zeros(1, 512, 960)
        x[:, :, :50] = 1.0  # Signal at beginning only

        output = model(x)

        # Information should spread temporally
        end_signal = output[:, :, -50:].abs().mean()
        assert end_signal > 0.001  # Signal reached end

    def test_feature_preservation(self, model: BiMamba2, sample_input: torch.Tensor) -> None:
        """Test feature dimension preserved."""
        output = model(sample_input)
        assert output.shape[1] == 512  # Channel dimension preserved

    def test_sequence_length_preservation(
        self, model: BiMamba2, sample_input: torch.Tensor
    ) -> None:
        """Test sequence length preserved."""
        output = model(sample_input)
        assert output.shape[2] == 960  # Sequence length preserved

    def test_parameter_count(self, model: BiMamba2) -> None:
        """Test model has expected parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        # Each layer has projections + norms, expect reasonable count
        assert total_params > 1_000_000  # At least 1M params
        assert total_params < 50_000_000  # Less than 50M params

    def test_integration_with_tcn(self) -> None:
        """Test integration with TCN output."""
        from src.brain_brr.models.tcn import TCNEncoder

        # Create TCN and BiMamba2
        tcn = TCNEncoder()
        bimamba = BiMamba2(d_model=512)

        # Create input
        x = torch.randn(2, 19, 15360)  # Full EEG input

        # Pass through TCN then BiMamba2
        tcn_out = tcn(x)
        assert tcn_out.shape == (2, 512, 960)

        mamba_out = bimamba(tcn_out)
        assert mamba_out.shape == (2, 512, 960)
        assert not torch.isnan(mamba_out).any()


class TestIntegrationPipeline:
    """Test full TCN → BiMamba2 → Projection pipeline."""

    def test_full_pipeline(self) -> None:
        """Test complete TCN pipeline integration."""
        from src.brain_brr.models import BiMamba2
        from src.brain_brr.models.tcn import TCNEncoder, ProjectionHead

        # Create full pipeline
        tcn = TCNEncoder()
        bimamba = BiMamba2(d_model=512)
        proj = ProjectionHead()

        # Create input
        x = torch.randn(2, 19, 15360)  # Full 60s window

        # Forward pass through pipeline
        encoded = tcn(x)
        assert encoded.shape == (2, 512, 960)

        mamba_out = bimamba(encoded)
        assert mamba_out.shape == (2, 512, 960)

        output = proj(mamba_out)
        assert output.shape == (2, 19, 15360)

        # Check no NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.serial
    def test_gradient_flow_pipeline(self) -> None:
        """Test gradients flow through entire pipeline."""
        from src.brain_brr.models import BiMamba2
        from src.brain_brr.models.tcn import TCNEncoder, ProjectionHead

        tcn = TCNEncoder()
        bimamba = BiMamba2(d_model=512)
        proj = ProjectionHead()

        x = torch.randn(2, 19, 15360, requires_grad=True)

        encoded = tcn(x)
        mamba_out = bimamba(encoded)
        output = proj(mamba_out)

        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().mean() > 0
