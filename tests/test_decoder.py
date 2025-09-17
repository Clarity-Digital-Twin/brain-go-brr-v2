"""Tests for U-Net decoder with skip connections."""

import pytest
import torch

from src.experiment.models import UNetDecoder, UNetEncoder


class TestUNetDecoder:
    """Test U-Net decoder component."""

    @pytest.fixture
    def decoder(self) -> UNetDecoder:
        """Create decoder instance."""
        return UNetDecoder(out_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def encoder(self) -> UNetEncoder:
        """Create encoder for integration tests."""
        return UNetEncoder(in_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Create sample EEG input."""
        return torch.randn(2, 19, 15360)

    @pytest.fixture
    def mock_skips(self) -> list[torch.Tensor]:
        """Create mock skip connections with correct shapes."""
        return [
            torch.randn(2, 64, 15360),  # Stage 1 (shallow)
            torch.randn(2, 128, 7680),  # Stage 2
            torch.randn(2, 256, 3840),  # Stage 3
            torch.randn(2, 512, 1920),  # Stage 4 (deep)
        ]

    def test_output_shape(self, decoder: UNetDecoder, encoder: UNetEncoder, sample_input: torch.Tensor) -> None:
        """Test decoder recovers original input dimensions."""
        encoded, skips = encoder(sample_input)
        output = decoder(encoded, skips)

        assert output.shape == sample_input.shape
        assert output.shape == (2, 19, 15360)

    def test_skip_compatibility_check(self, decoder: UNetDecoder, mock_skips: list[torch.Tensor]) -> None:
        """Test skip dimension validation."""
        # Correct shapes
        assert decoder.check_skip_compatibility(mock_skips)

        # Wrong number of skips
        assert not decoder.check_skip_compatibility(mock_skips[:3])

        # Wrong channel dimensions
        bad_skips = [
            torch.randn(2, 32, 15360),  # Wrong channels
            torch.randn(2, 128, 7680),
            torch.randn(2, 256, 3840),
            torch.randn(2, 512, 1920),
        ]
        assert not decoder.check_skip_compatibility(bad_skips)

        # Wrong spatial dimensions
        bad_skips = [
            torch.randn(2, 64, 15360),
            torch.randn(2, 128, 15360),  # Wrong length
            torch.randn(2, 256, 3840),
            torch.randn(2, 512, 1920),
        ]
        assert not decoder.check_skip_compatibility(bad_skips)

    def test_dimension_progression(self, decoder: UNetDecoder, mock_skips: list[torch.Tensor]) -> None:
        """Test upsampling progression through decoder stages."""
        x = torch.randn(2, 512, 960)  # Bottleneck input

        # Hook to capture intermediate shapes
        shapes = []

        def hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                shapes.append(output.shape)

        # Register hooks on upsample layers
        handles = []
        for module in decoder.upsample:
            handles.append(module.register_forward_hook(hook))

        _ = decoder(x, mock_skips)

        # Clean up hooks
        for h in handles:
            h.remove()

        # Check upsampling progression: 960 -> 1920 -> 3840 -> 7680 -> 15360
        expected_lengths = [1920, 3840, 7680, 15360]
        for i, shape in enumerate(shapes[:4]):
            assert shape[2] == expected_lengths[i], f"Stage {i} length mismatch"

    def test_skip_fusion_order(self, decoder: UNetDecoder) -> None:
        """Test skips are used in correct reverse order."""
        x = torch.randn(1, 512, 960)

        # Create skips with unique markers
        skips = [
            torch.ones(1, 64, 15360) * 1.0,   # Marker: 1
            torch.ones(1, 128, 7680) * 2.0,   # Marker: 2
            torch.ones(1, 256, 3840) * 3.0,   # Marker: 3
            torch.ones(1, 512, 1920) * 4.0,   # Marker: 4
        ]

        output = decoder(x, skips)

        # Output should be influenced by all skips
        # Stage 0 decoder uses skip[3] (marker 4)
        # Stage 1 decoder uses skip[2] (marker 3)
        # Stage 2 decoder uses skip[1] (marker 2)
        # Stage 3 decoder uses skip[0] (marker 1)
        assert output.shape == (1, 19, 15360)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self, decoder: UNetDecoder, encoder: UNetEncoder, sample_input: torch.Tensor) -> None:
        """Test gradients flow through encoder-decoder."""
        sample_input.requires_grad = True

        # Full pass
        encoded, skips = encoder(sample_input)
        output = decoder(encoded, skips)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert sample_input.grad.abs().mean() > 0

    def test_no_nan_inf(self, decoder: UNetDecoder, mock_skips: list[torch.Tensor]) -> None:
        """Test numerical stability."""
        x = torch.randn(2, 512, 960)
        output = decoder(x, mock_skips)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_skip_fusion_effect(self, decoder: UNetDecoder) -> None:
        """Test skips actually affect output."""
        x = torch.randn(1, 512, 960)

        # Zero skips
        zero_skips = [
            torch.zeros(1, 64, 15360),
            torch.zeros(1, 128, 7680),
            torch.zeros(1, 256, 3840),
            torch.zeros(1, 512, 1920),
        ]

        # Random skips
        signal_skips = [
            torch.randn(1, 64, 15360),
            torch.randn(1, 128, 7680),
            torch.randn(1, 256, 3840),
            torch.randn(1, 512, 1920),
        ]

        output_zero = decoder(x, zero_skips)
        output_signal = decoder(x, signal_skips)

        # Outputs should differ significantly
        diff = (output_signal - output_zero).abs().mean()
        assert diff > 0.1, "Skips should significantly affect output"

    def test_channel_progression(self, decoder: UNetDecoder) -> None:
        """Test channel dimensions through decoder."""
        assert hasattr(decoder, "upsample")
        assert hasattr(decoder, "decoder_blocks")
        assert hasattr(decoder, "output_conv")

        # Check number of stages
        assert len(decoder.upsample) == 4
        assert len(decoder.decoder_blocks) == 4

        # Check output conv
        assert decoder.output_conv.in_channels == 64
        assert decoder.output_conv.out_channels == 19

    def test_decoder_components(self, decoder: UNetDecoder) -> None:
        """Test decoder has expected architecture."""
        # Verify transposed convolutions
        for i, up in enumerate(decoder.upsample):
            assert isinstance(up, torch.nn.ConvTranspose1d)
            assert up.kernel_size == (2,)
            assert up.stride == (2,)

        # First upsample: 512 -> 256
        assert decoder.upsample[0].in_channels == 512
        assert decoder.upsample[0].out_channels == 256

        # Last upsample: 64 -> 64
        assert decoder.upsample[3].in_channels == 64
        assert decoder.upsample[3].out_channels == 64

    def test_deterministic_eval_mode(self, decoder: UNetDecoder, mock_skips: list[torch.Tensor]) -> None:
        """Test deterministic output in eval mode."""
        decoder.eval()
        x = torch.randn(2, 512, 960)

        with torch.no_grad():
            out1 = decoder(x, mock_skips)
            out2 = decoder(x, mock_skips)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_batch_sizes(self, decoder: UNetDecoder) -> None:
        """Test decoder handles various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 512, 960)
            skips = [
                torch.randn(batch_size, 64, 15360),
                torch.randn(batch_size, 128, 7680),
                torch.randn(batch_size, 256, 3840),
                torch.randn(batch_size, 512, 1920),
            ]

            output = decoder(x, skips)
            assert output.shape == (batch_size, 19, 15360)

    def test_parameter_count(self, decoder: UNetDecoder) -> None:
        """Test decoder has reasonable parameter count."""
        total_params = sum(p.numel() for p in decoder.parameters())
        # Decoder should have fewer params than encoder (no initial projection)
        assert total_params > 100_000  # At least 100k
        assert total_params < 10_000_000  # Less than 10M


class TestEncoderDecoderIntegration:
    """Test full encoder-decoder integration."""

    def test_full_unet_path(self) -> None:
        """Test complete encoder-decoder preserves information."""
        encoder = UNetEncoder(in_channels=19, base_channels=64, depth=4)
        decoder = UNetDecoder(out_channels=19, base_channels=64, depth=4)

        x = torch.randn(2, 19, 15360)

        # Encode
        bottleneck, skips = encoder(x)
        assert bottleneck.shape == (2, 512, 960)
        assert len(skips) == 4

        # Decode
        output = decoder(bottleneck, skips)
        assert output.shape == x.shape

        # Should preserve some information (not zero)
        assert output.abs().mean() > 0.01

    def test_skip_dimension_matching(self) -> None:
        """Test encoder skips match decoder expectations."""
        encoder = UNetEncoder(in_channels=19, base_channels=64, depth=4)
        decoder = UNetDecoder(out_channels=19, base_channels=64, depth=4)

        x = torch.randn(1, 19, 15360)
        _, skips = encoder(x)

        # Verify compatibility
        assert decoder.check_skip_compatibility(skips)

        # Verify exact shapes
        expected_shapes = [
            (1, 64, 15360),
            (1, 128, 7680),
            (1, 256, 3840),
            (1, 512, 1920),
        ]

        for skip, expected in zip(skips, expected_shapes):
            assert skip.shape == expected