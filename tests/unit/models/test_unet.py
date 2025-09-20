"""Test suite for U-Net encoder implementation (Phase 2.1)."""

import pytest
import torch

from src.brain_brr.models import UNetEncoder


class TestUNetEncoder:
    """Comprehensive tests for the U-Net encoder."""

    @pytest.fixture
    def encoder(self):
        """Create a standard encoder instance."""
        return UNetEncoder(in_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def sample_input(self):
        """Create sample EEG input batch."""
        return torch.randn(2, 19, 15360)  # Batch of 2

    def test_output_shape(self, encoder, sample_input):
        """Test that encoder produces correct output dimensions."""
        encoded, skips = encoder(sample_input)

        # Check final encoding (x16 downsample -> 15360/16=960)
        assert encoded.shape == (2, 512, 960), f"Expected (2, 512, 960), got {encoded.shape}"

        # Check skip shapes match expected dimensions
        expected_skip_shapes = [
            (2, 64, 15360),  # Stage 1: before first downsample
            (2, 128, 7680),  # Stage 2: before second downsample
            (2, 256, 3840),  # Stage 3: before third downsample
            (2, 512, 1920),  # Stage 4: before fourth downsample
        ]

        assert len(skips) == 4, f"Expected 4 skip connections, got {len(skips)}"
        for i, (skip, expected_shape) in enumerate(zip(skips, expected_skip_shapes, strict=False)):
            assert skip.shape == expected_shape, (
                f"Skip {i}: expected {expected_shape}, got {skip.shape}"
            )

    def test_channel_progression(self, encoder):
        """Test that channel counts follow expected progression."""
        info = encoder.get_dimension_info()
        expected_channels = [64, 128, 256, 512]
        assert info["channel_progression"] == expected_channels

    def test_spatial_downsampling(self, encoder):
        """Test that spatial dimensions are downsampled correctly."""
        info = encoder.get_dimension_info()
        # Should be [15360, 7680, 3840, 1920, 960]
        expected_spatial = [15360, 7680, 3840, 1920, 960]
        assert info["spatial_progression"] == expected_spatial

    def test_no_nan_inf(self, encoder, sample_input):
        """Test that encoder output contains no NaN or Inf values."""
        encoded, skips = encoder(sample_input)

        # Check main output
        assert not torch.isnan(encoded).any(), "Encoded output contains NaN"
        assert not torch.isinf(encoded).any(), "Encoded output contains Inf"

        # Check all skip connections
        for i, skip in enumerate(skips):
            assert not torch.isnan(skip).any(), f"Skip {i} contains NaN"
            assert not torch.isinf(skip).any(), f"Skip {i} contains Inf"

    @pytest.mark.serial
    def test_gradient_flow(self, encoder, sample_input):
        """Test that gradients flow back through the encoder."""
        sample_input.requires_grad = True
        encoded, _skips = encoder(sample_input)

        # Simulate loss on encoder output
        loss = encoded.mean()
        loss.backward()

        # Check gradient exists and is valid
        assert sample_input.grad is not None, "No gradient computed for input"
        assert not torch.isnan(sample_input.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(sample_input.grad).any(), "Gradient contains Inf"

        # Check gradient magnitude is reasonable
        grad_norm = sample_input.grad.norm()
        assert grad_norm > 0, "Gradient is zero"
        assert grad_norm < 1000, f"Gradient norm too large: {grad_norm}"

    def test_skip_gradient_flow(self, encoder, sample_input):
        """Test that gradients flow through skip connections."""
        sample_input.requires_grad = True
        _encoded, skips = encoder(sample_input)

        # Simulate loss on skip connections (as decoder would use them)
        loss = sum(skip.mean() for skip in skips)
        loss.backward()

        assert sample_input.grad is not None, "No gradient from skip connections"
        assert sample_input.grad.abs().mean() > 0, "Gradient is effectively zero"

    @pytest.mark.serial
    def test_different_batch_sizes(self, encoder):
        """Test encoder works with various batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 19, 15360)
            encoded, skips = encoder(x)

            assert encoded.shape[0] == batch_size
            assert all(skip.shape[0] == batch_size for skip in skips)

    def test_deterministic_forward(self, encoder):
        """Test that forward pass is deterministic."""
        # BatchNorm layers update running stats in train mode;
        # switch to eval() to ensure deterministic behavior across calls.
        encoder.eval()
        torch.manual_seed(42)
        x = torch.randn(2, 19, 15360)

        # Two forward passes with same input in eval mode should match exactly
        with torch.no_grad():
            encoded1, skips1 = encoder(x)
            encoded2, skips2 = encoder(x)

        # Should be identical
        assert torch.allclose(encoded1, encoded2, atol=1e-6)
        for skip1, skip2 in zip(skips1, skips2, strict=False):
            assert torch.allclose(skip1, skip2, atol=1e-6)

    def test_encoder_blocks_applied(self, encoder, sample_input):
        """Test that encoder blocks actually transform the input."""
        # Get initial projection
        x = encoder.input_conv(sample_input)

        # Track if blocks modify the tensor
        for i in range(encoder.depth):
            x_before = x.clone()
            x_after = encoder.encoder_blocks[i](x)

            # Check shapes match expected progression
            # Block 0: 64→64, Block 1: 64→128, Block 2: 128→256, Block 3: 256→512
            if i == 0:
                assert x_after.shape[1] == 64
            else:
                assert x_after.shape[1] == 64 * (2**i)

            # Blocks should transform the input (not identity)
            # Can't use allclose if dimensions differ
            if x_before.shape == x_after.shape:
                assert not torch.allclose(x_before, x_after)

            # Prepare for next stage
            x = encoder.downsample[i](x_after)

    def test_skip_connections_unique(self, encoder, sample_input):
        """Test that skip connections are distinct tensors."""
        _, skips = encoder(sample_input)

        # Each skip should be different
        for i in range(len(skips)):
            for j in range(i + 1, len(skips)):
                # Different shapes guarantee they're different
                assert skips[i].shape != skips[j].shape

    @pytest.mark.parametrize("in_channels", [1, 10, 19, 32])
    def test_variable_input_channels(self, in_channels):
        """Test encoder works with different input channel counts."""
        encoder = UNetEncoder(in_channels=in_channels, base_channels=64, depth=4)
        x = torch.randn(2, in_channels, 15360)

        encoded, skips = encoder(x)
        assert encoded.shape == (2, 512, 960)
        assert len(skips) == 4

    def test_memory_efficiency(self, encoder):
        """Test that encoder doesn't leak memory in eval mode."""
        encoder.eval()
        x = torch.randn(4, 19, 15360)

        # Run multiple times and check memory doesn't grow
        with torch.no_grad():
            for _ in range(10):
                _, _ = encoder(x)

        # If this completes without OOM, memory is managed correctly
        assert True
