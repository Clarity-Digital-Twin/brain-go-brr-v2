# PHASE2.4_DECODER.md - U-Net Decoder with Skip Connections

## 🎯 Phase 2.4 Goal
Build U-Net decoder that upsamples from bottleneck back to input resolution, fusing skip connections from encoder.

## 📋 Phase 2.4 Checklist
- [ ] 4-stage decoder with transposed convolutions
- [ ] Skip connection fusion at each stage
- [ ] Channel progression [512, 256, 128, 64]
- [ ] Output projection to 19 channels
- [ ] Unit tests for dimension alignment
- [ ] Verify skip ordering is correct

## 🔧 Implementation Files
```
src/experiment/models.py     # Decoder lives here (repo convention)
tests/test_decoder.py        # Decoder-specific tests
```

## 📐 Dimension Flow
```
Input: (B, 512, 960) from Mamba
    ↓ Upsample ×2 → Concat skip[3] (B, 512, 1920) → Conv → (B, 256, 1920)
    ↓ Upsample ×2 → Concat skip[2] (B, 256, 3840) → Conv → (B, 128, 3840)
    ↓ Upsample ×2 → Concat skip[1] (B, 128, 7680) → Conv → (B,  64, 7680)
    ↓ Upsample ×2 → Concat skip[0] (B,  64,15360) → Conv → (B,  64,15360)
    ↓ Conv1d(64→19)
Output: (B, 19, 15360)
```

## 🔨 Implementation

```python
# src/experiment/models.py (decoder section)

import torch
import torch.nn as nn
from typing import List
# Reuse ConvBlock defined in encoder section above


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections and progressive upsampling."""

    def __init__(
        self,
        out_channels: int = 19,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        self.depth = depth

        # Channel progression (reversed): [512, 256, 128, 64]
        channels = [base_channels * (2 ** (depth - 1 - i)) for i in range(depth)]
        # Skip channels (encoder pre-downsample features): [64, 128, 256, 512]
        skip_channels = [base_channels * (2 ** i) for i in range(depth)]

        # Build decoder stages (upsample at every step, total ×16 back to 15360)
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth):
            in_ch = channels[i]
            out_ch = channels[i + 1] if i < depth - 1 else base_channels

            # Transposed convolution for upsampling
            self.upsample.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # After upsample, concatenate with matching skip
            skip_idx = depth - 1 - i
            skip_ch = skip_channels[skip_idx]

            # Double convolution after skip concatenation
            self.decoder_blocks.append(nn.Sequential(
                ConvBlock(out_ch + skip_ch, out_ch),
                ConvBlock(out_ch, out_ch)
            ))

        # Output projection
        self.output_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Bottleneck features (B, 512, 960)
            skips: Skip connections from encoder [4 tensors]
                   Order: [stage1, stage2, stage3, stage4]
                   Shapes: [(B,64,15360), (B,128,7680), (B,256,3840), (B,512,1920)]

        Returns:
            Decoded output (B, 19, 15360)
        """
        # Process decoder stages with skips (deepest → shallowest)
        for i in range(self.depth):
            # Upsample
            x = self.upsample[i](x)

            # Concat with corresponding skip (reverse order: 3,2,1,0)
            skip_idx = self.depth - 1 - i
            x = torch.cat([x, skips[skip_idx]], dim=1)

            # Decoder block
            x = self.decoder_blocks[i](x)

        # Output projection
        return self.output_conv(x)  # (B, 19, 15360)

    def check_skip_compatibility(self, skips: List[torch.Tensor]) -> bool:
        """Verify skip dimensions match expected shapes."""
        expected_shapes = [
            (None, 64, 15360),   # Stage 1
            (None, 128, 7680),   # Stage 2
            (None, 256, 3840),   # Stage 3
            (None, 512, 1920),   # Stage 4
        ]

        if len(skips) != len(expected_shapes):
            return False

        for skip, expected in zip(skips, expected_shapes):
            if skip.shape[1:] != expected[1:]:
                return False

        return True
```

## 🧪 Test Suite

```python
# tests/test_decoder.py

import pytest
import torch
from src.experiment.models import UNetDecoder, UNetEncoder


class TestUNetDecoder:

    @pytest.fixture
    def decoder(self):
        return UNetDecoder(out_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def encoder(self):
        return UNetEncoder(in_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 19, 15360)

    def test_output_shape(self, decoder, encoder, sample_input):
        # Get encoder output and skips
        encoded, skips = encoder(sample_input)

        # Decode
        output = decoder(encoded, skips)

        # Should recover original shape
        assert output.shape == sample_input.shape

    def test_skip_compatibility(self, decoder):
        # Create mock skips with correct shapes
        skips = [
            torch.randn(2, 64, 15360),   # Stage 1
            torch.randn(2, 128, 7680),   # Stage 2
            torch.randn(2, 256, 3840),   # Stage 3
            torch.randn(2, 512, 1920),   # Stage 4
        ]

        assert decoder.check_skip_compatibility(skips)

        # Test with wrong shapes
        bad_skips = [torch.randn(2, 64, 15360)] * 4
        assert not decoder.check_skip_compatibility(bad_skips)

    def test_dimension_progression(self, decoder):
        # Track dimensions through decoder
        x = torch.randn(2, 512, 960)
        skips = [
            torch.randn(2, 64, 15360),
            torch.randn(2, 128, 7680),
            torch.randn(2, 256, 3840),
            torch.randn(2, 512, 1920),
        ]

        # Hook to capture intermediate shapes
        shapes = []
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes.append(output.shape)

        for module in decoder.upsample:
            module.register_forward_hook(hook)

        _ = decoder(x, skips)

        # Check upsampling progression
        expected_lengths = [1920, 3840, 7680]  # After each upsample
        for i, shape in enumerate(shapes[:3]):
            assert shape[2] == expected_lengths[i]

    def test_gradient_flow(self, decoder, encoder, sample_input):
        sample_input.requires_grad = True

        # Full encoder-decoder pass
        encoded, skips = encoder(sample_input)
        output = decoder(encoded, skips)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_no_nan_inf(self, decoder, encoder, sample_input):
        encoded, skips = encoder(sample_input)
        output = decoder(encoded, skips)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_skip_fusion(self, decoder):
        # Test that skips actually affect output
        x = torch.randn(1, 512, 960)

        # Skips with zeros
        zero_skips = [
            torch.zeros(1, 64, 15360),
            torch.zeros(1, 128, 7680),
            torch.zeros(1, 256, 3840),
            torch.zeros(1, 512, 1920),
        ]

        # Skips with signal
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
        assert diff > 0.1  # Significant difference
```

## ✅ Phase 2.4 Completion Criteria
1. Decoder implementation with correct upsampling
2. Skip connection fusion working
3. Tests pass: `pytest tests/test_decoder.py -v`
4. `make q` passes
5. Dimension alignment verified

---
**Status**: Ready for implementation
**Estimated Time**: 0.5 day
**Depends on**: PHASE2.1 (for skip shapes)
**Next**: PHASE2.5_FULL_MODEL.md
