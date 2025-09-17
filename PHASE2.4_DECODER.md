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
src/experiment/models/decoder.py    # U-Net decoder only
tests/test_decoder.py               # Decoder-specific tests
```

## 📐 Dimension Flow
```
Input: (B, 512, 960) from Mamba
    + skip[3]: (B, 512, 1920)
    ↓ Upsample ×2 → Concat → Conv
Stage 4: (B, 256, 1920)
    + skip[2]: (B, 256, 3840)
    ↓ Upsample ×2 → Concat → Conv
Stage 3: (B, 128, 3840)
    + skip[1]: (B, 128, 7680)
    ↓ Upsample ×2 → Concat → Conv
Stage 2: (B, 64, 7680)
    + skip[0]: (B, 64, 15360)
    ↓ Upsample ×2 → Concat → Conv
Stage 1: (B, 64, 15360)
    ↓ Conv1d(64→19)
Output: (B, 19, 15360)
```

## 🔨 Implementation

```python
# src/experiment/models/decoder.py

import torch
import torch.nn as nn
from typing import List
from src.experiment.models.encoder import ConvBlock  # Reuse from encoder


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

        # Build decoder stages
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Transposed convolution for upsampling
            self.upsample.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # Double convolution after skip concatenation
            # Input will be out_ch (upsampled) + out_ch (skip) = 2*out_ch
            self.decoder_blocks.append(nn.Sequential(
                ConvBlock(out_ch * 2, out_ch),
                ConvBlock(out_ch, out_ch)
            ))

        # Final stage processes last skip without upsampling
        # Bottleneck (512) + skip[3] (512) = 1024 channels
        self.final_block = nn.Sequential(
            ConvBlock(channels[0] * 2, channels[0]),
            ConvBlock(channels[0], channels[0])
        )

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
        # First, concatenate with deepest skip
        x = torch.cat([x, skips[3]], dim=1)  # (B, 1024, 960)
        x = self.final_block(x)  # (B, 512, 960)

        # Process decoder stages with remaining skips (reversed)
        for i in range(self.depth - 1):
            # Upsample
            x = self.upsample[i](x)

            # Get corresponding skip (reverse order: 2, 1, 0)
            skip_idx = self.depth - 2 - i
            skip = skips[skip_idx]

            # Concatenate with skip
            x = torch.cat([x, skip], dim=1)

            # Process through decoder block
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
from src.experiment.models.decoder import UNetDecoder
from src.experiment.models.encoder import UNetEncoder


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