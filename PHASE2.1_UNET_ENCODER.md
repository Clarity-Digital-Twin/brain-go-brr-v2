# PHASE2.1_UNET_ENCODER.md - U-Net Encoder Implementation

## 🎯 Phase 2.1 Goal
Build and test the U-Net encoder with 4-stage progressive downsampling and skip connections for spatial feature extraction.

## 📋 Phase 2.1 Checklist
- [ ] ConvBlock helper (Conv1d + BatchNorm + ReLU)
- [ ] 4-stage encoder with channel progression [64, 128, 256, 512]
- [ ] Skip connection collection
- [ ] Dimension validation at each stage
- [ ] Unit tests for shape preservation
- [ ] Forward pass test on sample data

## 🔧 Implementation Files
```
src/experiment/models.py     # U-Net encoder lives here (repo convention)
tests/test_encoder.py        # Encoder-specific tests
```

## 📐 Dimension Flow
```
Input: (B, 19, 15360)
  ↓ Input Conv (19→64)
(B, 64, 15360)
  ↓ Encoder Block 1 (64→64)
(B, 64, 15360) → skip[0]
  ↓ Downsample ×2
(B, 64, 7680)
  ↓ Encoder Block 2 (64→128)
(B, 128, 7680) → skip[1]
  ↓ Downsample ×2
(B, 128, 3840)
  ↓ Encoder Block 3 (128→256)
(B, 256, 3840) → skip[2]
  ↓ Downsample ×2
(B, 256, 1920)
  ↓ Encoder Block 4 (256→512)
(B, 512, 1920) → skip[3]
  ↓ Downsample ×2
(B, 512, 960) [Output]
```

## 🔨 Implementation

```python
# src/experiment/models.py (encoder section)

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Basic convolutional building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UNetEncoder(nn.Module):
    """U-Net encoder with progressive downsampling."""

    def __init__(
        self,
        in_channels: int = 19,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        self.depth = depth

        # Channel progression: [64, 128, 256, 512]
        channels = [base_channels * (2 ** i) for i in range(depth)]

        # Initial projection from 19 → 64 channels
        self.input_conv = ConvBlock(in_channels, channels[0], kernel_size=7, padding=3)

        # Build encoder stages
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(depth):
            # Encoder blocks take previous stage's output channels as input
            in_ch = channels[0] if i == 0 else channels[i-1]
            out_ch = channels[i]

            # Double convolution block
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(in_ch, out_ch),
                    ConvBlock(out_ch, out_ch),
                )
            )

            # Downsample maintains channel count
            self.downsample.append(
                nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input (B, 19, 15360)

        Returns:
            encoded: Final encoding (B, 512, 960)
            skips: List of 4 skip tensors for decoder
        """
        skips = []

        # Initial projection
        x = self.input_conv(x)  # (B, 64, 15360)

        # Process through encoder stages
        for i in range(self.depth):
            # Process through encoder block
            x = self.encoder_blocks[i](x)

            # Save skip AFTER block, BEFORE downsample
            skips.append(x)

            # Downsample for next stage
            x = self.downsample[i](x)

        # After loop: x is (B, 512, 960), skips are [(64,15360), (128,7680), (256,3840), (512,1920)]
        return x, skips
```

## 🧪 Test Suite

```python
# tests/test_encoder.py

import pytest
import torch
from src.experiment.models import UNetEncoder


class TestUNetEncoder:

    @pytest.fixture
    def encoder(self):
        return UNetEncoder(in_channels=19, base_channels=64, depth=4)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 19, 15360)  # Batch of 2

    def test_output_shape(self, encoder, sample_input):
        encoded, skips = encoder(sample_input)

        # Check final encoding (×16 downsample → 15360/16=960)
        assert encoded.shape == (2, 512, 960)

        # Check skip shapes
        expected_skip_shapes = [
            (2, 64, 15360),   # Stage 1
            (2, 128, 7680),   # Stage 2
            (2, 256, 3840),   # Stage 3
            (2, 512, 1920),   # Stage 4
        ]

        assert len(skips) == 4
        for skip, expected_shape in zip(skips, expected_skip_shapes):
            assert skip.shape == expected_shape

    def test_no_nan_inf(self, encoder, sample_input):
        encoded, skips = encoder(sample_input)

        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()

        for skip in skips:
            assert not torch.isnan(skip).any()
            assert not torch.isinf(skip).any()

    def test_gradient_flow(self, encoder, sample_input):
        sample_input.requires_grad = True
        encoded, _ = encoder(sample_input)

        # Simulate loss
        loss = encoded.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
```

## ✅ Phase 2.1 Completion Criteria
1. Encoder implementation with correct dimensions
2. Tests pass: `pytest tests/test_encoder.py -v`
3. `make q` passes
4. Skip connections verified for decoder use

---
**Status**: Ready for implementation
**Estimated Time**: 0.5 day
**Next**: PHASE2.2_RESCNN_STACK.md
