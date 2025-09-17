# PHASE2_MODEL_ARCHITECTURE.md - Bi-Mamba-2 + U-Net + ResCNN Implementation

## 🎯 Phase 2 Goal
Build the complete neural architecture combining U-Net spatial processing, ResCNN feature extraction, and bidirectional Mamba-2 temporal modeling for O(N) seizure detection.

## 📋 Phase 2 Checklist
- [ ] U-Net encoder (4 stages, skip connections)
- [ ] U-Net decoder (4 stages, skip fusion)
- [ ] ResCNN stack (3 blocks, multi-scale kernels)
- [ ] Bidirectional Mamba-2 bottleneck
- [ ] Full model assembly with detection head
- [ ] Forward pass validation

## 🔧 Implementation Files
```
src/experiment/models.py       # All model components
tests/test_models.py          # Architecture tests
scripts/test_forward.py       # Forward pass validation
```

## 🏗️ Architecture Overview

### Data Flow
```
Input: (B, 19, 15360) @ 256 Hz
    ↓
U-Net Encoder: 4 stages, ×16 downsample
    ↓
ResCNN: Extract multi-scale features
    ↓
Bi-Mamba-2: Bidirectional temporal modeling
    ↓
U-Net Decoder: 4 stages, ×16 upsample
    ↓
Output: (B, 15360) probabilities
```

### Dimension Tracking
| Stage | Shape | Downsample | Notes |
|-------|-------|------------|-------|
| Input | (B, 19, 15360) | ×1 | 60s @ 256 Hz |
| Enc-1 | (B, 64, 7680) | ×2 | First encoding |
| Enc-2 | (B, 128, 3840) | ×4 | Second encoding |
| Enc-3 | (B, 256, 1920) | ×8 | Third encoding |
| Enc-4 | (B, 512, 960) | ×16 | Fourth encoding |
| Bottleneck | (B, 512, 960) | ×16 | Mamba processing |
| Dec-4 | (B, 256, 1920) | ×8 | + skip from Enc-3 |
| Dec-3 | (B, 128, 3840) | ×4 | + skip from Enc-2 |
| Dec-2 | (B, 64, 7680) | ×2 | + skip from Enc-1 |
| Dec-1 | (B, 19, 15360) | ×1 | + skip from Input |
| Output | (B, 15360) | ×1 | Per-sample probs |

## 🔨 Component 1: U-Net Encoder

```python
# src/experiment/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""

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
    """U-Net encoder with progressive downsampling and skip connections."""

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

        # Initial projection
        self.input_conv = ConvBlock(in_channels, channels[0], kernel_size=7, padding=3)

        # Encoder stages
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(depth):
            in_ch = channels[0] if i == 0 else channels[i-1]
            out_ch = channels[i]

            # Double convolution block
            self.encoder_blocks.append(nn.Sequential(
                ConvBlock(in_ch if i == 0 else out_ch, out_ch),
                ConvBlock(out_ch, out_ch)
            ))

            # Downsampling (except last stage)
            if i < depth - 1:
                self.downsample.append(
                    nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)
                )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, 19, 15360)

        Returns:
            encoded: Final encoding (B, 512, 960)
            skips: List of skip connections for decoder
        """
        skips = []

        # Initial projection
        x = self.input_conv(x)  # (B, 64, 15360)

        # Encoder stages
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skips.append(x)  # Save for skip connection

            # Downsample (except last stage)
            if i < self.depth - 1:
                x = self.downsample[i](x)

        return x, skips
```

## 🔨 Component 2: ResCNN Stack

```python
# src/experiment/models.py (continued)

class ResCNNBlock(nn.Module):
    """Residual CNN block with multi-scale kernels."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-scale convolutions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels // len(kernel_sizes),
                         kernel_size=k, padding=k//2),
                nn.BatchNorm1d(channels // len(kernel_sizes)),
                nn.ReLU(inplace=True)
            )
            for k in kernel_sizes
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale processing
        branches = [branch(x) for branch in self.branches]

        # Concatenate multi-scale features
        multi_scale = torch.cat(branches, dim=1)

        # Fusion and residual connection
        out = self.fusion(multi_scale)
        return self.relu(out + x)  # Residual connection


class ResCNNStack(nn.Module):
    """Stack of ResCNN blocks for feature extraction."""

    def __init__(
        self,
        channels: int = 512,
        num_blocks: int = 3,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[
            ResCNNBlock(channels, kernel_sizes, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
```

## 🔨 Component 3: Bidirectional Mamba-2

```python
# src/experiment/models.py (continued)

# Conditional import for GPU/CPU compatibility
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, using fallback Conv1d")


class BiMamba2(nn.Module):
    """Bidirectional Mamba-2 for temporal modeling."""

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        if MAMBA_AVAILABLE:
            # Real Mamba-2 layers
            self.forward_mambas = nn.ModuleList([
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=2
                )
                for _ in range(num_layers)
            ])

            self.backward_mambas = nn.ModuleList([
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=2
                )
                for _ in range(num_layers)
            ])
        else:
            # Fallback for CPU-only testing
            print("Using Conv1d fallback for Mamba-2")
            self.forward_mambas = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
                for _ in range(num_layers)
            ])

            self.backward_mambas = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
                for _ in range(num_layers)
            ])

        # Layer norm and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, L) where C=512, L=960

        Returns:
            Bidirectional output (B, C, L)
        """
        B, C, L = x.shape

        # Transpose for Mamba processing: (B, L, C)
        x_forward = x.transpose(1, 2)
        x_backward = x.flip(dims=[2]).transpose(1, 2)

        # Process through layers
        for i in range(len(self.forward_mambas)):
            # Forward direction
            if MAMBA_AVAILABLE:
                z_forward = self.forward_mambas[i](x_forward)
            else:
                z_forward = self.forward_mambas[i](x_forward.transpose(1, 2)).transpose(1, 2)

            x_forward = self.layer_norms[i](x_forward + self.dropout(z_forward))

            # Backward direction
            if MAMBA_AVAILABLE:
                z_backward = self.backward_mambas[i](x_backward)
            else:
                z_backward = self.backward_mambas[i](x_backward.transpose(1, 2)).transpose(1, 2)

            x_backward = self.layer_norms[i](x_backward + self.dropout(z_backward))

        # Flip backward to align with forward
        x_backward = x_backward.flip(dims=[1])

        # Concatenate and project
        x_combined = torch.cat([x_forward, x_backward], dim=-1)  # (B, L, 2C)
        x_output = self.output_proj(x_combined)  # (B, L, C)

        # Transpose back: (B, C, L)
        return x_output.transpose(1, 2)
```

## 🔨 Component 4: U-Net Decoder

```python
# src/experiment/models.py (continued)

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

        # Decoder stages
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Upsampling
            self.upsample.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # Double convolution after concatenation with skip
            self.decoder_blocks.append(nn.Sequential(
                ConvBlock(out_ch + out_ch, out_ch),  # After concat with skip
                ConvBlock(out_ch, out_ch)
            ))

        # Final decoder block (no upsampling)
        self.decoder_blocks.append(nn.Sequential(
            ConvBlock(channels[-1] + base_channels, base_channels),
            ConvBlock(base_channels, base_channels)
        ))

        # Output projection
        self.output_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Encoded features (B, 512, 960)
            skips: Skip connections from encoder [4 tensors]

        Returns:
            Decoded output (B, 19, 15360)
        """
        # Process decoder stages
        for i in range(self.depth - 1):
            # Upsample
            x = self.upsample[i](x)

            # Concatenate with skip connection
            skip = skips[self.depth - 2 - i]  # Reverse order
            x = torch.cat([x, skip], dim=1)

            # Decoder block
            x = self.decoder_blocks[i](x)

        # Final stage (use first skip)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.decoder_blocks[-1](x)

        # Output projection
        return self.output_conv(x)
```

## 🔨 Component 5: Full Model Assembly

```python
# src/experiment/models.py (continued)

class SeizureDetectorV2(nn.Module):
    """Complete Bi-Mamba-2 + U-Net + ResCNN architecture."""

    def __init__(
        self,
        in_channels: int = 19,
        base_channels: int = 64,
        encoder_depth: int = 4,
        mamba_layers: int = 6,
        mamba_d_state: int = 16,
        rescnn_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # Components
        self.encoder = UNetEncoder(in_channels, base_channels, encoder_depth)
        self.rescnn = ResCNNStack(
            channels=base_channels * (2 ** (encoder_depth - 1)),  # 512
            num_blocks=rescnn_blocks,
            dropout=dropout
        )
        self.mamba = BiMamba2(
            d_model=base_channels * (2 ** (encoder_depth - 1)),  # 512
            d_state=mamba_d_state,
            num_layers=mamba_layers,
            dropout=dropout
        )
        self.decoder = UNetDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            depth=encoder_depth
        )

        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through full architecture.

        Args:
            x: Input EEG (B, 19, 15360)

        Returns:
            Seizure probabilities (B, 15360)
        """
        # Encode with skip connections
        encoded, skips = self.encoder(x)  # (B, 512, 960), [4 skips]

        # Multi-scale feature extraction
        features = self.rescnn(encoded)  # (B, 512, 960)

        # Bidirectional temporal modeling
        temporal = self.mamba(features)  # (B, 512, 960)

        # Decode with skip connections
        decoded = self.decoder(temporal, skips)  # (B, 19, 15360)

        # Detection head
        output = self.detection_head(decoded)  # (B, 1, 15360)

        return output.squeeze(1)  # (B, 15360)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> dict:
        """Get information about model layers."""
        info = {
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'rescnn_params': sum(p.numel() for p in self.rescnn.parameters()),
            'mamba_params': sum(p.numel() for p in self.mamba.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'head_params': sum(p.numel() for p in self.detection_head.parameters()),
            'total_params': self.count_parameters()
        }
        return info
```

## 🧪 Model Validation Script

```python
# scripts/test_forward.py

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.models import SeizureDetectorV2

def validate_model():
    """Validate model forward pass and dimensions."""

    print("="*50)
    print("Testing SeizureDetectorV2 Forward Pass")
    print("="*50)

    # Initialize model
    model = SeizureDetectorV2()
    model.eval()

    # Print model info
    info = model.get_layer_info()
    print(f"\nModel Parameters:")
    for key, value in info.items():
        print(f"  {key}: {value:,}")

    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 19, 15360)
    print(f"\nInput shape: {input_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output shape: {output.shape}")

    # Validate output
    assert output.shape == (batch_size, 15360), f"Expected (B, 15360), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output not in [0, 1]"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("\n✅ All validations passed!")

    # Memory usage estimate (approximate)
    param_bytes = info['total_params'] * 4  # float32
    print(f"\nApproximate model size: {param_bytes / 1024**2:.1f} MB")

    # Test with different batch sizes
    print("\nTesting different batch sizes:")
    for bs in [1, 8, 16, 32]:
        x = torch.randn(bs, 19, 15360)
        with torch.no_grad():
            y = model(x)
        print(f"  Batch {bs}: {x.shape} -> {y.shape} ✅")

if __name__ == "__main__":
    validate_model()
```

## 📊 Model Statistics

### Parameter Count (Estimated)
| Component | Parameters |
|-----------|------------|
| U-Net Encoder | ~5M |
| ResCNN Stack | ~3M |
| Bi-Mamba-2 | ~12M |
| U-Net Decoder | ~5M |
| Detection Head | ~20 |
| **Total** | **~25M** |

### Memory Requirements
- Model weights: ~100 MB (float32)
- Batch size 16: ~2 GB VRAM
- Batch size 32: ~4 GB VRAM
- Training overhead: +50%

## 🎯 Quality Metrics

### Architecture Validation
- [ ] Forward pass successful
- [ ] Output shape matches input length
- [ ] Output values in [0, 1]
- [ ] No NaN/Inf in output
- [ ] Skip connections preserved
- [ ] Gradient flow verified

### Performance Targets
- [ ] Forward pass < 100ms (GPU)
- [ ] Backward pass < 300ms (GPU)
- [ ] Memory stable across epochs
- [ ] Supports batch sizes 16-32

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **OOM on GPU** | Reduce batch size or use gradient accumulation |
| **Mamba-SSM not installed** | Falls back to Conv1d for CPU testing |
| **Dimension mismatch** | Check skip connection indices |
| **Gradient vanishing** | Add residual connections |
| **Slow convergence** | Initialize with Xavier/He initialization |

## ✅ Phase 2 Completion Criteria

1. **Code complete**: All model components implemented
2. **Tests pass**: `pytest tests/test_models.py -v`
3. **Quality check**: `make q` passes
4. **Forward validation**: `python scripts/test_forward.py` successful
5. **Documentation**: Architecture diagram updated

## 📝 Next Steps
After Phase 2 completion:
1. Move to PHASE3_TRAINING_PIPELINE.md
2. Implement loss functions and training loop
3. Integrate with data pipeline from Phase 1

---
**Status**: Ready for implementation
**Estimated Time**: 2-3 days
**Dependencies**: torch, mamba-ssm (optional for GPU)