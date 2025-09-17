# PHASE2.5_FULL_MODEL.md - Complete Model Assembly

## 🎯 Phase 2.5 Goal
Assemble all components (Encoder, ResCNN, Bi-Mamba, Decoder) into the complete SeizureDetectorV2 model with detection head.

## 📋 Phase 2.5 Checklist
- [ ] Import all component modules
- [ ] Wire components in correct order
- [ ] Add sigmoid detection head
- [ ] Parameter counting utilities
- [ ] Full forward pass test
- [ ] End-to-end gradient flow test

## 🔧 Implementation Files
```
src/experiment/models.py       # Full assembly (same module as components)
tests/test_full_model.py       # Integration tests
scripts/test_model.py          # Manual validation
```

## 📐 Complete Architecture Flow
```
Input: (B, 19, 15360) @ 256 Hz
    ↓
U-Net Encoder → (B, 512, 960) + 4 skips
    ↓
ResCNN Stack → (B, 512, 960)
    ↓
Bi-Mamba-2 → (B, 512, 960)
    ↓
U-Net Decoder + skips → (B, 19, 15360)
    ↓
Detection Head (Conv1d + Sigmoid) → (B, 15360)
    ↓
Output: Per-sample seizure probabilities
```

## 🔨 Implementation

```python
# src/experiment/models.py (Full model section)

import torch
import torch.nn as nn
from typing import Dict, Any, List

# Assumes UNetEncoder, ResCNNStack, BiMamba2, UNetDecoder are defined above in this file


class SeizureDetectorV2(nn.Module):
    """Complete Bi-Mamba-2 + U-Net + ResCNN architecture for seizure detection."""

    def __init__(
        self,
        # Input/Output
        in_channels: int = 19,
        # U-Net params
        base_channels: int = 64,
        encoder_depth: int = 4,
        # Mamba params
        mamba_layers: int = 6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 5,
        # ResCNN params
        rescnn_blocks: int = 3,
        rescnn_kernels: List[int] = None,
        # Regularization
        dropout: float = 0.1,
    ):
        super().__init__()

        # Default kernel sizes if not provided
        if rescnn_kernels is None:
            rescnn_kernels = [3, 5, 7]

        # Save config for reproducibility
        self.config = {
            'in_channels': in_channels,
            'base_channels': base_channels,
            'encoder_depth': encoder_depth,
            'mamba_layers': mamba_layers,
            'mamba_d_state': mamba_d_state,
            'mamba_d_conv': mamba_d_conv,
            'rescnn_blocks': rescnn_blocks,
            'rescnn_kernels': rescnn_kernels,
            'dropout': dropout,
        }

        # Calculate bottleneck channels
        bottleneck_channels = base_channels * (2 ** (encoder_depth - 1))  # 512

        # Build components
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=encoder_depth
        )

        self.rescnn = ResCNNStack(
            channels=bottleneck_channels,
            num_blocks=rescnn_blocks,
            kernel_sizes=rescnn_kernels,
            dropout=dropout
        )

        self.mamba = BiMamba2(
            d_model=bottleneck_channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            num_layers=mamba_layers,
            dropout=dropout
        )

        self.decoder = UNetDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            depth=encoder_depth
        )

        # Detection head: project to single channel + sigmoid
        self.detection_head = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete architecture.

        Args:
            x: Input EEG windows (B, 19, 15360)

        Returns:
            Seizure probabilities (B, 15360) in [0, 1]
        """
        # Encode with skip connections
        encoded, skips = self.encoder(x)
        # encoded: (B, 512, 960), skips: 4 tensors

        # Multi-scale feature extraction
        features = self.rescnn(encoded)
        # features: (B, 512, 960)

        # Bidirectional temporal modeling
        temporal = self.mamba(features)
        # temporal: (B, 512, 960)

        # Decode with skip connections
        decoded = self.decoder(temporal, skips)
        # decoded: (B, 19, 15360)

        # Detection head
        output = self.detection_head(decoded)
        # output: (B, 1, 15360)

        # Squeeze channel dimension
        return output.squeeze(1)  # (B, 15360)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> Dict[str, Any]:
        """Get detailed information about model layers."""
        info = {
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'rescnn_params': sum(p.numel() for p in self.rescnn.parameters()),
            'mamba_params': sum(p.numel() for p in self.mamba.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'head_params': sum(p.numel() for p in self.detection_head.parameters()),
            'total_params': self.count_parameters(),
            'config': self.config
        }
        return info

    def get_memory_usage(self, batch_size: int = 16) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        # Model parameters (float32)
        param_bytes = self.count_parameters() * 4

        # Rough activation memory estimate
        # Largest activation is at input: B × 19 × 15360
        activation_bytes = batch_size * 19 * 15360 * 4

        # Include intermediate activations (rough 3x multiplier)
        total_activation_bytes = activation_bytes * 3

        return {
            'model_size_mb': param_bytes / (1024**2),
            'activation_size_mb': total_activation_bytes / (1024**2),
            'total_size_mb': (param_bytes + total_activation_bytes) / (1024**2)
        }
```

## 🧪 Test Suite

```python
# tests/test_full_model.py

import pytest
import torch
from src.experiment.models import SeizureDetectorV2


class TestSeizureDetectorV2:

    @pytest.fixture
    def model(self):
        return SeizureDetectorV2(
            in_channels=19,
            base_channels=64,
            encoder_depth=4,
            mamba_layers=6,
            mamba_d_state=16,
            rescnn_blocks=3
        )

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 19, 15360)  # Batch of 4

    def test_forward_shape(self, model, sample_input):
        output = model(sample_input)
        assert output.shape == (4, 15360)

    def test_output_range(self, model, sample_input):
        output = model(sample_input)

        # Sigmoid ensures [0, 1] range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_gradient_flow(self, model, sample_input):
        sample_input.requires_grad = True
        output = model(sample_input)

        # Simulate binary cross-entropy loss
        target = torch.rand_like(output)
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()

    def test_no_nan_inf(self, model, sample_input):
        output = model(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_parameter_count(self, model):
        info = model.get_layer_info()

        # Verify components add up
        component_sum = (
            info['encoder_params'] +
            info['rescnn_params'] +
            info['mamba_params'] +
            info['decoder_params'] +
            info['head_params']
        )
        assert component_sum == info['total_params']

        # Check reasonable size (10-50M params expected)
        assert 10_000_000 < info['total_params'] < 50_000_000

    def test_memory_usage(self, model):
        mem_info = model.get_memory_usage(batch_size=16)

        # Model should be < 200 MB
        assert mem_info['model_size_mb'] < 200

        # Total with batch 16 should be < 4 GB
        assert mem_info['total_size_mb'] < 4000

    def test_different_batch_sizes(self, model):
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 19, 15360)
            output = model(x)
            assert output.shape == (batch_size, 15360)

    def test_deterministic(self, model):
        torch.manual_seed(42)
        x = torch.randn(2, 19, 15360)

        # Two forward passes should give same result
        out1 = model(x)
        out2 = model(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_config_storage(self, model):
        info = model.get_layer_info()
        config = info['config']

        assert config['in_channels'] == 19
        assert config['base_channels'] == 64
        assert config['encoder_depth'] == 4
        assert config['mamba_layers'] == 6
        assert config['mamba_d_state'] == 16
```

## 🚀 Validation Script

```python
# scripts/test_model.py

#!/usr/bin/env python
"""Test complete model forward pass and statistics."""

import sys
from pathlib import Path
import torch
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.models import SeizureDetectorV2


def main():
    print("="*60)
    print("SeizureDetectorV2 Full Model Test")
    print("="*60)

    # Initialize model
    model = SeizureDetectorV2()
    model.eval()

    # Model info
    info = model.get_layer_info()
    print("\n📊 Model Statistics:")
    print(f"  Encoder params:  {info['encoder_params']:,}")
    print(f"  ResCNN params:   {info['rescnn_params']:,}")
    print(f"  Mamba params:    {info['mamba_params']:,}")
    print(f"  Decoder params:  {info['decoder_params']:,}")
    print(f"  Head params:     {info['head_params']:,}")
    print(f"  Total params:    {info['total_params']:,}")

    # Memory estimates
    for bs in [1, 8, 16, 32]:
        mem = model.get_memory_usage(bs)
        print(f"\n💾 Batch size {bs}:")
        print(f"  Model:      {mem['model_size_mb']:.1f} MB")
        print(f"  Activation: {mem['activation_size_mb']:.1f} MB")
        print(f"  Total:      {mem['total_size_mb']:.1f} MB")

    # Test forward pass
    print("\n🔄 Testing forward pass...")
    test_batch_sizes = [1, 4, 8, 16]

    for bs in test_batch_sizes:
        x = torch.randn(bs, 19, 15360)

        # Time forward pass
        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Batch {bs:2d}: {x.shape} → {y.shape}  ({elapsed:.1f} ms)")

        # Validate output
        assert y.shape == (bs, 15360)
        assert torch.all(y >= 0) and torch.all(y <= 1)
        assert not torch.isnan(y).any()

    print("\n✅ All tests passed!")

    # GPU test if available
    if torch.cuda.is_available():
        print("\n🎮 Testing on GPU...")
        model = model.cuda()
        x = torch.randn(16, 19, 15360).cuda()

        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  GPU forward pass (batch 16): {elapsed:.1f} ms")
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
```

## ✅ Phase 2.5 Completion Criteria
1. Full model assembled with all components
2. Forward pass produces correct shapes
3. Output values in [0, 1] range
4. Tests pass: `pytest tests/test_full_model.py -v`
5. `make q` passes
6. `python scripts/test_model.py` successful

## 📊 Expected Performance
- **Parameters**: ~25M (estimated)
- **Model Size**: ~100 MB (float32)
- **Inference Speed** (GPU):
  - Batch 1: < 20ms
  - Batch 16: < 100ms
- **Memory Usage** (GPU):
  - Batch 16: ~2 GB
  - Batch 32: ~4 GB

---
**Status**: Ready for implementation
**Estimated Time**: 0.5 day
**Depends on**: PHASE2.1-2.4 all complete
**Next**: PHASE3_TRAINING_PIPELINE.md
