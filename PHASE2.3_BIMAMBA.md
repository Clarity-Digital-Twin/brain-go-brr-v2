# PHASE2.3_BIMAMBA.md - Bidirectional Mamba-2 Implementation

## 🎯 Phase 2.3 Goal
Implement bidirectional Mamba-2 for O(N) temporal modeling, with CPU fallback for testing without GPU.

## 📋 Phase 2.3 Checklist
- [ ] Conditional import of mamba-ssm
- [ ] Forward Mamba-2 layers
- [ ] Backward Mamba-2 layers
- [ ] Bidirectional state fusion
- [ ] CPU fallback with Conv1d
- [ ] Unit tests for both paths

## 🔧 Implementation Files
```
src/experiment/models.py    # Bi-Mamba components live here (repo convention)
tests/test_mamba.py         # Mamba-specific tests
```
Note: `mamba-ssm` is an optional GPU extra. CPU-only dev/CI use the fallback path by default.
To enable GPU path locally: `uv sync -E gpu`.

## 📐 Architecture
```
Input: (B, 512, 960) from ResCNN
    ↓
Transpose to (B, 960, 512) for sequence processing
    ↓
Layer 1-6:
    ├── Forward Mamba-2(d_model=512, d_state=16)
    └── Backward Mamba-2(d_model=512, d_state=16)
    ↓ Concatenate → Project → LayerNorm + Residual
    ↓
Output projection: 1024→512
    ↓
Transpose back to (B, 512, 960)
```

## 🔨 Implementation

```python
# src/experiment/models.py (Mamba section)

import torch
import torch.nn as nn
from typing import Optional

# Conditional import for GPU/CPU compatibility
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, using Conv1d fallback")


class BiMamba2Layer(nn.Module):
    """Single bidirectional Mamba-2 layer."""

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        if MAMBA_AVAILABLE:
            # Real Mamba-2 for GPU
            self.forward_mamba = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.backward_mamba = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Conv1d fallback for CPU testing
            self.forward_mamba = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.backward_mamba = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # Fusion and normalization
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, L, D) where L=960, D=512

        Returns:
            Bidirectional output (B, L, D)
        """
        B, L, D = x.shape
        residual = x

        # Forward direction
        if MAMBA_AVAILABLE:
            x_forward = self.forward_mamba(x)
        else:
            # Conv1d expects (B, C, L)
            x_forward = self.forward_mamba(x.transpose(1, 2)).transpose(1, 2)

        # Backward direction (flip sequence)
        x_backward = x.flip(dims=[1])
        if MAMBA_AVAILABLE:
            x_backward = self.backward_mamba(x_backward)
        else:
            x_backward = self.backward_mamba(x_backward.transpose(1, 2)).transpose(1, 2)

        # Flip backward to align
        x_backward = x_backward.flip(dims=[1])

        # Concatenate bidirectional features
        x_combined = torch.cat([x_forward, x_backward], dim=-1)  # (B, L, 2D)

        # Project back to d_model
        x_output = self.output_proj(x_combined)  # (B, L, D)

        # Add residual and normalize
        output = self.layer_norm(residual + self.dropout(x_output))

        return output


class BiMamba2(nn.Module):
    """Stack of bidirectional Mamba-2 layers."""

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of bidirectional layers
        self.layers = nn.ModuleList([
            BiMamba2Layer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, L) where C=512, L=960

        Returns:
            Temporal output (B, C, L)
        """
        # Transpose for sequence processing: (B, L, C)
        x = x.transpose(1, 2)

        # Process through bidirectional layers
        for layer in self.layers:
            x = layer(x)

        # Transpose back: (B, C, L)
        return x.transpose(1, 2)

    def get_complexity(self) -> str:
        """Return complexity analysis."""
        if MAMBA_AVAILABLE:
            return "O(N) with Mamba-2 SSM"
        else:
            return "O(N) with Conv1d fallback"
```

## 🧪 Test Suite

```python
# tests/test_mamba.py

import pytest
import torch
import torch.nn as nn
from src.experiment.models import BiMamba2Layer, BiMamba2, MAMBA_AVAILABLE


class TestBiMamba2Layer:

    @pytest.fixture
    def layer(self):
        return BiMamba2Layer(d_model=512, d_state=16)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 960, 512)  # (B, L, D)

    def test_output_shape(self, layer, sample_input):
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_bidirectional_processing(self, layer):
        # Create asymmetric input
        x = torch.zeros(1, 960, 512)
        x[:, :100, :] = 1.0  # Signal at start

        output = layer(x)

        # Output should differ from input due to bidirectional processing
        assert not torch.allclose(output, x)

        # Check that information propagates
        # (backward pass should spread early signal)
        later_signal = output[:, -100:, :].abs().mean()
        assert later_signal > 0

    def test_residual_connection(self, layer, sample_input):
        output = layer(sample_input)

        # Should preserve some input information via residual
        correlation = torch.cosine_similarity(
            sample_input.flatten(),
            output.flatten(),
            dim=0
        )
        assert correlation > 0.5  # Some preservation


class TestBiMamba2:

    @pytest.fixture
    def model(self):
        return BiMamba2(d_model=512, num_layers=6)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 512, 960)  # (B, C, L)

    def test_output_shape(self, model, sample_input):
        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_gradient_flow(self, model, sample_input):
        sample_input.requires_grad = True
        output = model(sample_input)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_complexity(self, model):
        complexity = model.get_complexity()
        assert "O(N)" in complexity

        if MAMBA_AVAILABLE:
            assert "Mamba-2" in complexity
        else:
            assert "Conv1d" in complexity

    def test_no_nan_inf(self, model, sample_input):
        output = model(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    def test_mamba_path(self, model):
        # Verify Mamba-2 path is active when available
        assert not isinstance(model.layers[0].forward_mamba, nn.Conv1d)

    @pytest.mark.skipif(MAMBA_AVAILABLE, reason="Testing fallback only")
    def test_fallback_path(self, model):
        # Verify Conv1d fallback works
        assert isinstance(model.layers[0].forward_mamba, nn.Conv1d)
```

## ✅ Phase 2.3 Completion Criteria
1. Bidirectional Mamba-2 implementation
2. CPU fallback working for dev testing
3. Tests pass: `pytest tests/test_mamba.py -v`
4. `make q` passes
5. O(N) complexity verified

---
**Status**: Ready for implementation
**Estimated Time**: 1 day
**Depends on**: PHASE2.2 completion
**Next**: PHASE2.4_DECODER.md
