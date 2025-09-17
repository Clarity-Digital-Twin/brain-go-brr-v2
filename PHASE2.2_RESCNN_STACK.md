# PHASE2.2_RESCNN_STACK.md - Residual CNN Stack Implementation

## 🎯 Phase 2.2 Goal
Build multi-scale ResCNN blocks for feature extraction at the bottleneck, processing the encoded representation with kernels [3, 5, 7].

## 📋 Phase 2.2 Checklist
- [ ] ResCNNBlock with multi-scale branches
- [ ] Residual connections for gradient flow
- [ ] ResCNNStack with 3 blocks
- [ ] Spatial dropout for regularization
- [ ] Unit tests for shape preservation
- [ ] Verify residual connections work

## 🔧 Implementation Files
```
src/experiment/models/rescnn.py    # ResCNN components only
tests/test_rescnn.py               # ResCNN-specific tests
```

## 📐 Architecture
```
Input: (B, 512, 960) from encoder
    ↓
ResCNNBlock 1: Multi-scale k=[3,5,7]
    ├── Branch 1: Conv1d(k=3) → 170 channels
    ├── Branch 2: Conv1d(k=5) → 170 channels
    └── Branch 3: Conv1d(k=7) → 172 channels
    ↓ Concatenate → Fusion → Add residual
ResCNNBlock 2: Same structure
    ↓
ResCNNBlock 3: Same structure
    ↓
Output: (B, 512, 960) - shape preserved!
```

## 🔨 Implementation

```python
# src/experiment/models/rescnn.py

import torch
import torch.nn as nn
from typing import List


class ResCNNBlock(nn.Module):
    """Residual CNN block with multi-scale kernels."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolution branches
        # Split channels evenly across branches
        branch_channels = channels // len(kernel_sizes)
        remainder = channels % len(kernel_sizes)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Add remainder channels to last branch
            out_ch = branch_channels + (remainder if i == len(kernel_sizes) - 1 else 0)

            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(channels, out_ch, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, L) where C=512, L=960

        Returns:
            Output with residual connection (B, C, L)
        """
        # Process through multi-scale branches
        branches_out = []
        for branch in self.branches:
            branches_out.append(branch(x))

        # Concatenate multi-scale features
        multi_scale = torch.cat(branches_out, dim=1)

        # Fusion and residual connection
        fused = self.fusion(multi_scale)
        output = self.relu(fused + x)  # Residual connection

        return output


class ResCNNStack(nn.Module):
    """Stack of ResCNN blocks for deep feature extraction."""

    def __init__(
        self,
        channels: int = 512,
        num_blocks: int = 3,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks

        # Stack of ResCNN blocks
        self.blocks = nn.ModuleList([
            ResCNNBlock(channels, kernel_sizes, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features (B, 512, 960)

        Returns:
            Enhanced features (B, 512, 960)
        """
        for block in self.blocks:
            x = block(x)

        return x

    def get_receptive_field(self) -> int:
        """Calculate total receptive field."""
        # Each block has max kernel size 7
        # With 3 blocks: 7 + 6 + 6 = 19 samples
        max_kernel = max(self.blocks[0].kernel_sizes)
        return max_kernel + (self.num_blocks - 1) * (max_kernel - 1)
```

## 🧪 Test Suite

```python
# tests/test_rescnn.py

import pytest
import torch
from src.experiment.models.rescnn import ResCNNBlock, ResCNNStack


class TestResCNNBlock:

    @pytest.fixture
    def block(self):
        return ResCNNBlock(channels=512, kernel_sizes=[3, 5, 7], dropout=0.1)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 512, 960)  # Batch of 4

    def test_shape_preservation(self, block, sample_input):
        output = block(sample_input)
        assert output.shape == sample_input.shape

    def test_residual_connection(self, block):
        # Test with zeros - should return zeros due to residual
        x = torch.zeros(2, 512, 960)
        output = block(x)
        # Output shouldn't be all zeros if processing works
        assert output.abs().sum() > 0

    def test_multi_scale_branches(self, block):
        # Check that all branches are used
        x = torch.randn(1, 512, 960)

        # Hook to capture branch outputs
        branch_outputs = []
        def hook(module, input, output):
            branch_outputs.append(output)

        for branch in block.branches:
            branch.register_forward_hook(hook)

        _ = block(x)
        assert len(branch_outputs) == 3  # 3 kernel sizes

        # Check branch dimensions sum to 512
        total_channels = sum(out.shape[1] for out in branch_outputs)
        assert total_channels == 512


class TestResCNNStack:

    @pytest.fixture
    def stack(self):
        return ResCNNStack(channels=512, num_blocks=3)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 512, 960)

    def test_output_shape(self, stack, sample_input):
        output = stack(sample_input)
        assert output.shape == sample_input.shape

    def test_gradient_flow(self, stack, sample_input):
        sample_input.requires_grad = True
        output = stack(sample_input)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

        # Check gradient magnitude (residuals should help)
        grad_norm = sample_input.grad.norm()
        assert grad_norm > 0
        assert grad_norm < 1000  # Not exploding

    def test_receptive_field(self, stack):
        rf = stack.get_receptive_field()
        assert rf == 19  # 7 + 6 + 6

    def test_no_nan_inf(self, stack, sample_input):
        output = stack(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
```

## ✅ Phase 2.2 Completion Criteria
1. ResCNN implementation with multi-scale kernels
2. Tests pass: `pytest tests/test_rescnn.py -v`
3. `make q` passes
4. Residual connections verified
5. Shape preservation confirmed

---
**Status**: Ready for implementation
**Estimated Time**: 0.5 day
**Depends on**: PHASE2.1 completion
**Next**: PHASE2.3_BIMAMBA.md