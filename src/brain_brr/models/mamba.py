"""Bidirectional Mamba-2 components for O(N) sequence modeling.

Slow (10-600s) temporal scale: Full seizure dynamics via Bi-Mamba-2
Context (±5min): Pre/post-ictal patterns via bidirectional SSM
O(N) complexity avoids transformer's O(N²) cost on long EEG sequences
"""

import os
import warnings
from typing import cast

import torch
import torch.nn as nn

# Conditional import for GPU/CPU compatibility
try:
    from mamba_ssm import Mamba2

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not available; using Conv1d fallback for BiMamba2. "
        "Fallback is for shape validation only and is not functionally equivalent.",
        stacklevel=1,
    )


class BiMamba2Layer(nn.Module):
    """Single bidirectional Mamba-2 layer.

    Args:
        d_model: Feature dimension (matches encoder bottleneck)
        d_state: SSM state dimension
        d_conv: Conv kernel size (default 5 to match schemas/configs)
        expand: Expansion factor in Mamba component
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv  # Public/configured kernel (docs/schemas default=5)

        # Optional override to force fallback even if CUDA/Mamba are available
        self._force_fallback = os.getenv("SEIZURE_MAMBA_FORCE_FALLBACK", "0") == "1"

        # Mamba CUDA kernel supports width in {2, 3, 4}; coerce if needed (GPU path only)
        _allowed = (2, 3, 4)
        self._mamba_conv_k = d_conv if d_conv in _allowed else 4
        if MAMBA_AVAILABLE and d_conv not in _allowed:
            warnings.warn(
                f"Mamba CUDA path coerced conv kernel from {d_conv}→{self._mamba_conv_k} "
                "(CUDA op supports only 2-4). CPU fallback still uses configured kernel.",
                stacklevel=1,
            )

        # Always create both paths - decide at runtime
        if MAMBA_AVAILABLE:
            # Real Mamba-2 for GPU
            self.forward_mamba_real = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=self._mamba_conv_k, expand=expand
            )
            self.backward_mamba_real = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=self._mamba_conv_k, expand=expand
            )
        else:
            self.forward_mamba_real = None
            self.backward_mamba_real = None

        # Fallback for CPU/testing (Conv1d to match docs/tests; operates on (B, C, L))
        # WARNING: This is NOT functionally equivalent to Mamba-2 SSM!
        padding = max(0, self.d_conv // 2)
        self.forward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding
        )
        self.backward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding
        )

        # Fusion and normalization
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def forward_mamba(self) -> nn.Module:
        """Compatibility property for tests."""
        return (
            self.forward_mamba_real
            if self.forward_mamba_real is not None
            else self.forward_mamba_fallback
        )

    @property
    def backward_mamba(self) -> nn.Module:
        """Compatibility property for tests."""
        return (
            self.backward_mamba_real
            if self.backward_mamba_real is not None
            else self.backward_mamba_fallback
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input bidirectionally.

        Args:
            x: Input (B, L, D) where L=960, D=512

        Returns:
            Bidirectional output (B, L, D)
        """
        residual = x

        # Use real Mamba only if:
        # 1. Library is available
        # 2. CUDA is available
        # 3. Input is on CUDA
        # 4. Kernel width is supported (2-4 for causal_conv1d)
        use_mamba = (
            MAMBA_AVAILABLE
            and x.is_cuda  # check tensor first to avoid noisy CUDA init on CPU
            and torch.cuda.is_available()
            and self._mamba_conv_k in (2, 3, 4)  # causal_conv1d constraint
            and not self._force_fallback
        )

        # Forward direction
        try:
            x_forward = (
                self.forward_mamba_real(x)
                if use_mamba
                else self.forward_mamba_fallback(x.transpose(1, 2)).transpose(1, 2)
            )
        except (AttributeError, RuntimeError) as e:
            # Mamba CUDA kernel not available, fall back to Conv1d
            if "causal_conv1d" in str(e) or "NoneType" in str(e):
                x_forward = self.forward_mamba_fallback(x.transpose(1, 2)).transpose(1, 2)
            else:
                raise

        # Backward direction (flip sequence)
        x_backward = x.flip(dims=[1])
        try:
            if use_mamba:
                x_backward = self.backward_mamba_real(x_backward)
            else:
                # Conv1d fallback with transpose
                x_backward = self.backward_mamba_fallback(x_backward.transpose(1, 2)).transpose(
                    1, 2
                )
        except (AttributeError, RuntimeError) as e:
            # Mamba CUDA kernel not available, fall back to Conv1d
            if "causal_conv1d" in str(e) or "NoneType" in str(e):
                x_backward = self.backward_mamba_fallback(x_backward.transpose(1, 2)).transpose(
                    1, 2
                )
            else:
                raise

        # Flip backward to align
        x_backward = x_backward.flip(dims=[1])

        # Concatenate bidirectional features
        x_combined = torch.cat([x_forward, x_backward], dim=-1)  # (B, L, 2D)

        # Project back to d_model
        x_output = self.output_proj(x_combined)  # (B, L, D)

        # Add residual and normalize
        output = self.layer_norm(residual + self.dropout(x_output))

        return cast(torch.Tensor, output)


class BiMamba2(nn.Module):
    """Stack of bidirectional Mamba-2 layers for O(N) temporal modeling.

    Processes sequences with linear complexity, avoiding the O(N²) cost
    of transformers on long EEG sequences.

    Args:
        d_model: Model dimension (512 for encoder bottleneck)
        d_state: SSM state dimension (16 default)
        d_conv: Temporal conv kernel (5 default)
        num_layers: Number of bidirectional layers (6 default)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of bidirectional layers
        self.layers = nn.ModuleList(
            [
                BiMamba2Layer(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through bidirectional Mamba-2 stack.

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


__all__ = ["MAMBA_AVAILABLE", "BiMamba2", "BiMamba2Layer"]