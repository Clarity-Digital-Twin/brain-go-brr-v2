"""Bidirectional Mamba-2 components for O(N) sequence modeling.

Slow (10-600s) temporal scale: Full seizure dynamics via Bi-Mamba-2
Context (±5min): Pre/post-ictal patterns via bidirectional SSM
O(N) complexity avoids transformer's O(N²) cost on long EEG sequences
"""

import warnings
from typing import cast

import torch
import torch.nn as nn
from src.brain_brr.utils.env import env

# Conditional import for GPU/CPU compatibility
# No longer needed - we use d_conv=4 everywhere now
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
        d_conv: Conv kernel size (default 4; CUDA supports 2-4)
        expand: Expansion factor in Mamba component
        headdim: Head dimension for Mamba2 (must satisfy (d_model * expand) / headdim is multiple of 8)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv  # Now always 4 or less, no coercion needed
        self.expand = expand
        self.headdim = headdim

        # Validate headdim requirement for CUDA kernels
        ratio = (d_model * expand) / headdim
        if ratio != int(ratio):
            raise ValueError(
                f"Invalid headdim: (d_model * expand) / headdim = ({d_model} * {expand}) / {headdim} = {ratio} must be an integer"
            )
        if int(ratio) % 8 != 0:
            warnings.warn(
                f"(d_model * expand) / headdim = {int(ratio)} should be multiple of 8 for best CUDA performance",
                stacklevel=2,
            )

        # Optional override to force fallback even if CUDA/Mamba are available
        self._force_fallback = env.force_mamba_fallback()

        # Always create both paths - decide at runtime
        if MAMBA_AVAILABLE:
            # Real Mamba-2 for GPU
            try:
                self.forward_mamba_real = Mamba2(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim
                )
                self.backward_mamba_real = Mamba2(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim
                )
                print("[MAMBA] Successfully created Mamba2 layers", flush=True)
            except Exception as e:
                print(f"[MAMBA] Failed to create Mamba2 layers: {e}", flush=True)
                self.forward_mamba_real = None
                self.backward_mamba_real = None
        else:
            print("[MAMBA] Mamba-SSM not available, using fallback", flush=True)
            self.forward_mamba_real = None
            self.backward_mamba_real = None

        # Fallback for CPU/testing (Conv1d to match docs/tests; operates on (B, C, L))
        # WARNING: This is NOT functionally equivalent to Mamba-2 SSM!
        # Use 'same' padding to preserve sequence length
        padding = "same"
        # Depthwise conv fallback for CPU speed (preserves channel count)
        self.forward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding, groups=d_model
        )
        self.backward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding, groups=d_model
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
            and self.d_conv in (2, 3, 4)  # CUDA causal_conv1d constraint
            and not self._force_fallback
        )

        # Forward direction
        try:
            if use_mamba and self.forward_mamba_real is not None:
                x_forward = self.forward_mamba_real(x)
            else:
                x_forward = self.forward_mamba_fallback(x.transpose(1, 2)).transpose(1, 2)
        except (AttributeError, RuntimeError, TypeError) as e:
            # Mamba CUDA kernel not available, fall back to Conv1d
            print(f"[MAMBA] Forward pass error, using fallback: {e}", flush=True)
            if (
                "causal_conv1d" in str(e)
                or "NoneType" in str(e)
                or "object is not callable" in str(e)
            ):
                x_forward = self.forward_mamba_fallback(x.transpose(1, 2)).transpose(1, 2)
            else:
                raise

        # Backward direction (flip sequence)
        x_backward = x.flip(dims=[1]).contiguous()  # Ensure contiguous after flip
        try:
            if use_mamba and self.backward_mamba_real is not None:
                x_backward = self.backward_mamba_real(x_backward)
            else:
                # Conv1d fallback with transpose
                x_backward = self.backward_mamba_fallback(x_backward.transpose(1, 2)).transpose(
                    1, 2
                )
        except (AttributeError, RuntimeError, TypeError) as e:
            # Mamba CUDA kernel not available, fall back to Conv1d
            print(f"[MAMBA] Backward pass error, using fallback: {e}", flush=True)
            if (
                "causal_conv1d" in str(e)
                or "NoneType" in str(e)
                or "object is not callable" in str(e)
            ):
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
        d_conv: Temporal conv kernel (4 default)
        expand: Expansion factor (2 default)
        headdim: Head dimension (must satisfy (d_model * expand) / headdim is multiple of 8)
        num_layers: Number of bidirectional layers (6 default)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.expand = expand
        self.headdim = headdim

        # Stack of bidirectional layers
        self.layers = nn.ModuleList(
            [
                BiMamba2Layer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                )
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
        x = x.transpose(1, 2).contiguous()  # Ensure contiguous for CUDA kernels

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
