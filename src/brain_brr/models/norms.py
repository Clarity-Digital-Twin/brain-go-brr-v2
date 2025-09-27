"""Normalization modules for V3 architectural stability.

Part of PR-1: Boundary Normalization fix for unbounded information flow.
These modules provide normalization at component boundaries to prevent
activation explosion and gradient instability.
"""

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm with similar stability benefits.
    Used in Mamba reference implementation.

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor with shape (..., dim)

        Returns:
            Normalized tensor with same shape
        """
        # Compute RMS along last dimension
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        x = x / (rms + self.eps)
        return self.scale * x


class LayerScale(nn.Module):
    """Learnable scaling of residual branches.

    From "Going Deeper with Image Transformers" (Touvron et al., 2021).
    Prevents feature collapse in deep residual networks by starting
    residuals with small weights.

    Args:
        dim: Dimension of the features
        init_value: Initial scaling factor (default 0.1 from paper)
    """

    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input by learnable factor.

        Args:
            x: Input tensor with shape (..., dim)

        Returns:
            Scaled tensor with same shape
        """
        return self.gamma * x


def create_norm_layer(
    norm_type: str,
    dim: int,
    eps: float = 1e-5
) -> nn.Module | None:
    """Factory function to create normalization layers.

    Args:
        norm_type: Type of normalization ("layernorm", "rmsnorm", "none")
        dim: Dimension to normalize
        eps: Epsilon for numerical stability

    Returns:
        Normalization module or None if norm_type is "none"
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "none" or not norm_type:
        return None
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


__all__ = ["LayerScale", "RMSNorm", "create_norm_layer"]