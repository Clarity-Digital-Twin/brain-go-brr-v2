"""PR-3: Adjacency matrix conditioning for numerical stability.

This module implements well-conditioned adjacency matrix operations
to prevent eigendecomposition failures in dynamic Laplacian PE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


def condition_adjacency(
    adjacency: torch.Tensor,
    top_k: int = 3,
    tau: float = 1.0,
    force_symmetric: bool = False,
    row_softmax: bool = False,
    ema_beta: float | None = None,
    prev_adjacency: torch.Tensor | None = None,
) -> torch.Tensor:
    """Condition adjacency matrix for stable eigendecomposition.

    Args:
        adjacency: Input adjacency matrix (B, T, N, N)
        top_k: Number of neighbors to keep per node
        tau: Temperature for softmax normalization
        force_symmetric: Whether to force symmetry
        row_softmax: Whether to apply masked row-wise softmax
        ema_beta: EMA coefficient for temporal smoothing (None=disabled)
        prev_adjacency: Previous adjacency for EMA (None for first call)

    Returns:
        Conditioned adjacency matrix (B, T, N, N)
    """
    B, T, N, _ = adjacency.shape  # noqa: N806

    # Apply top-k sparsification first (already done in detector.py)
    # We receive the adjacency after top-k, so we work with that

    if row_softmax:
        # Zero diagonal to avoid self-loops dominating
        eye = torch.eye(N, device=adjacency.device, dtype=adjacency.dtype)
        eye = eye.view(1, 1, N, N).expand(B, T, -1, -1)
        adjacency = adjacency * (1.0 - eye)

        # Masked row-wise softmax
        # Find which entries are non-zero (after top-k thresholding)
        mask = adjacency != 0

        # Apply softmax only to non-zero entries
        adjacency_for_softmax = adjacency / tau
        # Mask out zeros with large negative value
        adjacency_for_softmax = adjacency_for_softmax.masked_fill(~mask, -1e9)
        adjacency = func.softmax(adjacency_for_softmax, dim=-1)

        # Restore zeros where mask is False
        adjacency = adjacency * mask.float()

    # Within-sequence temporal EMA smoothing
    if ema_beta is not None and T > 1:
        smoothed = torch.empty_like(adjacency)
        smoothed[:, 0] = adjacency[:, 0]
        for t in range(1, T):
            smoothed[:, t] = ema_beta * smoothed[:, t - 1] + (1 - ema_beta) * adjacency[:, t]
        adjacency = smoothed

    # Force symmetry for valid Laplacian
    if force_symmetric:
        adjacency = (adjacency + adjacency.transpose(-2, -1)) / 2

    # Add identity fallback for disconnected nodes
    # Check for rows with very small sum (disconnected nodes)
    row_sums = adjacency.sum(dim=-1, keepdim=True)  # (B, T, N, 1)
    disconnected = row_sums < 1e-6  # (B, T, N, 1)

    if disconnected.any():
        # Add small self-loop to disconnected nodes
        eye = torch.eye(N, device=adjacency.device, dtype=adjacency.dtype)
        eye = eye.view(1, 1, N, N).expand(B, T, -1, -1)
        # Only add to diagonal where nodes are disconnected
        disconnected_diag = disconnected.squeeze(-1)  # (B, T, N)
        identity_add = torch.diag_embed(disconnected_diag.float() * 0.1)  # Small self-loop
        adjacency = adjacency + identity_add

    return adjacency


def compute_stable_laplacian(
    adjacency: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute numerically stable graph Laplacian.

    Args:
        adjacency: Adjacency matrix (B*T, N, N) or (B, T, N, N)
        normalize: Whether to use normalized Laplacian
        eps: Regularization epsilon

    Returns:
        Laplacian matrix with same shape as input
    """
    # Handle both (B*T, N, N) and (B, T, N, N) shapes
    original_shape = adjacency.shape
    if len(original_shape) == 4:
        B, T, N, _ = original_shape  # noqa: N806
        adjacency = adjacency.reshape(B * T, N, N)
    else:
        N = adjacency.shape[-1]  # noqa: N806

    device = adjacency.device
    dtype = adjacency.dtype

    if normalize:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        degrees = adjacency.sum(dim=-1).clamp_min(eps)  # Avoid division by zero
        d_inv_sqrt = torch.diag_embed(degrees.rsqrt())
        a_norm = d_inv_sqrt @ adjacency @ d_inv_sqrt

        # Identity matrix
        identity = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        identity = identity.expand_as(a_norm)

        laplacian = identity - a_norm
    else:
        # Unnormalized Laplacian: L = D - A
        degrees = torch.diag_embed(adjacency.sum(dim=-1))
        laplacian = degrees - adjacency

    # Add regularization for numerical stability
    identity = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    identity = identity.expand_as(laplacian)
    laplacian = laplacian + eps * identity

    # Restore original shape
    if len(original_shape) == 4:
        laplacian = laplacian.reshape(B, T, N, N)

    return laplacian


class TemporalEMA(nn.Module):
    """Exponential moving average for temporal smoothing of adjacency matrices.

    This version performs within-sequence EMA by default (no persistent state).
    Cross-forward EMA can be enabled with persistent=True.
    """

    def __init__(self, beta: float = 0.9, persistent: bool = False):
        super().__init__()
        self.beta = beta
        self.persistent = persistent
        if persistent:
            self.register_buffer("prev_adjacency", None)
            self.prev_adjacency: torch.Tensor | None = None

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply temporal EMA smoothing.

        Args:
            adjacency: Input adjacency (B, T, N, N)

        Returns:
            Smoothed adjacency (B, T, N, N)
        """
        _, T, _, _ = adjacency.shape  # noqa: N806

        if T == 1 and not self.persistent:
            # No temporal dimension to smooth
            return adjacency

        # Within-sequence causal EMA (default)
        smoothed = torch.empty_like(adjacency)

        if self.persistent and hasattr(self, "prev_adjacency") and self.prev_adjacency is not None:
            # Use stored state for first timestep
            smoothed[:, 0] = self.beta * self.prev_adjacency + (1 - self.beta) * adjacency[:, 0]
        else:
            # No history, use first timestep as-is
            smoothed[:, 0] = adjacency[:, 0]

        # Apply EMA across time dimension
        for t in range(1, T):
            smoothed[:, t] = self.beta * smoothed[:, t - 1] + (1 - self.beta) * adjacency[:, t]

        # Store last timestep for next forward if persistent
        if self.persistent:
            self.prev_adjacency = smoothed[:, -1].detach()

        return smoothed
