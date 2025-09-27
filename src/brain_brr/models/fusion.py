"""PR-4: Gated fusion modules for node/edge stream combination.

This module implements learnable gating mechanisms to replace simple
additive fusion, preventing edge noise from dominating node features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Learnable gating for node/edge stream fusion.

    Instead of simple addition (node + edge), uses a learned gate
    to weight the edge contribution based on both features.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Apply gated fusion.

        Args:
            node: Node features (B, N, T, D)
            edge: Edge features (B, N, T, D)

        Returns:
            Fused features (B, N, T, D)
        """
        # Compute gate from concatenated features
        combined = torch.cat([node, edge], dim=-1)
        gate = self.sigmoid(self.gate_proj(combined))
        gate = self.dropout(gate)

        # Gated combination: node is primary, edge is gated
        output = node + gate * edge
        return output


class MultiHeadGatedFusion(nn.Module):
    """Multi-head attention-style gating for richer interactions.

    Uses multi-head attention between node and edge streams
    to learn complex interaction patterns.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate projections for query (node), key/value (edge)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Apply multi-head gated fusion.

        Args:
            node: Node features (B, N, T, D)
            edge: Edge features (B, N, T, D)

        Returns:
            Fused features (B, N, T, D)
        """
        B, N, T, D = node.shape  # noqa: N806

        # Project to Q, K, V
        Q = self.q_proj(node)  # noqa: N806
        K = self.k_proj(edge)  # noqa: N806
        V = self.v_proj(edge)  # noqa: N806

        # Reshape for multi-head attention
        Q = Q.reshape(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)  # noqa: N806
        K = K.reshape(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)  # noqa: N806
        V = V.reshape(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)  # noqa: N806
        # Shape: (B, N, num_heads, T, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)

        # Reshape back
        out = out.transpose(2, 3).reshape(B, N, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection with node features
        return node + out