"""Dynamic graph builder for time-varying electrode connectivity.

Based on EvoBrain implementation with proven parameters for EEG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class DynamicGraphBuilder(nn.Module):
    """Build time-evolving adjacency matrices from features.

    FROM EVOBRAIN (lines 970-981):
    - Dynamic graph per timestep
    - Top-k sparsification (critical!)
    - Threshold pruning at 1e-4
    - Symmetric adjacency for undirected graphs
    """

    def __init__(
        self,
        similarity: str = "cosine",  # EvoBrain default
        top_k: int = 3,  # EvoBrain: proven best for EEG
        threshold: float = 1e-4,  # EvoBrain: edge weight cutoff
        temperature: float = 0.1,
    ):
        super().__init__()
        self.similarity = similarity
        self.top_k = top_k
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Build dynamic adjacency matrices.

        Args:
            features: (B, 19, T, D) electrode features

        Returns:
            adjacency: (B, T, 19, 19) time-varying adjacency
        """
        batch_size, n_nodes, seq_len, feat_dim = features.shape

        # Reshape for batch processing
        features_flat = features.permute(0, 2, 1, 3)  # (B, T, 19, D)
        features_flat = features_flat.reshape(batch_size * seq_len, n_nodes, feat_dim)

        # Compute similarity
        if self.similarity == "cosine":
            # Normalize features
            features_norm = func.normalize(features_flat, p=2, dim=-1)
            # Compute cosine similarity
            adjacency = torch.bmm(features_norm, features_norm.transpose(1, 2))
            # Scale by temperature
            adjacency = adjacency / self.temperature
        elif self.similarity == "correlation":
            # Center features
            features_centered = features_flat - features_flat.mean(dim=-1, keepdim=True)
            # Compute correlation
            adjacency = torch.bmm(features_centered, features_centered.transpose(1, 2))
            # Normalize
            std = features_centered.std(dim=-1, keepdim=True) + 1e-6
            adjacency = adjacency / (std @ std.transpose(1, 2))
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

        # Apply softmax for probability distribution
        adjacency = func.softmax(adjacency, dim=-1)

        # Top-k sparsification (EvoBrain critical!)
        if self.top_k < n_nodes:
            # Keep only top-k edges per node
            topk_vals, topk_idx = torch.topk(adjacency, self.top_k, dim=-1)
            adjacency_sparse = torch.zeros_like(adjacency)
            adjacency_sparse.scatter_(-1, topk_idx, topk_vals)
            adjacency = adjacency_sparse

        # Threshold pruning (EvoBrain: remove weak edges)
        adjacency = torch.where(adjacency > self.threshold, adjacency, torch.zeros_like(adjacency))

        # Make symmetric (undirected graph for EEG)
        adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2

        # Reshape back
        adjacency = adjacency.reshape(batch_size, seq_len, n_nodes, n_nodes)

        return adjacency
