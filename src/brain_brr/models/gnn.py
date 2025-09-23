"""Graph neural network module for spatial reasoning.

Pure PyTorch implementation based on EvoBrain architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class GraphChannelMixer(nn.Module):
    """Dynamic GNN with Laplacian PE support.

    FROM EVOBRAIN MODEL:
    - SSGConv with alpha=0.05 (line 332)
    - Edge transform + Softplus (lines 869-870)
    - 2-layer GNN with skip connections
    - Laplacian PE concatenated to node features
    """

    def __init__(
        self,
        d_model: int = 512,
        n_electrodes: int = 19,
        n_layers: int = 2,  # EvoBrain: 2-layer GNN
        dropout: float = 0.1,
        use_residual: bool = True,
        alpha: float = 0.05,  # SSGConv mixing parameter
    ):
        super().__init__()
        self.d_model = d_model
        self.n_electrodes = n_electrodes
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.alpha = alpha  # SSGConv alpha for EEG

        # Edge weight transform (EvoBrain lines 869-870)
        self.edge_transform = nn.Linear(1, 1)
        self.edge_activate = nn.Softplus()

        # Graph convolution layers (SSGConv-like behavior)
        self.graph_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # Layer norm and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Apply graph neural network.

        Args:
            features: (B, 19, T, D) electrode features
            adjacency: (B, T, 19, 19) dynamic adjacency

        Returns:
            enhanced: (B, 19, T, D) enhanced features
        """
        batch_size, n_nodes, seq_len, feat_dim = features.shape

        # Reshape for batch processing
        x = features.permute(0, 2, 1, 3)  # (B, T, 19, D)
        x = x.reshape(batch_size * seq_len, n_nodes, feat_dim)
        adj = adjacency.reshape(batch_size * seq_len, n_nodes, n_nodes)

        # Transform edge weights (EvoBrain style)
        adj_weights = self.edge_transform(adj.unsqueeze(-1))
        adj_weights = self.edge_activate(adj_weights).squeeze(-1)

        # Normalize adjacency (row-wise for stability)
        row_sum = adj_weights.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj_weights / row_sum

        # Robustness: if a row has no outgoing edges after sparsification/thresholding,
        # fall back to an identity row so the node keeps its own features.
        # This prevents degenerate zero rows from producing undefined behavior.
        row_has_edges = (adj_weights > 0).any(dim=-1, keepdim=True)
        if not row_has_edges.all():
            identity = torch.eye(n_nodes, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(
                0
            )  # (1, N, N)
            adj_norm = adj_norm * row_has_edges + identity * (~row_has_edges)

        # Apply GNN layers
        for i in range(self.n_layers):
            # Store residual
            residual = x if self.use_residual else 0

            # Graph convolution: aggregate neighbor features
            # This implements SSGConv-like behavior with alpha mixing
            x_neighbors = torch.bmm(adj_norm, x)  # (B*T, 19, D)

            # Mix self and neighbor features (SSGConv alpha)
            x_mixed = (1 - self.alpha) * x + self.alpha * x_neighbors

            # Transform features
            x = self.graph_layers[i](x_mixed)

            # Add residual
            if self.use_residual:
                x = x + residual

            # Layer norm and activation
            x = self.layer_norms[i](x)
            x = func.gelu(x)
            x = self.dropout(x)

        # Reshape back
        x = x.reshape(batch_size, seq_len, n_nodes, feat_dim)
        x = x.permute(0, 2, 1, 3)  # (B, 19, T, D)

        return x
