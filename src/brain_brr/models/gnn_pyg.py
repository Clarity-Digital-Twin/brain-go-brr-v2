"""PyTorch Geometric implementation with Laplacian PE.

Based on EvoBrain architecture with proven EEG parameters.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as func

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import SSGConv
    from torch_geometric.transforms import AddLaplacianEigenvectorPE

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not installed. Install with: uv sync -E graph", stacklevel=2)


class GraphChannelMixerPyG(nn.Module):
    """Dynamic GNN with Laplacian PE using PyTorch Geometric.

    FROM EVOBRAIN:
    - SSGConv with alpha=0.05 (line 332)
    - Laplacian PE k=16 (line 858)
    - Edge transform with Softplus (lines 869-870)
    - 2-layer GNN with residuals
    """

    def __init__(
        self,
        d_model: int = 64,  # Per-electrode feature dimension
        n_electrodes: int = 19,
        k_eigenvectors: int = 16,  # EvoBrain default
        alpha: float = 0.05,  # SSGConv alpha for EEG
        k_hops: int = 2,  # 2-hop neighborhood
        n_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric required for this module. Install with: uv sync -E graph"
            )

        self.d_model = d_model
        self.n_electrodes = n_electrodes
        self.k_eigenvectors = k_eigenvectors
        self.n_layers = n_layers
        self.use_residual = use_residual

        # Laplacian PE (EvoBrain line 858)
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=k_eigenvectors)

        # Edge weight transform (EvoBrain lines 869-870)
        self.edge_transform = nn.Linear(1, 1)
        self.edge_activate = nn.Softplus()

        # SSGConv layers (EvoBrain lines 331-334)
        # First layer: input dim includes PE
        input_dim = d_model + k_eigenvectors

        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            layer_input_dim = input_dim if i == 0 else d_model
            self.gnn_layers.append(
                SSGConv(
                    in_channels=layer_input_dim,
                    out_channels=d_model,
                    alpha=alpha,
                    K=k_hops,
                )
            )

        # Normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GNN with Laplacian PE.

        Args:
            features: (B, 19, T, D) electrode features
            adjacency: (B, T, 19, 19) dynamic adjacency

        Returns:
            enhanced: (B, 19, T, D) enhanced features
        """
        batch_size, n_nodes, seq_len, _ = features.shape
        device = features.device

        # Process each timestep
        outputs = []
        for t in range(seq_len):
            # Get features and adjacency for this timestep
            x_t = features[:, :, t, :]  # (B, 19, D)
            adj_t = adjacency[:, t, :, :]  # (B, 19, 19)

            # Create batch of graphs
            batch_list = []
            for b in range(batch_size):
                # Create edge index from adjacency (only non-zero edges)
                edge_indices = (adj_t[b] > 0).nonzero(as_tuple=False)
                edge_index = edge_indices.t()  # (2, E)
                edge_weight = adj_t[b][edge_indices[:, 0], edge_indices[:, 1]]  # (E,)

                # Transform edge weights (EvoBrain style)
                edge_weight = self.edge_transform(edge_weight.unsqueeze(-1))
                edge_weight = self.edge_activate(edge_weight).squeeze(-1)

                # Create graph data
                data = Data(
                    x=x_t[b],  # (19, D)
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

                # Add Laplacian PE (EvoBrain line 950)
                data = self.laplacian_pe(data)
                batch_list.append(data)

            # Batch graphs
            batch = Batch.from_data_list(batch_list).to(device)

            # Concatenate features with PE for first layer
            x = batch.x  # (B*19, D)
            if hasattr(batch, "laplacian_eigenvector_pe"):
                pe = batch.laplacian_eigenvector_pe  # (B*19, k_eigenvectors)
                x_with_pe = torch.cat([x, pe], dim=-1)  # (B*19, D+k)
            else:
                # Fallback if PE computation fails (e.g., disconnected graph)
                x_with_pe = torch.cat(
                    [x, torch.zeros(x.size(0), self.k_eigenvectors).to(device)], dim=-1
                )

            # Apply GNN layers
            for i in range(self.n_layers):
                # Use PE-concatenated features only for first layer
                layer_input = x_with_pe if i == 0 else x

                # Store residual
                residual = x if (self.use_residual and i > 0) else None

                # Apply SSGConv
                x = self.gnn_layers[i](
                    layer_input,
                    batch.edge_index,
                    batch.edge_weight,
                )

                # Add residual
                if residual is not None:
                    x = x + residual

                # Layer norm and activation
                x = self.layer_norms[i](x)
                x = func.gelu(x)
                x = self.dropout(x)

            # Reshape back to batch
            x = x.reshape(batch_size, n_nodes, self.d_model)
            outputs.append(x)

        # Stack timesteps
        output = torch.stack(outputs, dim=2)  # (B, 19, T, D)

        return output
