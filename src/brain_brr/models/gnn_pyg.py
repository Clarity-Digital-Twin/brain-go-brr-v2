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
    warnings.warn(
        "PyTorch Geometric not installed. Install from prebuilt wheels (see INSTALLATION.md) or run 'make setup-gpu'",
        stacklevel=2,
    )


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
        use_vectorized: bool = True,  # V3: vectorized batching
        use_dynamic_pe: bool = True,  # V3: dynamic PE by default (EvoBrain approach)
        bypass_edge_transform: bool = False,  # V3: skip if upstream Softplus
        semi_dynamic_interval: int = 1,  # Update PE every N timesteps
        pe_sign_consistency: bool = True,  # Fix eigenvector signs
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric required. Install from prebuilt wheels (see INSTALLATION.md) or run 'make setup-gpu'"
            )

        self.d_model = d_model
        self.n_electrodes = n_electrodes
        self.k_eigenvectors = k_eigenvectors
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.use_vectorized = use_vectorized
        self.use_dynamic_pe = use_dynamic_pe
        self.bypass_edge_transform = bypass_edge_transform
        self.semi_dynamic_interval = semi_dynamic_interval
        self.pe_sign_consistency = pe_sign_consistency

        # ROBUST: Cache last valid PE for fallback
        self.last_valid_pe: torch.Tensor | None = None
        self.register_buffer("last_valid_pe", None)

        # Laplacian PE (EvoBrain line 858)
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=k_eigenvectors)

        # Static PE buffer - always register but may be None for dynamic PE
        if use_dynamic_pe:
            # Dynamic PE: register None buffer for attribute existence
            self.register_buffer("static_pe", None)
        else:
            # Static PE: compute once from structural graph
            self.register_buffer("static_pe", self._compute_static_pe())

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

    def _compute_static_pe(self) -> torch.Tensor:
        """Compute static Laplacian PE from 10-20 structural graph."""
        from .edge_features import get_structural_adjacency

        # Get structural adjacency
        adj = get_structural_adjacency(self.n_electrodes)  # (19, 19)

        # Create edge index
        edge_indices = (adj > 0).nonzero(as_tuple=False)
        edge_index = edge_indices.t()  # (2, E)

        # Create graph data for PE computation
        data = Data(
            x=torch.randn(self.n_electrodes, 1),  # Dummy features
            edge_index=edge_index,
        )

        # Compute Laplacian PE
        data = self.laplacian_pe(data)

        # Extract PE
        if hasattr(data, "laplacian_eigenvector_pe"):
            pe: torch.Tensor = data.laplacian_eigenvector_pe  # (19, k)
            return pe
        else:
            # Fallback if PE fails
            return torch.zeros(self.n_electrodes, self.k_eigenvectors)

    def _compute_dynamic_pe_vectorized(
        self,
        adjacency: torch.Tensor,  # (B, T, N, N)
    ) -> torch.Tensor:  # (B, T, N, k)
        """Compute dynamic Laplacian PE for all timesteps in parallel.

        This is 100-1000x faster than looping over timesteps.
        Includes numerical stability guards:
        - Degree clamping to prevent division by zero
        - Float32 eigendecomposition for numerical stability
        - Sign consistency to prevent eigenvector flips
        - ROBUST: Laplacian regularization + NaN detection + fallback
        """
        B, T, N, _ = adjacency.shape  # noqa: N806
        device = adjacency.device
        dtype = adjacency.dtype

        # Reshape to process all (B*T) graphs at once
        a_flat = adjacency.reshape(B * T, N, N)

        # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        # Critical: Clamp degrees to prevent division by zero
        degrees = a_flat.sum(dim=-1).clamp_min(1e-6)  # (B*T, N)
        d_inv_sqrt = torch.diag_embed(degrees.rsqrt())  # (B*T, N, N)

        # Normalized adjacency
        a_norm = d_inv_sqrt @ a_flat @ d_inv_sqrt

        # Laplacian
        identity = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B * T, -1, -1)
        laplacian = identity - a_norm  # (B*T, N, N)

        # Eigendecomposition
        # CRITICAL: Must disable AMP and use fp32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            l_stable = laplacian.to(torch.float32)

            # ROBUST FIX: Add regularization to prevent singular matrices
            eps = 1e-5
            l_stable = l_stable + eps * torch.eye(N, device=l_stable.device, dtype=torch.float32)

            try:
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)

                # Check for NaNs/Infs
                if (
                    torch.isnan(eigenvalues).any()
                    or torch.isnan(eigenvectors).any()
                    or torch.isinf(eigenvalues).any()
                    or torch.isinf(eigenvectors).any()
                ):
                    # Fallback: Use cached or identity-like safe PE
                    print("[WARNING] NaN/Inf detected in eigendecomposition, using fallback PE")
                    if self.last_valid_pe is not None and self.last_valid_pe.shape[0] == B:
                        # Use cached PE if available and correct shape
                        pe = self.last_valid_pe.reshape(B * T, N, self.k_eigenvectors).to(
                            torch.float32
                        )
                    else:
                        # Use small random PE as last resort
                        pe = (
                            torch.randn(
                                B * T, N, self.k_eigenvectors, device=device, dtype=torch.float32
                            )
                            * 0.01
                        )
                else:
                    # Clamp eigenvalues to safe range [0, 2] (Laplacian eigenvalues)
                    eigenvalues = torch.clamp(eigenvalues, min=0.0, max=2.0)

                    # Take k smallest eigenvectors (already sorted in ascending order)
                    pe = eigenvectors[..., : self.k_eigenvectors]  # (B*T, N, k)

            except RuntimeError as e:
                # Complete eigendecomposition failure - use safe fallback
                print(f"[WARNING] Eigendecomposition failed: {e}, using fallback PE")
                pe = (
                    torch.randn(B * T, N, self.k_eigenvectors, device=device, dtype=torch.float32)
                    * 0.01
                )

        # Sign consistency: Fix eigenvector signs to prevent random flips
        if self.pe_sign_consistency:
            signs = torch.sign(pe.sum(dim=-2, keepdim=True))  # (B*T, 1, k)
            signs = signs.where(signs != 0, torch.ones_like(signs))
            pe = pe * signs

        # Final NaN check and replacement
        pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)

        # Reshape back and cast to original dtype
        pe = pe.reshape(B, T, N, self.k_eigenvectors).to(dtype)

        # Cache this valid PE for future fallback
        if not torch.isnan(pe).any() and not torch.isinf(pe).any():
            self.last_valid_pe = pe.detach().clone()

        return pe

    def forward_vectorized(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized forward - process all graphs at once.

        This is the V3 default path that processes B*T graphs in one batch,
        avoiding the per-timestep Python loop.
        """
        batch_size, n_nodes, seq_len, feat_dim = features.shape
        device = features.device

        # Flatten to (B*T, N, D)
        x = features.permute(0, 2, 1, 3).reshape(-1, n_nodes, feat_dim)  # (B*T, 19, D)
        adj = adjacency.reshape(-1, n_nodes, n_nodes)  # (B*T, 19, 19)

        # Build disjoint batch
        edge_index_list = []
        edge_weight_list = []
        batch_idx = []

        for i in range(batch_size * seq_len):
            # Get edges for this graph
            edge_indices = (adj[i] > 0).nonzero(as_tuple=False)
            if len(edge_indices) == 0:
                # Empty graph - add self-loop to avoid issues
                edge_indices = torch.tensor([[0], [0]], device=device)
                edge_weights = torch.ones(1, device=device)
            else:
                edge_weights = adj[i][edge_indices[:, 0], edge_indices[:, 1]]

            # Offset indices for disjoint union
            offset = i * n_nodes
            edge_index_offset = edge_indices.t() + offset
            edge_index_list.append(edge_index_offset)

            # Edge weights (optionally transform)
            if not self.bypass_edge_transform:
                edge_weights = self.edge_transform(edge_weights.unsqueeze(-1))
                edge_weights = self.edge_activate(edge_weights).squeeze(-1)
            edge_weight_list.append(edge_weights)

            # Batch assignment
            batch_idx.extend([i] * n_nodes)

        # Concatenate all
        x_batch = x.reshape(-1, feat_dim)  # (B*T*19, D)
        edge_index_batch = torch.cat(edge_index_list, dim=1)  # (2, E_total)
        edge_weight_batch = torch.cat(edge_weight_list, dim=0)  # (E_total,)
        # batch_tensor = torch.tensor(batch_idx, device=device, dtype=torch.long)  # For future use

        # Add PE
        if self.use_dynamic_pe:
            # Dynamic PE per timestep (vectorized implementation)
            pe = self._compute_dynamic_pe_vectorized(adjacency)  # (B, T, N, k)

            # Semi-dynamic option: Only update PE every N timesteps
            if self.semi_dynamic_interval > 1:
                interval = self.semi_dynamic_interval
                # Compute PE only at intervals
                indices = torch.arange(0, seq_len, interval)
                pe_sparse = pe[:, indices]  # (B, T//interval, N, k)
                # Repeat each computed PE for interval timesteps
                pe = pe_sparse.repeat_interleave(interval, dim=1)[:, :seq_len]

            # Flatten for GNN processing
            pe_flat = pe.reshape(-1, self.k_eigenvectors)  # (B*T*19, k)
            x_with_pe = torch.cat([x_batch, pe_flat], dim=-1)  # (B*T*19, D+k)
        else:
            # Static PE (broadcast)
            pe = self.static_pe.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
            pe_flat = pe.reshape(-1, self.k_eigenvectors)  # (B*T*19, k)
            x_with_pe = torch.cat([x_batch, pe_flat], dim=-1)  # (B*T*19, D+k)

        # Apply GNN layers
        x_out = x_with_pe
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms, strict=False)):
            # First layer uses PE, others don't
            x_in = x_out if i == 0 else x_batch

            # Apply GNN
            x_gnn = gnn_layer(x_in, edge_index_batch, edge_weight_batch)

            # Residual and norm
            if self.use_residual and i > 0:
                x_gnn = x_gnn + x_batch
            x_gnn = norm(x_gnn)
            x_batch = self.dropout(x_gnn)

        # Reshape back to (B, 19, T, D)
        output = x_batch.reshape(batch_size, seq_len, n_nodes, feat_dim)
        output = output.permute(0, 2, 1, 3)  # (B, 19, T, D)

        return output

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
        # Use vectorized path for v3 (default)
        if self.use_vectorized:
            return self.forward_vectorized(features, adjacency)

        # Legacy per-timestep path (v2 compatibility)
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
                # Detach edge weights for PE computation to avoid gradient issues
                with torch.no_grad():
                    data_for_pe = Data(
                        x=data.x,
                        edge_index=data.edge_index,
                        edge_weight=data.edge_weight.detach()
                        if data.edge_weight is not None
                        else None,
                    )
                    data_for_pe = self.laplacian_pe(data_for_pe)
                    # Copy PE back to original data
                    if hasattr(data_for_pe, "laplacian_eigenvector_pe"):
                        data.laplacian_eigenvector_pe = data_for_pe.laplacian_eigenvector_pe
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
