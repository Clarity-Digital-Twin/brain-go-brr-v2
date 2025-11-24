"""PyTorch Geometric implementation with Laplacian PE.

Based on EvoBrain architecture with proven EEG parameters.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as func

from src.brain_brr.constants import (
    EIGENVALUE_CLAMP_MAX,
    EPSILON_LAPLACIAN,
    EPSILON_NUMERICAL,
    GNN_SSGCONV_ALPHA_DEFAULT,
)

from .adjacency import compute_stable_laplacian, condition_adjacency

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import WarmupScheduleConfig

# Module logger
logger = logging.getLogger(__name__)

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
        alpha: float = GNN_SSGCONV_ALPHA_DEFAULT,  # SSGConv alpha for EEG
        k_hops: int = 2,  # 2-hop neighborhood
        n_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_vectorized: bool = True,  # V3: vectorized batching
        use_dynamic_pe: bool = True,  # V3: dynamic PE by default (EvoBrain approach)
        bypass_edge_transform: bool = False,  # V3: skip if upstream Softplus
        semi_dynamic_interval: int = 1,  # Update PE every N timesteps
        pe_sign_consistency: bool = True,  # Fix eigenvector signs
        # PR-3: Adjacency conditioning parameters
        adj_row_softmax: bool = False,
        adj_softmax_tau: float = 1.0,
        adj_ema_beta: float | None = None,
        adj_force_symmetric: bool = False,
        laplacian_eps: float = EPSILON_LAPLACIAN,
        warmup_config: WarmupScheduleConfig | None = None,  # v3.4.1: Warmup schedules
        laplacian_normalize: bool = True,
    ):
        """Initialize GraphChannelMixerPyG with dynamic Laplacian PE.

        Args:
            d_model: Feature dimension per electrode (default: 64, EvoBrain baseline)
            n_electrodes: Number of EEG electrodes (default: 19)
            k_eigenvectors: Number of Laplacian eigenvectors for PE (default: 16)
            alpha: SSGConv skip connection strength (default: 0.05)
            k_hops: Neighborhood size for message passing (default: 2)
            n_layers: Number of GNN layers (default: 2)
            dropout: Dropout rate (default: 0.1)
            use_residual: Enable residual connections (default: True)
            use_vectorized: Use vectorized batch processing (default: True)
            use_dynamic_pe: Compute PE dynamically per timestep (default: True)
            bypass_edge_transform: Skip edge transform if upstream uses Softplus (default: False)
            semi_dynamic_interval: PE update frequency in timesteps (default: 1)
            pe_sign_consistency: Fix eigenvector sign flips (default: True)
            adj_row_softmax: Apply row-wise softmax to adjacency (default: False)
            adj_softmax_tau: Temperature for adjacency softmax (default: 1.0)
            adj_ema_beta: EMA decay for adjacency smoothing (default: None)
            adj_force_symmetric: Force adjacency matrix symmetry (default: False)
            laplacian_eps: Epsilon for Laplacian stability (default: EPSILON_LAPLACIAN)
            warmup_config: Optional warmup schedule for adjacency tau (default: None)
            laplacian_normalize: Normalize Laplacian (default: True)
        """
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

        # PR-3: Adjacency conditioning parameters
        self.adj_row_softmax = adj_row_softmax
        self.adj_softmax_tau = adj_softmax_tau
        self.adj_ema_beta = adj_ema_beta
        self.adj_force_symmetric = adj_force_symmetric
        self.laplacian_eps = laplacian_eps
        self.laplacian_normalize = laplacian_normalize

        # v3.4.1: Warmup schedules (stored config)
        self.warmup_config = warmup_config
        self.global_step = 0  # Updated from training loop

        # ROBUST: Cache last valid PE for fallback
        # CRITICAL: Initialize with dummy tensor (not None) to ensure buffer appears in state_dict
        # from initialization, preventing checkpoint incompatibility (see CHECKPOINT_BUFFER_BUG.md)
        # Placeholder shape (1,1,1,k) will be overwritten with actual PE (B,T,N,k) during forward
        self.register_buffer(
            "last_valid_pe",
            torch.zeros(1, 1, 1, k_eigenvectors, dtype=torch.float32),
            persistent=True,  # Explicit: save in checkpoints
        )
        self.last_valid_pe: torch.Tensor

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

    def set_global_step(self, step: int) -> None:
        """Update global step for warmup schedules (v3.4.1).

        Called from training loop before each forward pass.
        """
        self.global_step = step

    def set_warmup_config(self, config: WarmupScheduleConfig | None) -> None:
        """Update warmup configuration (v3.4.1).

        Safe to call at runtime when schedules change or are disabled.
        """
        self.warmup_config = config

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

    def _add_jitter_for_stability(
        self,
        laplacian: torch.Tensor,
        jitter_scale: float = 1e-5,
    ) -> torch.Tensor:
        """Add random diagonal jitter to break eigenvalue degeneracy.

        Args:
            laplacian: Laplacian matrix (B, N, N) or (N, N)
            jitter_scale: Scale of random noise (default 1e-5)

        Returns:
            Jittered Laplacian (in-place modified, but also returned)

        Why this works:
            Eigenvalue degeneracy (multiple λᵢ ≈ λⱼ) crashes cuSOLVER.
            Random jitter breaks symmetry: λᵢ + ε_i != λⱼ + ε_j
            Constant shift (laplacian_eps) preserves gaps: doesn't help.
        """
        if laplacian.ndim == 3:
            batch, N, _ = laplacian.shape  # noqa: N806
            jitter = (
                torch.randn(batch, N, device=laplacian.device, dtype=laplacian.dtype) * jitter_scale
            )
        else:
            N = laplacian.size(0)  # noqa: N806
            jitter = torch.randn(N, device=laplacian.device, dtype=laplacian.dtype) * jitter_scale
        laplacian.diagonal(dim1=-2, dim2=-1).add_(jitter)
        return laplacian

    def _validate_and_process_eigendecomp(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate eigendecomposition results and extract PE.

        Args:
            eigenvalues: Eigenvalues from torch.linalg.eigh
            eigenvectors: Eigenvectors from torch.linalg.eigh

        Returns:
            (pe, eigenvalues): Processed PE and clamped eigenvalues

        Raises:
            RuntimeError: If NaN/Inf detected
        """
        if (
            torch.isnan(eigenvalues).any()
            or torch.isnan(eigenvectors).any()
            or torch.isinf(eigenvalues).any()
            or torch.isinf(eigenvectors).any()
        ):
            raise RuntimeError("NaN/Inf in eigendecomposition")

        eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=EIGENVALUE_CLAMP_MAX)
        pe = eigenvectors[..., : self.k_eigenvectors]
        return pe, eigenvalues

    def _compute_dynamic_pe_vectorized(
        self,
        adjacency: torch.Tensor,  # (B, T, N, N)
    ) -> torch.Tensor:  # (B, T, N, k)
        """Compute dynamic Laplacian PE for all timesteps in parallel.

        Computes PE while maintaining gradient flow for adjacency learning.
        """
        B, T, N, _ = adjacency.shape  # noqa: N806
        device = adjacency.device
        dtype = adjacency.dtype

        # Use adjacency directly - gradients should flow for learning
        adj_pe = adjacency

        # PR-3: Condition adjacency matrix for stability (row-softmax/EMA/symmetry)
        if self.adj_row_softmax or self.adj_ema_beta or self.adj_force_symmetric:
            adj_pe = condition_adjacency(
                adj_pe,
                tau=self.adj_softmax_tau,
                force_symmetric=self.adj_force_symmetric,
                row_softmax=self.adj_row_softmax,
                ema_beta=self.adj_ema_beta,
                global_step=self.global_step,  # v3.4.1: Warmup support
                warmup_config=self.warmup_config,  # v3.4.1: Warmup support
            )

        # Reshape to process all (B*T) graphs at once
        a_flat = adj_pe.reshape(B * T, N, N)

        # PR-3: Use stable Laplacian computation
        laplacian = compute_stable_laplacian(
            a_flat,
            normalize=self.laplacian_normalize,
            eps=self.laplacian_eps,
        )  # (B*T, N, N)

        # Eigendecomposition in fp32 without AMP for numerical stability
        # CRITICAL: Device-aware autocast to support CPU/MPS training
        device_type = laplacian.device.type

        if device_type == "cuda":
            # CUDA: Disable AMP for eigendecomposition (numerical stability)
            with torch.amp.autocast("cuda", enabled=False):
                l_stable = laplacian.to(torch.float32)
                self._add_jitter_for_stability(l_stable)

                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
                    eigenvectors = eigenvectors.detach()
                    pe, eigenvalues = self._validate_and_process_eigendecomp(
                        eigenvalues, eigenvectors
                    )

                except RuntimeError as e:
                    logger.warning(f"GPU eigendecomp failed: {e}, trying CPU fallback")
                    try:
                        l_cpu = l_stable.cpu()
                        evals_cpu, evecs_cpu = torch.linalg.eigh(l_cpu)
                        eigenvalues = evals_cpu.to(device)
                        eigenvectors = evecs_cpu.to(device).detach()
                        pe, eigenvalues = self._validate_and_process_eigendecomp(
                            eigenvalues, eigenvectors
                        )
                        logger.info("CPU fallback successful")

                    except RuntimeError as cpu_e:
                        logger.warning(f"CPU eigendecomp failed: {cpu_e}, using last valid PE")
                        if self.last_valid_pe.shape[0] == B and self.last_valid_pe.shape[1] == T:
                            pe = self.last_valid_pe.reshape(B * T, N, self.k_eigenvectors).to(
                                torch.float32
                            )
                        else:
                            pe = (
                                torch.randn(
                                    B * T,
                                    N,
                                    self.k_eigenvectors,
                                    device=device,
                                    dtype=torch.float32,
                                )
                                * 0.01
                            )
        else:
            # CPU/MPS: No autocast needed (already in fp32 context)
            l_stable = laplacian.to(torch.float32)
            self._add_jitter_for_stability(l_stable)

            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
                eigenvectors = eigenvectors.detach()
                pe, eigenvalues = self._validate_and_process_eigendecomp(eigenvalues, eigenvectors)

            except RuntimeError as e:
                logger.warning(f"Eigendecomp failed: {e}, using fallback PE")
                if self.last_valid_pe.shape[0] == B and self.last_valid_pe.shape[1] == T:
                    pe = self.last_valid_pe.reshape(B * T, N, self.k_eigenvectors).to(torch.float32)
                else:
                    pe = (
                        torch.randn(
                            B * T,
                            N,
                            self.k_eigenvectors,
                            device=device,
                            dtype=torch.float32,
                        )
                        * 0.01
                    )

        # Sign consistency
        if self.pe_sign_consistency:
            signs = torch.sign(pe.sum(dim=-2, keepdim=True))  # (B*T, 1, k)
            signs = signs.where(signs != 0, torch.ones_like(signs))
            pe = pe * signs

        # Final NaN check and replacement
        pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)

        # Reshape back and cast to original dtype
        pe = pe.reshape(B, T, N, self.k_eigenvectors).to(dtype)

        # Cache this valid PE for future fallback (updates buffer in-place)
        if not torch.isnan(pe).any() and not torch.isinf(pe).any():
            # In-place update to maintain buffer status
            self.last_valid_pe = pe.detach().clone()

        # Return PE with gradients enabled for adjacency learning
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
                # Create in COO format directly (2, num_edges)
                edge_index_offset = (
                    torch.tensor([[0], [0]], device=device, dtype=torch.long) + i * n_nodes
                )
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
            # Semi-dynamic option: Only update PE every N timesteps
            if self.semi_dynamic_interval > 1:
                interval = self.semi_dynamic_interval
                # OPTIMIZED: Compute PE only at intervals (not all timesteps)
                indices = torch.arange(0, seq_len, interval, device=adjacency.device)
                # Extract adjacency only for selected timesteps
                adjacency_sparse = adjacency[:, indices]  # (B, T//interval, N, N)
                # Compute PE only for selected timesteps (5x faster!)
                pe_sparse = self._compute_dynamic_pe_vectorized(
                    adjacency_sparse
                )  # (B, T//interval, N, k)
                # Repeat each computed PE for interval timesteps
                pe = pe_sparse.repeat_interleave(interval, dim=1)[:, :seq_len]
            else:
                # Full dynamic: compute PE for every timestep
                pe = self._compute_dynamic_pe_vectorized(adjacency)  # (B, T, N, k)

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
        # PR-5: Removed conditional batch clamps (PR-3 conditioning provides stability)

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

        # Per-timestep path for compatibility
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
