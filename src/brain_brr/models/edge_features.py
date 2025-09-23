"""Edge feature extraction and adjacency assembly for V3 dual-stream architecture.

This module implements the edge stream processing pipeline:
1. Generate all unique electrode pairs (171 for 19 nodes)
2. Compute edge features via cosine similarity
3. Assemble learned adjacency matrices with sparsity control
"""

from __future__ import annotations

import torch


def pair_indices_undirected(n: int = 19) -> list[tuple[int, int]]:
    """Generate all unique undirected pairs for n nodes.

    For 19 electrodes: n*(n-1)/2 = 19*18/2 = 171 pairs.

    Args:
        n: Number of nodes (default 19 for 10-20 montage)

    Returns:
        List of (i,j) tuples where i < j, length n*(n-1)/2

    Examples:
        >>> pairs = pair_indices_undirected(3)
        >>> pairs
        [(0, 1), (0, 2), (1, 2)]
        >>> len(pair_indices_undirected(19))
        171
    """
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def edge_scalar_series(
    elec: torch.Tensor,
    *,
    metric: str = "cosine",
    pairs: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Compute edge features as scalar similarity series (vectorized).

    Computes pairwise similarities for all node pairs at each timestep, then
    packs the upper-triangular (i<j) entries into an edge list.

    Args:
        elec: Electrode features (B, N, T, D)
        metric: "cosine" or "correlation"
        pairs: Optional precomputed (i,j) list (len=E)

    Returns:
        (B, E, T, 1) edge feature tensor
    """
    _, n_nodes, _, _ = elec.shape
    device = elec.device

    if pairs is None:
        pairs = pair_indices_undirected(n_nodes)
    idx_i = torch.tensor([i for i, _ in pairs], device=device, dtype=torch.long)
    idx_j = torch.tensor([j for _, j in pairs], device=device, dtype=torch.long)

    # Reorder to (B, T, N, D)
    x = elec.permute(0, 2, 1, 3)

    if metric == "cosine":
        x_norm = x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))  # (B,T,N,N)
    elif metric == "correlation":
        x_center = x - x.mean(dim=-1, keepdim=True)
        num = torch.matmul(x_center, x_center.transpose(-1, -2))
        denom = torch.sqrt(x_center.pow(2).sum(dim=-1) + 1e-8)
        denom_mat = denom.unsqueeze(-1) * denom.unsqueeze(-2)
        sim = num / denom_mat
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Select upper-triangular entries into E
    sim_e = sim[:, :, idx_i, idx_j]  # (B,T,E)
    return sim_e.permute(0, 2, 1).unsqueeze(-1)


def assemble_adjacency(
    edge_weights: torch.Tensor,
    *,
    n_nodes: int = 19,
    top_k: int = 3,
    threshold: float = 1e-4,
    symmetric: bool = True,
    identity_fallback: bool = True,
    pairs: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Assemble adjacency matrices from edge weights.

    Converts flat edge weights to adjacency matrices with:
    1. Top-k sparsification per row
    2. Threshold pruning
    3. Symmetrization
    4. Identity fallback for disconnected nodes

    Args:
        edge_weights: Edge weights (B, E, T) where E = n_nodes*(n_nodes-1)/2
        n_nodes: Number of nodes
        top_k: Keep top-k edges per node
        threshold: Minimum edge weight to keep
        symmetric: Make adjacency symmetric
        identity_fallback: Add self-loops for disconnected nodes
        pairs: Pre-computed pairs, if None uses pair_indices_undirected

    Returns:
        Adjacency matrices (B, T, n_nodes, n_nodes)

    Examples:
        >>> weights = torch.rand(2, 171, 960)
        >>> adj = assemble_adjacency(weights)
        >>> adj.shape
        torch.Size([2, 960, 19, 19])
    """
    batch_size, _, seq_len = edge_weights.shape
    device = edge_weights.device

    if pairs is None:
        pairs = pair_indices_undirected(n_nodes)

    # Initialize adjacency matrices
    adj = torch.zeros(batch_size, seq_len, n_nodes, n_nodes, device=device)

    # Fill adjacency from edge weights (vectorized)
    idx_i = torch.tensor([i for i, _ in pairs], device=device, dtype=torch.long)
    idx_j = torch.tensor([j for _, j in pairs], device=device, dtype=torch.long)
    adj_bt = adj.view(batch_size * seq_len, n_nodes, n_nodes)
    w_bt = edge_weights.permute(0, 2, 1).reshape(batch_size * seq_len, -1)  # (B*T,E)
    bt = torch.arange(batch_size * seq_len, device=device).unsqueeze(1)
    adj_bt[bt, idx_i.unsqueeze(0).expand_as(w_bt), idx_j.unsqueeze(0).expand_as(w_bt)] = w_bt
    if symmetric:
        adj_bt[bt, idx_j.unsqueeze(0).expand_as(w_bt), idx_i.unsqueeze(0).expand_as(w_bt)] = w_bt

    # Apply top-k sparsification per row
    if top_k < n_nodes:
        k = min(top_k, n_nodes)
        adj_flat = adj.view(batch_size * seq_len, n_nodes, n_nodes)
        topk_vals, topk_idx = torch.topk(adj_flat, k, dim=-1)
        adj_sparse = torch.zeros_like(adj_flat)
        adj_sparse.scatter_(-1, topk_idx, topk_vals)
        adj = adj_sparse.view(batch_size, seq_len, n_nodes, n_nodes)

    # Apply threshold
    adj = torch.where(adj > threshold, adj, torch.zeros_like(adj))

    # Ensure symmetry (after sparsification)
    if symmetric:
        adj = (adj + adj.transpose(-1, -2)) / 2

    # Identity fallback for disconnected nodes
    if identity_fallback:
        row_sums = adj.sum(dim=-1)  # (B, T, n_nodes)

        # Find disconnected nodes
        disconnected = row_sums < threshold

        # Set diagonal via view assignment
        diag_view = torch.diagonal(adj, dim1=-2, dim2=-1)  # (B,T,N)
        diag_new = torch.where(disconnected, torch.ones_like(diag_view), diag_view)
        diag_view.copy_(diag_new)

    return adj


def get_structural_adjacency(n_nodes: int = 19) -> torch.Tensor:
    """Get structural adjacency for 10-20 montage.

    Defines physical electrode neighbors based on standard positions.
    Used for static Laplacian PE computation.

    Args:
        n_nodes: Number of nodes (must be 19 for 10-20)

    Returns:
        Binary adjacency matrix (n_nodes, n_nodes)

    Raises:
        ValueError: If n_nodes != 19
    """
    if n_nodes != 19:
        raise ValueError(f"Structural adjacency only defined for 19 nodes, got {n_nodes}")

    adj = torch.zeros(19, 19)

    # Define edges based on physical proximity
    edges = [
        # Frontal connections
        (0, 1),
        (0, 4),  # Fp1 - F3, Fp1 - F7
        (11, 12),
        (11, 15),  # Fp2 - F4, Fp2 - F8
        # Central chain (left)
        (1, 2),
        (2, 3),  # F3 - C3, C3 - P3
        # Central chain (right)
        (12, 13),
        (13, 14),  # F4 - C4, C4 - P4
        # Temporal chain (left)
        (4, 5),
        (5, 6),  # F7 - T3, T3 - T5
        # Temporal chain (right)
        (15, 16),
        (16, 17),  # F8 - T4, T4 - T6
        # Occipital connections (left)
        (3, 7),
        (6, 7),  # P3 - O1, T5 - O1
        # Occipital connections (right)
        (14, 18),
        (17, 18),  # P4 - O2, T6 - O2
        # Midline chain
        (8, 9),
        (9, 10),  # Fz - Cz, Cz - Pz
        # Cross-hemisphere (frontal)
        (1, 8),
        (8, 12),  # F3 - Fz, Fz - F4
        # Cross-hemisphere (central)
        (2, 9),
        (9, 13),  # C3 - Cz, Cz - C4
        # Cross-hemisphere (parietal)
        (3, 10),
        (10, 14),  # P3 - Pz, Pz - P4
    ]

    # Fill adjacency matrix
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1  # Symmetric

    return adj
