# Edge Features and Adjacency

File: `src/brain_brr/models/edge_features.py`

- Pairwise metric: cosine (default) or correlation
- 171 unique edges for 19 electrodes
- Post Mamba weights → assemble adjacency `(B, 960, 19, 19)`
- Top-k per node (default 3), threshold prune (1e-4), symmetrize, identity fallback

Details

- Pair index list is the undirected upper triangle (i<j) of the 19×19 matrix: `pair_indices_undirected()`.
- `edge_scalar_series` computes, for each timestep, the N×N similarity, then packs upper‑triangle entries to shape `(B,E,T,1)` with `E=171`.
- Temporal edge modeling applies a learned 1→D lift (`D=edge_mamba_d_model`, default 16), BiMamba2 on `(B*E,D,T)`, and a D→1 projection + Softplus.
- `assemble_adjacency` maps `(B,E,T)` back to `(B,T,N,N)` and applies:
  - Top‑k per row (sparsity)
  - Thresholding (prune weak edges)
  - Symmetry (average with transpose)
  - Identity fallback (guarantees at least self‑loops for disconnected nodes)

Notes

- In V3, adjacency is learned from the edge stream; the heuristic builder is only used in V2.
- For V3, `graph.edge_*` fields control sparsification and pruning.
