# Edge Features and Adjacency

File: `src/brain_brr/models/edge_features.py`

- Pairwise metric: cosine (default) or correlation
- 171 unique edges for 19 electrodes
- Post Mamba weights → assemble adjacency `(B, 960, 19, 19)`
- Top-k per node (default 3), threshold prune (1e-4), symmetrize, identity fallback
