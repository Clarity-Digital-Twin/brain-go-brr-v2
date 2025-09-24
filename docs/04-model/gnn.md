# GraphChannelMixer (PyG)

File: `src/brain_brr/models/gnn_pyg.py`

- Vectorized over all timesteps (B×T graphs)
- Dynamic Laplacian PE by default (k=16), computed from the learned adjacency per timestep
- 2× SSGConv (α=0.05) with residual + LayerNorm + Dropout
- Bypass edge transform when edge weights already Softplus’ed upstream

Vectorized path (V3)

- Flattens `(B,19,T,D)` → `(B*T,19,D)` and `(B,T,19,19)` → `(B*T,19,19)`.
- Builds a disjoint batch for PyG; constructs `edge_index`/`edge_weight` from the adjacency.
- Concatenates Laplacian PE (k=16) to node features only on the first GNN layer.
- Applies SSGConv → LayerNorm → Dropout; residuals from layer 2 onward.
- Returns `(B,19,T,D)` reshaped back from the batched result.

Dynamic vs static PE

- Dynamic PE is configurable via `graph.use_dynamic_pe` (default false in schema for backward compat; recommended true for V3).
- Dynamic PE implementation is vectorized across B×T with sign-consistency and optional semi-dynamic interval (`graph.semi_dynamic_interval`).
- Static PE remains available when `use_dynamic_pe: false` and is computed once from the structural 10–20 montage.

Bypass edge transform

- In V3, `bypass_edge_transform=True` because edge weights are already transformed by `Linear+Softplus` in the edge stream.
