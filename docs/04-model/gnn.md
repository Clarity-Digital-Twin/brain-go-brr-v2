# GraphChannelMixer (PyG)

File: `src/brain_brr/models/gnn_pyg.py`

- Vectorized over all timesteps (B*960 graphs)
- Static Laplacian PE (k=16) from structural 10–20 graph
- 2× SSGConv (α=0.05) with residual + norm
- Bypass edge transform when edge weights already Softplus’ed

Vectorized path (V3)

- Flattens `(B,19,T,D)` to `(B*T,19,D)` and `(B,T,19,19)` to `(B*T,19,19)`.
- Builds a disjoint batch of graphs by iterating Python‑side over `B*T` to construct `edge_index` and `edge_weight` from the adjacency.
- Concatenates static Laplacian PE (k=16) to node features only on the first GNN layer.
- Applies SSGConv → LayerNorm → Dropout; residuals from layer 2 onward.
- Returns `(B,19,T,D)` reshaped back from the batched result.

Static vs dynamic PE

- V3 forces static PE: `use_dynamic_pe=False` in model construction; config fields for dynamic PE are ignored in V3.
- Dynamic PE is only relevant for V2‑style experiments.

Bypass edge transform

- In V3, `bypass_edge_transform=True` because edge weights have already passed through a `Linear+Softplus` in the edge stream.
