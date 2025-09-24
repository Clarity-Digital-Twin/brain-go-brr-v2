# GraphChannelMixer (PyG)

File: `src/brain_brr/models/gnn_pyg.py`

- Vectorized over all timesteps (B*960 graphs)
- Static Laplacian PE (k=16) from structural 10–20 graph
- 2× SSGConv (α=0.05) with residual + norm
- Bypass edge transform when edge weights already Softplus’ed
