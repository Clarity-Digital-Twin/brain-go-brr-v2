# Performance Optimization

- Use vectorized GNN path (default)
- Keep batch sizes conservative on 24GB VRAM
- Avoid Python loops in hot paths (GNN batching is already vectorized)
- Dynamic PE is recommended for V3 (default in configs).
  - Safeguards in `gnn_pyg.py` handle ill‑conditioned adjacencies (regularization, cached PE fallback, nan_to_num).
  - If you need extra headroom (ablation), set `graph.use_dynamic_pe: false` to use static PE.
