# Performance Optimization

- Use vectorized GNN path (default)
- Keep batch sizes conservative on 24GB VRAM
- Avoid Python loops in hot paths (GNN batching is already vectorized)
- Dynamic PE is recommended for V3 (default in configs).
  - Safeguards in `gnn_pyg.py` handle ill‑conditioned adjacencies (regularization, cached PE fallback, nan_to_num).
  - If you need extra headroom (ablation), set `graph.use_dynamic_pe: false` to use static PE.

Dynamic PE memory levers (practical)

- Semi‑dynamic PE: set `semi_dynamic_interval: 5–10` to reduce eigendecompositions 5–10× with negligible impact.
- Batch size: memory scales linearly; prefer `batch_size: 4` on RTX 4090 with dynamic PE.
- Full dynamic on 4090: use `batch_size: 3` if you must keep `interval: 1`.
- A100‑80GB: use `batch_size: 64`, `interval: 1`, `mixed_precision: true`.
