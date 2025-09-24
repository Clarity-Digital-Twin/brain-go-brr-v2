# Performance Optimization

- Use vectorized GNN path (default)
- Keep batch sizes conservative on 24GB VRAM
- Avoid Python loops in hot paths (GNN batching is already vectorized)
- Dynamic PE is recommended for V3 (default in configs); if you need extra headroom, set `graph.use_dynamic_pe: false` to use static PE.
