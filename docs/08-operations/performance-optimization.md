# Performance Optimization

- Use vectorized GNN path (default)
- Keep batch sizes conservative on 24GB VRAM
- Avoid Python loops in hot paths (see potential GNN batching optimization)
- Use static PE to reduce overhead
