# Laplacian Positional Encoding (LPE)

Goal
- Provide node positional features via graph Laplacian eigenvectors; support dynamic updates over time for V3.

Dynamic LPE (vectorized)
- Compute per‑timestep normalized Laplacian L and take the k smallest eigenvectors.
- Implementation uses a vectorized path across all timesteps; eigendecomposition runs with AMP disabled for stability.

Numerical stability
- Disable autocast for eigendecomposition; compute in float32 (or float64 if needed).
- Clamp degrees and add small diagonal regularization upstream to avoid singularities.
- Apply `nan_to_num` and cached‑PE fallback if needed (see `gnn_pyg.py`).
- Sign consistency: make each eigenvector’s sum non‑negative (or align to previous timestep if using temporal alignment).

Semi‑dynamic mode
- `semi_dynamic_interval: N` computes PE every N timesteps and repeats in between.
- Greatly reduces memory while preserving accuracy (interval 5 works well on RTX 4090).

Config knobs
```yaml
model:
  graph:
    use_dynamic_pe: true
    k_eigenvectors: 16
    semi_dynamic_interval: 1   # 1 = fully dynamic; 5–10 for memory relief
```

Code anchors
- Vectorized dynamic PE: `src/brain_brr/models/gnn_pyg.py`
- V3 architecture: `docs/04-model/v3-architecture.md`
