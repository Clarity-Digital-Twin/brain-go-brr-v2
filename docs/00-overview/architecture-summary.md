# Architecture Summary

- End‑to‑end: Input EEG → TCN → Dual‑Stream (Node Mamba, Edge Mamba) → Vectorized GNN (PyG) → Back‑Projection/Detection → Post‑processing
- Complexity: O(N) in sequence length across TCN, Mamba streams, and GNN (per timestep; vectorized over all timesteps)

ASCII overview

```
Input (B,19,15360)
  → TCNEncoder (8×, 64→512, stride_down=16) → (B,512,960)
  → Electrode proj 512→19×64 → (B,19,960,64)
    ├─ Node Mamba (BiMamba2, d_model=64, n_layers=6)
    └─ Edge features (cosine/corr) → (B,171,960,1)
         → 1→16 lift → Edge Mamba (BiMamba2, d_model=16, n_layers=2) → 16→1 → Softplus
         → Assemble adjacency (top‑k=3, thresh=1e‑4, sym, I+ fallback) → (B,960,19,19)
  → Vectorized GNN (PyG SSGConv×2, Laplacian PE k=16; dynamic by default) → (B,19,960,64)
  → Back‑proj 19×64→512 → (B,512,960) → Head (→19) → Upsample 960→15360 → Conv1d(19→1) → (B,15360)
```

Shapes (V3)

- Input: `(B, 19, 15360)`
- TCN out: `(B, 512, 960)`
- Electrode features: `(B, 19, 960, 64)`
- Edge features: `(B, 171, 960, 1)`
- Edge weights: `(B, 171, 960)` → adjacency `(B, 960, 19, 19)`
- GNN out: `(B, 19, 960, 64)`
- Bottleneck merged: `(B, 512, 960)` → Upsampled `(B, 19, 15360)` → Logits `(B, 15360)`

V2 vs V3

- V2 (deprecated): `TCN → BiMamba2(512) → Head`; optional heuristic dynamic graph (cosine + top‑k + threshold) and PyG
- V3 (recommended): `TCN → Node Mamba(64) + Edge Mamba(16) → Learned adjacency → Vectorized PyG + Laplacian PE (dynamic by default) → Head`

Note

- Current default in configs may still be `tcn` for backward compatibility in tests; deprecation warnings are emitted and the default will switch to `v3` in a subsequent release. Prefer setting `model.architecture: v3` now.

Adjacency specifics (V3)

- Metric: cosine (default) or correlation
- Top‑k per row (default 3), threshold prune (default 1e‑4), symmetrize, identity fallback for disconnected nodes
- Bypass edge transform inside GNN because weights are already Softplus’ed upstream

Code references

- Detector: `src/brain_brr/models/detector.py`
- Edge/Adjacency: `src/brain_brr/models/edge_features.py`
- GNN: `src/brain_brr/models/gnn_pyg.py`
- Mamba: `src/brain_brr/models/mamba.py`
- TCN/Head: `src/brain_brr/models/tcn.py`

Validated decisions (concise)

- k=3 sparsity (EvoBrain): exact match, no change needed.
- Time‑then‑graph ordering: temporal first (TCN+BiMamba), spatial next (GNN), vectorized for efficiency.
- Dynamic PE: enabled with stability guards; semi‑dynamic interval recommended on 24GB VRAM.
