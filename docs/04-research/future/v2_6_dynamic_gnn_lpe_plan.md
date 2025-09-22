# v2.6 Dynamic GNN + Laplacian PE (Implementation Plan)

This plan extracts the actionable pieces from EvoBrain to add a Dynamic GNN stage with Laplacian positional encoding after Bi‑Mamba in the TCN path.

## Target Architecture
```
EEG → TCN Encoder → Bi‑Mamba → [Dynamic GNN + LPE] → Projection → TCN Decoder → Detection
```

## Core Components
1) DynamicGraphBuilder (time‑varying adjacency per timestep)
2) GraphChannelMixer (GNN; start pure‑Torch, optionally PyG/SSGConv)
3) Laplacian PE (k=16) when PyG is enabled

## Integration Points
- Insert after Bi‑Mamba, before projection/upsample.
- Gate with `graph.enabled` in config.

## Minimal API and Shapes
- Input to graph: `(B, 19, T, D)` electrode features at bottleneck
- Adjacency: `(B, T, 19, 19)`
- Output: `(B, 19, T, D)` (same shape)

## Phases
Phase 1 (pure‑Torch MVP)
- Cosine similarity adjacency + top‑k + threshold
- Simple normalized A times features with residual
- Unit/integration tests; no PyG dependency

Phase 2 (PyG optional)
- SSGConv(alpha=0.05), `AddLaplacianEigenvectorPE(k=16)`
- Optional extras group `graph` in `pyproject.toml`

Phase 3 (tests/ablation)
- Static vs dynamic, with/without LPE, k variations

## References
- docs/04-research/future/gnn-tcn-stack.md
- EvoBrain code snippets (SSGConv, Laplacian PE)

Note: Keep CI green by guarding imports and keeping graph disabled by default.

