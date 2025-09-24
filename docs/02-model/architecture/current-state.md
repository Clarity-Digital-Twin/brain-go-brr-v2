# Current Architecture State (Read First)

This summarizes the active runtime path vs historical design to avoid confusion during development and deployment.

## Current Reality (v2.x)
```
EEG → TCN Encoder → Bi‑Mamba → Projection → Upsample → Detection
```
- TCN replaced both U‑Net and ResCNN.
- Default configs use `architecture: tcn` and this path is training now.

## Legacy Path (kept for ablations)
```
EEG → U‑Net Encoder → ResCNN → Bi‑Mamba → U‑Net Decoder → Detection
```

## Integration (GNN)
GNN is inserted after Bi‑Mamba at the bottleneck (before projection/upsample):

```
TCN → Bi‑Mamba → [GNN here] → Projection → Upsample → Detection
```

See also:
- docs/02-model/architecture/tcn-replacement.md (details and rationale)
- docs/04-research/future/CANONICAL-ROADMAP.md (status and next steps)

## GNN + LPE Status (v3)
- PyTorch Geometric GNN with Laplacian PE is integrated and used when `graph.enabled=true`.
- Two paths exist:
  - v2 (default): heuristic adjacency via similarity + top‑k + threshold.
  - v3 (select `model.architecture: v3`): learned adjacency via an edge temporal stream (edge Bi‑Mamba → Linear+Softplus → sparsify) as per EvoBrain.
- v3 uses a vectorized PyG forward across time and a static Laplacian PE buffer by default.
- Channel order must remain canonical (19‑ch 10–20) when constructing graphs.
- Mamba CUDA kernels support conv kernel {2,3,4}; we set 4 on CUDA.
