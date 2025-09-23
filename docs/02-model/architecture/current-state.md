# Current Architecture State (Read First)

This summarizes the active runtime path vs historical design to avoid confusion during development and deployment.

## Current Reality (v2.3)
```
EEG → TCN Encoder → Bi‑Mamba → Projection → Upsample → Detection
```
- TCN replaced both U‑Net and ResCNN.
- Modal configs use `architecture: tcn` and this path is training now.

## Legacy Path (kept for ablations)
```
EEG → U‑Net Encoder → ResCNN → Bi‑Mamba → U‑Net Decoder → Detection
```

## Integration Point for Next Step (GNN)
Insert GNN after Bi‑Mamba at the bottleneck (before projection/upsample):

```
TCN → Bi‑Mamba → [GNN here] → Projection → Upsample → Detection
```

See also:
- docs/02-model/architecture/tcn-replacement.md (details and rationale)
- docs/04-research/future/CANONICAL-ROADMAP.md (status and next steps)

## GNN + LPE Status (v2.6 transition)
- PyTorch Geometric GNN with Laplacian PE is integrated and used when `graph.enabled=true`.
- Current adjacency source: heuristic dynamic builder (cosine/correlation + top‑k + threshold).
- Planned (not yet implemented): learned adjacency via an edge temporal stream (edge Mamba → Linear+Softplus → sparsify) as per EvoBrain.
- Channel order must remain canonical (19‑ch 10–20) when constructing graphs.
- Mamba CUDA kernels support conv kernel {2,3,4}; defaults coerce to 4 on CUDA.
