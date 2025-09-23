# ✅ Pure GNN + Laplacian PE — Cleanup and Implementation Plan

Goal: implement a sequential‑then‑graph stack with learned adjacency (edge Mamba stream) and Laplacian PE. Remove heuristic adjacency builders (cosine/correlation) and reflect this across code and docs.

## Architecture (target)
- Node stream: TCN → Bi‑Mamba (B, 512, 960)
- Projection to electrodes: (B, 19, T, 64)
- Edge stream: per‑edge scalar features per timestep → Bi‑Mamba → Linear + Softplus → adjacency (B, T, 19, 19)
- GNN: PyG SSGConv (α=0.05) with Laplacian PE (k=16), applied per timestep
- Back‑projection: (B, 19×64, T) → 1×1 Conv to (B, 512, 960)

## Code cleanup (remove/replace)
- Remove heuristic graph builder usage (cosine/correlation). File can be deprecated and then deleted.
- Remove “pure torch” GNN as default; keep PyG+LPE as the supported backend for the pure GNN path.
- Update configs: deprecate `similarity`, `top_k`, `temperature`, `threshold` under `graph`. Replace with `edge_*` controls applied after edge stream.

## Additions (implement)
- Edge feature extractor: build per‑edge scalar features from electrode features each timestep.
- Edge temporal model: Bi‑Mamba over edges across time (shared across edges, batched as (B, E, T)).
- Edge→weight head: Linear + Softplus; apply sparsity (top‑k), thresholding, and symmetrization; guard against zero rows.
- Detector wiring: sequential (node) → projection → edge stream → adjacency → GNN+PE → back‑projection → detection.

## Config changes
- New: `graph.edge_features: {"cosine", "correlation", "coherence"}` (default: "cosine" as base scalar, not a final heuristic).
- New: `graph.edge_top_k`, `graph.edge_threshold`, `graph.edge_temperature` (applied post‑edge‑Mamba, not pre‑learned).
- New: `graph.reduce_edge: {"mamba","gru","lstm"}` (default: "mamba"); reuse `mamba` cfg.
- Keep: `graph.use_pyg=True`, `graph.k_eigenvectors=16`, `graph.alpha=0.05`, `graph.n_layers=2`.
- Deprecate: `graph.similarity`, `graph.top_k`, `graph.temperature`, `graph.threshold` (raise deprecation warning; map to new names).

## Tests (update/add)
- Remove tests that rely on `DynamicGraphBuilder` adjacency semantics.
- Add unit tests for edge extractor and adjacency assembly (shape, symmetry, sparsity, identity fallback).
- Integration: full detector forward with PyG+LPE + edge stream; gradient flow from output to input.
- Parameter parity: PyG vs pure torch no longer required; PyG is canonical for GNN+LPE.

## Docs (update now)
- AGENTS.md: reflect TCN + Bi‑Mamba + Dynamic GNN (PyG+LPE) and learned adjacency via edge Mamba; Mamba `conv_kernel=4`.
- README.md: add v2.6 design decisions (learned adjacency; PyG+LPE only).
- v2_6_dynamic_gnn_lpe_plan.md and v2_6_dynamic_gnn_lpe_CORRECTED.md: remove heuristic builder content; specify edge stream + learned adjacency.
- CLEANUP_DEBT.md: migration checklist for code removal and config deprecations.

## Why this (brief)
- Temporal edge dynamics matter; heuristics lose information.
- Laplacian PE stabilizes GNN across changing graphs.
- Pure learned adjacency keeps the architecture end‑to‑end trainable.
