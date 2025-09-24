# V3.0 — TCN + Full EvoBrain Dual‑Stream with PyG GNN + Laplacian PE

Status: Proposed (ready to implement via TDD). This plan is aligned to the current codebase and the EvoBrain design, with precise shapes and file‑level tasks. It replaces heuristic adjacency with a learned edge stream and fixes GNN performance via vectorization + static PE.

## Executive Decision
Skip incremental. Implement the complete dual‑stream EvoBrain backend (node+edge) atop our solid TCN front end. No half measures.

## Canonical Flow and Shapes (exact)

- Input window: `(B, 19, 15360)` — 60 seconds at 256 Hz
- TCN encoder: `(B, 19, 15360) → (B, 512, 960)` (stride_down=16)
- Project to electrode features: `Conv1d(512→19*64)` → `(B, 19*64, 960)`
- Reshape for per‑electrode features: `(B, 19, 960, 64)`

Dual‑stream EvoBrain backend:

1) Node temporal stream (per electrode)
- Batch electrodes: `(B*19, 64, 960)` → `BiMamba2(d_model=64, n_layers=6)` → `(B*19, 64, 960)` → back to `(B, 19, 960, 64)`.

2) Edge temporal stream (learned adjacency)
- Edge scalar series per pair and timestep from electrode features — default metric: cosine (or correlation) on the 64‑d vectors at each t.
- Output: `(B, E=171, 960, 1)` for 19 undirected nodes.
- Batch edges: `(B*E, 1, 960)` → `BiMamba2(d_model=1, n_layers=2)` → `(B*E, 1, 960)`.
- Edge→weight head: `Linear(1→1)+Softplus` → `(B, E, 960)` non‑negative weights.
- Assemble adjacency per timestep: map edges to `(B, 960, 19, 19)`, symmetrize, top‑k per row (k=3), threshold (1e‑4), identity fallback for empty rows.

3) GNN + Laplacian PE (PyTorch Geometric, vectorized over time)
- Flatten across time: `(B, 19, 960, 64)` → `(B*960, 19, 64)`; adjacency `(B*960, 19, 19)`.
- Build one disjoint `Batch` for all graphs; run `SSGConv` stack once; reshape back to `(B, 19, 960, 64)`.
- Laplacian PE: static buffer computed once from the canonical 10–20 structural graph (unweighted, undirected), shape `(19, k=16)`, broadcast to first GNN layer. Keep a `dynamic_pe` flag off by default.

4) Back‑projection and detection
- Permute/reshape `(B, 19, 960, 64)` → `(B, 19*64, 960)` → `Conv1d(19*64→512)` → `(B, 512, 960)`.
- `ProjectionHead` upsamples `(B, 512, 960)` → `(B, 19, 15360)`; detection head outputs `(B, 15360)` logits.

Notes:
- Time length is 960 everywhere post‑TCN (earlier “60” was incorrect).
- Node Mamba preserves 64 features per electrode; projection to 512 happens after the GNN when mapping back to bottleneck space.
- “GNN last‑timestep only” can be an optional ablation flag, but the canonical path processes all timesteps (now vectorized).

## Files to Add/Change (surgical, with signatures)

1) Edge features and adjacency assembly
- File: `src/brain_brr/models/edge_features.py`
  - `def pair_indices_undirected(n: int) -> list[tuple[int,int]]`
  - `def edge_scalar_series(elec: torch.Tensor, *, metric: str='cosine') -> torch.Tensor`
    - Input: `(B, 19, T, 64)`; Output: `(B, E, T, 1)`.
  - `def assemble_adjacency(edge_weights: torch.Tensor, *, n_nodes: int=19, top_k: int=3, threshold: float=1e-4, symmetric: bool=True, identity_fallback: bool=True) -> torch.Tensor`
    - Input: `(B, E, T)`; Output: `(B, T, 19, 19)`.

2) New detector (v3)
- File: `src/brain_brr/models/detector_v3.py`
  - `class SeizureDetectorV3(nn.Module)`
  - Members: `tcn_encoder`, `proj_to_electrodes`, `node_mamba`, `edge_mamba`, `edge_head`, `gnn (GraphChannelMixerPyG)`, `proj_from_electrodes`, `proj_head`, `detection_head`.
  - `forward(x: torch.Tensor) -> torch.Tensor`
    - TCN → electrodes `(B,19,960,64)` → node stream `(B,19,960,64)`.
    - Edge stream `(B,E,960,1)` → weights `(B,E,960)` → adjacency `(B,960,19,19)`.
    - GNN (vectorized) on `(B,19,960,64)` with adjacency → back‑project → decode → `(B,15360)` logits.

3) PyG GNN vectorization + static PE
- File: `src/brain_brr/models/gnn_pyg.py`
  - Extend `GraphChannelMixerPyG` with:
    - Flags: `use_vectorized=True`, `use_dynamic_pe=False`, `bypass_edge_transform=False`.
    - Buffer: `static_pe` `(19, k)` computed once from 10–20 structural graph. Broadcast to `(B*T, 19, k)` at first layer.
    - Vectorized forward when `use_vectorized`: flatten `B*T`, build disjoint super‑graph, run `SSGConv` once, reshape back. Keep existing loop path behind a flag during transition.

4) Config/schema updates
- File: `src/brain_brr/config/schemas.py`
  - Add `graph.edge_features: Literal['cosine','correlation'] = 'cosine'`.
  - Add `graph.edge_top_k: int = 3`, `graph.edge_threshold: float = 1e-4`, `graph.edge_temperature: float = 0.1`.
  - Soft‑deprecate `similarity/top_k/threshold/temperature` by mapping to new names with a warning.
  - Keep `k_eigenvectors`, `alpha`, `n_layers`, `dropout`, `use_residual`. PyG is required for v3.

## TDD Checklist (must pass)

Unit — edge pipeline
- `test_pair_indices_undirected()` → 171 pairs for N=19; mapping indices correct.
- `test_edge_scalar_series_cosine_shape()` → `(B=2,19,T=10,64)` → `(2,171,10,1)`; all finite.
- `test_assemble_adjacency_topk_threshold_symmetry()` → top‑k per row then threshold; symmetric adjacency; identity fallback for empty rows; handles small T and all‑zeros.

Unit — GNN vectorization & PE
- `test_gnn_vectorized_preserves_shape()` for `(B=2,N=19,T=5,D=64)` and sparse symmetric adjacency.
- `test_gnn_static_pe_buffer_shape()` → buffer `(19,k)` on correct device; broadcast OK.
- `test_gnn_bypass_edge_transform_flag()` avoids double transforms when upstream Softplus is used.

Integration — detector v3
- `test_v3_forward_no_nan_and_shape()` — `(B=2,19,15360)` → `(B,15360)` logits, no NaNs.
- `test_v3_graph_disabled_matches_temporal_only_shape()` sanity path when graph is disabled.

Performance (soft marker)
- Guard that the vectorized path is used (no per‑timestep Data churn); optional timing bound on CPU behind a `performance` marker.

## Defaults (match EvoBrain + our constraints)
- Node Mamba: `d_model=64`, `n_layers=6`, `d_state=16`, `d_conv=4`.
- Edge Mamba: `d_model=1`, `n_layers=2`, `d_state=8`, `d_conv=4`.
- GNN: `SSGConv` with `alpha=0.05`, `K=2`, `n_layers=2`, `k_eigenvectors=16`.
- Adjacency: `edge_top_k=3`, `edge_threshold=1e-4`, symmetric with identity fallback.
- Laplacian PE: static buffer default; `dynamic_pe=False` for performance.

## Environment/Setup Notes
- PyG must match Torch/CUDA; install via data.pyg.org wheels. Local 4090: AMP off initially, gradient_clip 0.5–1.0; WSL2 `num_workers=0`. A100: enable AMP once stable; larger batches amortize overhead.
- Mamba: use `conv_kernel=4` (CUDA supports {2,3,4}); set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d fallback for debug.
- Channel order must match canonical montage in `src/brain_brr/constants.py` when mapping edge pairs.

## Rollout Strategy
1) Land edge features + adjacency assembly + tests.
2) Add detector_v3 with dual streams; integration test.
3) Vectorize PyG GNN and add static PE; switch v3 to `use_vectorized=True`; tests.
4) Update configs to select `architecture: v3` for local and modal.
5) Remove the heuristic `graph_builder` path once v3 is stable and tests migrated.

## Go/No‑Go Criteria
- All unit and integration tests pass; detector v3 produces finite outputs of correct shape.
- Vectorized GNN path verified (no inner Python loops over T/B; disjoint super‑graph batching).
- Training sanity on a small run without NaNs (AMP off initially on 4090).

## Appendix — Rationale vs. EvoBrain
- We follow EvoBrain’s dual SNN streams (node + edge) and learned adjacency.
- We retain our TCN front‑end (strong performance, already integrated) instead of EvoBrain’s exact preprocessing.
- Processing all timesteps is the canonical setting; “last‑timestep only” can be supported as an ablation flag.

---

## Source & Reference Index (for implementers)

- Our codebase
  - Detector wiring (time‑then‑graph): `src/brain_brr/models/detector.py:130–162`, factory at `:187–236`.
  - Bi‑Mamba2 (constraints, conv kernel): `src/brain_brr/models/mamba.py`.
  - PyG GNN (current slow path): `src/brain_brr/models/gnn_pyg.py:103` (timestep loop), `:110–141` (per‑batch Data), `:130–138` (PE recompute).
  - Heuristic adjacency builder (to be replaced): `src/brain_brr/models/graph_builder.py:34–90`.
  - Canonical montage (channel order): `src/brain_brr/constants.py` (list of 19 electrodes).
  - TCN and projection heads: `src/brain_brr/models/tcn.py`.

- Literature & reference repo
  - EvoBrain paper summary: `literature/markdown/EVOBRAIN.md/EVOBRAIN.md`.
  - EvoBrain reference args (top‑k, dynamic graph, node/edge SNNs): `reference_repos/EvoBrain-FBC5/args.py`.
  - EvoBrain model folder: `reference_repos/EvoBrain-FBC5/model/` (dual‑stream SNNs and GNN components).

## Current GNN+LPE Issues (Investigated) and V3 Fix

Problems observed (confirmed in code):
- Per‑timestep loop creates thousands of tiny PyG `Data` objects per forward: `src/brain_brr/models/gnn_pyg.py:103,110–141`.
- Laplacian PE recomputed inside the loop (repeated eigendecomp): `src/brain_brr/models/gnn_pyg.py:130–138`.
- All graph work runs on CPU; GPU under‑utilized.

V3 resolution (in this plan):
- Vectorized GNN forward over all `(B*T)` graphs via one disjoint super‑graph batch (no per‑timestep loops).
- Static Laplacian PE buffer `(19,k)` computed once from 10–20 topology and broadcast at forward; `dynamic_pe` kept as an off‑by‑default flag.
- Replace heuristic adjacency with learned adjacency from the edge stream (Bi‑Mamba + Linear+Softplus + top‑k/threshold/symmetry + identity fallback).

## Potential Blockers and Mitigations

- PyG install alignment
  - Ensure `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster` match Torch/CUDA; use wheels from `https://data.pyg.org` as we did locally.

- Mamba CUDA kernel constraint
  - Use `conv_kernel=4` (CUDA supports {2,3,4}); set env `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d fallback if needed for debug.

- WSL2 data loader behavior
  - Use `num_workers=0`, `pin_memory=false`, `persistent_workers=false` (already in local configs) to avoid hangs.

- Channel ordering for edges
  - Always map pairs using the canonical montage in `src/brain_brr/constants.py` to keep edge indexing consistent.

- Validation length vs training
  - Validation may have more windows than training (by design); logging added in `src/brain_brr/train/loop.py` prevents “hung” perception.

