# V3.0 — TCN + Full EvoBrain Dual‑Stream (Node + Edge) with GNN + LPE

Status: Proposed (ready to implement via TDD). This document aligns precisely with the current codebase (TCN + Bi‑Mamba + PyG GNN), the EvoBrain design (dual SNN streams + learned adjacency), and our constraints (19 electrodes, 60 s @ 256 Hz → 15360 samples).

## Canonical Shapes and Flow (exact)

- Input window: `(B, 19, 15360)` — 60 seconds at 256 Hz
- TCN encoder (`src/brain_brr/models/tcn.py`): `(B, 512, 15360) → (B, 512, 960)` with `stride_down=16`
- Project to electrode features: `Conv1d(512 → 19*64)`: `(B, 19*64, 960)` → reshape `→ (B, 19, 960, 64)`

Dual‑stream EvoBrain backend:

1) Node temporal stream (per electrode)
- Reshape to `(B*19, 64, 960)` → `BiMamba2(d_model=64, n_layers=6)` → `(B*19, 64, 960)` → reshape back `(B, 19, 960, 64)`.

2) Edge temporal stream (learned adjacency)
- Edge scalar feature per pair and timestep: for each `(i,j)`, compute a scalar series over time from electrode features (default: cosine across the 64‑d feature vectors at each `t`).
- Output shape: `(B, E=171, 960, 1)` (undirected upper‑triangle without self‑edges).
- Reshape to `(B*E, 1, 960)` → `BiMamba2(d_model=1, n_layers=2)` → `(B*E, 1, 960)` → reshape back `(B, E, 960, 1)`.
- Edge→weight head: `Linear(1→1) + Softplus` → `(B, E, 960)` non‑negative weights.
- Assemble adjacency per timestep: map edges to `(B, 960, 19, 19)`, symmetrize, top‑k per row (default 3), threshold (default 1e‑4), identity fallback for empty rows.

3) GNN + Laplacian PE (PyG, vectorized across time)
- Vectorize across time: flatten `(B, 19, 960, 64)` → `(B*960, 19, 64)` and adjacency `(B*960, 19, 19)`.
- Build a single disjoint `Batch` for all graphs; run SSGConv layers once per forward; reshape back `(B, 19, 960, 64)`.
- Laplacian PE: default to a static PE buffer computed once from the canonical 10–20 structural graph (unweighted, undirected), shape `(19, k=16)`, broadcast to `(B*960, 19, 16)`. Keep a `dynamic_pe` feature flag for research; default off for performance.

4) Back‑projection and detection (unchanged)
- Permute/reshape `(B, 19, 960, 64)` → `(B, 19*64, 960)` → `Conv1d(19*64→512)` → `(B, 512, 960)`.
- ProjectionHead upsamples `(B, 512, 960)` → `(B, 19, 15360)` and detection head produces `(B, 15360)` logits.

Notes:
- All time lengths are 960 after the TCN stride‑down; prior docs that used 60 timesteps were incorrect.
- Our BiMamba2 expects `(B, C, L)`; for nodes we batch electrodes, for edges we batch edges.

## Precisely What’s Different vs. V2.x

- Replace heuristic adjacency (`graph_builder.py`) with learned adjacency from the edge stream (Mamba + Linear + Softplus + top‑k/threshold/symmetry + identity fallback).
- Refactor PyG GNN to a vectorized forward over all timesteps and use a static Laplacian PE buffer by default.
- Keep the “time‑then‑graph” order as today; do not insert extra temporal blocks after GNN.

## Implementation Plan (files and signatures)

1) Edge features and adjacency assembly
- File: `src/brain_brr/models/edge_features.py`
  - `def pair_indices_undirected(n: int) -> list[tuple[int,int]]` — upper‑triangle index list (cached).
  - `def edge_scalar_series(elec: torch.Tensor, *, metric: str='cosine') -> torch.Tensor` → `(B,E,T,1)` from `(B,19,T,64)`.
  - `def assemble_adjacency(edge_weights: torch.Tensor, *, n_nodes: int=19, top_k: int=3, threshold: float=1e-4, symmetric: bool=True, identity_fallback: bool=True) -> torch.Tensor` → `(B,T,19,19)`.

2) Dual streams in detector (new v3 class; v2 remains intact)
- File: `src/brain_brr/models/detector_v3.py`
  - `class SeizureDetectorV3(nn.Module)`
  - Members: `tcn_encoder`, `proj_to_electrodes`, `node_mamba: BiMamba2(d_model=64,n_layers=6)`, `edge_mamba: BiMamba2(d_model=1,n_layers=2)`, `edge_head = nn.Sequential(nn.Linear(1,1), nn.Softplus())`, `gnn: GraphChannelMixerPyG(d_model=64, k_eigenvectors=16, alpha=0.05, K=2, n_layers=2)`, `proj_from_electrodes: Conv1d(19*64→512)`, `proj_head`, `detection_head`.
  - `forward(x: torch.Tensor) -> torch.Tensor`
    - TCN → `(B,512,960)`; to electrodes → `(B,19,960,64)`.
    - Node stream: `(B*19,64,960)` → Mamba → `(B,19,960,64)`.
    - Edge features: `(B,E,960,1)`; `(B*E,1,960)` → Mamba → `(B,E,960,1)` → head → `(B,E,960)`.
    - Adjacency: `(B,960,19,19)`.
    - GNN (vectorized across time): `(B,19,960,64)` + adjacency → `(B,19,960,64)`.
    - Back‑projection and detection (as v2): logits `(B,15360)`.

3) PyG GNN vectorization + static PE
- File: `src/brain_brr/models/gnn_pyg.py` (extend existing `GraphChannelMixerPyG`)
  - Add buffers/flags in `__init__`: `self.static_pe` (19×k), `use_vectorized=True`, `use_dynamic_pe=False`, `bypass_edge_transform=False`.
  - Build `static_pe` once from canonical 10–20 structural graph; fallback to zeros if needed.
  - New forward path when `use_vectorized` is True: flatten `B*T`, build disjoint super‑graph, broadcast static PE to first layer, SSGConv stack once, reshape back.
  - Keep the loop path behind a flag for BC during the migration.

4) Config/schema additions
- File: `src/brain_brr/config/schemas.py`
  - Add `graph.edge_features: Literal['cosine','correlation'] = 'cosine'`
  - Add `graph.edge_top_k: int = 3`, `graph.edge_threshold: float = 1e-4`, `graph.edge_temperature: float = 0.1`.
  - Keep `graph.k_eigenvectors`, `graph.alpha`, `graph.n_layers`, `graph.dropout`, `graph.use_residual`.
  - Soft‑deprecate `similarity/top_k/threshold/temperature` fields by mapping to the `edge_*` names with a warning.

## TDD Checklist

Unit — edge pipeline
- `test_pair_indices_undirected()` returns `E=171` pairs for `N=19` and correct symmetry mapping.
- `test_edge_scalar_series_cosine_shape()` for `(B=2,19,T=10,64)` → `(2,171,10,1)`; all finite.
- `test_assemble_adjacency_topk_threshold_symmetry()`
  - Top‑k applied per row, then threshold; adjacency symmetric; identity fallback fills empty rows on extreme sparsity.

Unit — GNN vectorization
- `test_gnn_vectorized_preserves_shape()` for `(B=2,N=19,T=5,D=64)` and sparse symmetric adjacency.
- `test_gnn_static_pe_buffer_shape_and_dtype()` ensures `(19,k)` buffer on correct device, broadcast works.
- `test_gnn_bypass_edge_transform_flag()` ensures no double transform when upstream edge weights already Softplus’ed.

Integration — detector v3
- `test_v3_forward_no_nan_and_shape()` for random input `(B=2,19,15360)` produces `(B,15360)` logits; no NaNs.
- `test_v3_graph_disabled_matches_v2_temporal_only_shape()` sanity check path parity when `graph.enabled=false`.

Performance (soft, non‑flaky)
- `test_v3_forward_is_vectorized_marker()` logs per‑batch forward time on CPU with a lenient bound; or gated under a marker to keep CI fast.

## Defaults and Hyperparameters
- Node Mamba: `d_model=64`, `n_layers=6`, `d_state=16`, `d_conv=4`.
- Edge Mamba: `d_model=1`, `n_layers=2`, `d_state=8`, `d_conv=4`.
- GNN: `SSGConv` with `alpha=0.05`, `K=2`, `n_layers=2`, `k_eigenvectors=16`.
- Adjacency: `edge_top_k=3`, `edge_threshold=1e-4`, symmetric with identity fallback.
- Laplacian PE: static buffer by default; `dynamic_pe=False` unless explicitly enabled for research.

## Notes vs. EvoBrain
- We follow EvoBrain’s dual SNN streams (node + edge) and learned adjacency.
- We retain our TCN front end (instead of EvoBrain’s preprocessing block) — empirically strong and already integrated.
- Applying GNN per timestep is default; for ablations a “last‑timestep only” option can be added later, but it’s not the canonical path.

## Rollout Strategy
1) Land edge features + adjacency assembly + tests.
2) Land detector_v3 with node/edge streams (using current GNN forward).
3) Land GNN vectorized/static‑PE path + tests; switch detector_v3 to `use_vectorized=True`.
4) Update configs (local + modal) to select `architecture: v3`.

This sequence keeps v2 stable while making v3 both correct and performant.

