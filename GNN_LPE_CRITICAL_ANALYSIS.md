# 🔴 CRITICAL: GNN + Laplacian PE — Audit, Findings, and Fix Plan

## Executive Summary
- Conceptually correct: time-then-graph pipeline matches EvoBrain’s flow (temporal model first, graph per timestep).
- Practically slow: current PyG path constructs thousands of small graphs in Python and recomputes Laplacian PE inside nested loops, causing severe CPU bottlenecks and low GPU utilization.
- Fixable now: we can vectorize across timesteps and precompute a static PE buffer (per 10–20 montage) to remove eigendecomposition from the hot path. Learned adjacency via an edge temporal stream remains a planned addition.

## Ground-Truth, With File References
- Detector integrates GNN after temporal modeling (time-then-graph): `src/brain_brr/models/detector.py:88` and `:130–162`.
- Adjacency is heuristic (cosine/correlation + top-k + threshold): `src/brain_brr/models/graph_builder.py:34–90`.
- PyG GNN forward is per-timestep and per-sample, building `Data` objects in Python:
  - Timestep loop: `src/brain_brr/models/gnn_pyg.py:103`.
  - Inner batch loop + Data creation: `src/brain_brr/models/gnn_pyg.py:110–141`.
  - Laplacian PE recomputed for each `Data`: `src/brain_brr/models/gnn_pyg.py:130–138`.

## Problems Identified (what’s expensive)
- Nested Python loops over T and B create thousands of `Data` objects per forward; overhead dominates.
- `AddLaplacianEigenvectorPE` runs inside the loop, doing an eigendecomposition per graph per timestep. With 19 nodes it’s small, but the repetition + object churn is the killer.
- All graph operations run on CPU; GPU sits idle waiting for preprocessing.

Note on “architecture order”: our order (TCN→Bi‑Mamba→GNN→projection) is fine and matches the intended time‑then‑graph paradigm. The issue is not the ordering, it’s the inefficient per‑timestep implementation.

## What EvoBrain Does vs. Us
- EvoBrain: separate SNNs for nodes and edges (Mamba/GRU), learned adjacency from the edge stream, then GNN with PE per timestep.
- Us (now): Mamba for nodes, heuristic adjacency (no edge stream yet), GNN with PE per timestep implemented via Python loops and transform calls.

We are missing the edge temporal stream (learned adjacency). That’s a separate, still‑planned improvement.

## Fix Plan (non-breaking, incremental)
Short-term goal: keep current functionality, remove the bottlenecks.

1) Vectorize across timesteps and batch
- Flatten `(B, N, T, D)` → `(B*T, N, D)` and `(B*T, N, N)`.
- Build a single disjoint super-graph for all `(B*T)` graphs:
  - Compute `edge_index` by taking `nonzero` over `(B*T, N, N)` and offset node indices with `g*N`.
  - Concatenate node features to shape `(B*T*N, D)` and run SSGConv layers once.
- Reshape back to `(B, N, T, D)`.

2) Replace dynamic PE transform with a static PE buffer (default)
- Compute a fixed Laplacian PE once for the canonical 10–20 topology (unweighted, undirected base graph) and register as a buffer of shape `(N, k)`.
- Broadcast and concatenate to node features at every forward; this preserves spatial/positional structure at negligible cost.
- Keep a feature flag for “dynamic_pe” to enable recomputation later when we have a fast batched eigen path (off by default).

3) Keep edge-weight transform, but make it optional
- Current code applies `Linear+Softplus` to edge weights. Heuristic builder already softmaxes; we should allow bypassing the extra transform via a flag (default: keep existing behavior for BC; later set to bypass when learned weights are already Softplus’ed upstream).

4) Prepare for learned adjacency (next PR)
- Add edge temporal stream (Bi‑Mamba2) to produce edge weights across time from per-edge features; assemble adjacency with top‑k/threshold/symmetry.
- Detector swaps out `graph_builder` for the learned adjacency path.

## TDD Outline (what to test)
Unit (fast):
- Shape invariants: `(B, N, T, D)` in → `(B, N, T, D)` out for typical sizes (B=2,N=19,T in {10, 64, 960}).
- Vectorized batching correctness: constructing a disjoint super‑graph from `(B*T, N, N)` via `nonzero` yields the same number of edges as summing per‑graph edges; guarantees symmetry when adjacency is symmetric.
- Static PE buffer: present, correct dtype/device, shape `(N, k)`, broadcast to `(B*T, N, k)`.
- Optional edge transform: when bypass flag is set, ensure no second transform is applied.

Integration (PyG enabled):
- End‑to‑end forward (detector with GNN enabled): no NaNs; output shape correct; laplacian path no longer called in hot loop.
- Performance regression guard (soft): log forward time over a tiny batch (marker `performance`), assert it doesn’t exceed a loose bound on CI CPU; keep lenient to avoid flakiness.

## Implementation Sketch (safe changes)
- Add flags to `GraphChannelMixerPyG.__init__`:
  - `use_vectorized: bool = True`
  - `use_dynamic_pe: bool = False`
  - `bypass_edge_transform: bool = False`
- Register `self.static_pe: (N, k)` buffer built once from a canonical 10–20 structural graph (unweighted). On failure, fall back to zeros.
- New forward path when `use_vectorized` is True:
  - Build `edge_index` and `edge_weight` for all `(B*T)` graphs in one pass; run SSGConv stack once; reshape.
  - Concatenate `static_pe` (broadcasted) to the first layer input.
- Keep current loop-based path behind the flag for BC while migrating tests.

Trade‑off note: dynamic PE per timestep is theoretically cleaner but in practice too slow with current tooling. Static PE captures node identity/geometry and is a common/accepted approximation for spectral PEs in dynamic‑edge settings.

## Configuration Guidance (until code lands)
- For long runs today: if training time is critical, set `graph.enabled: false` until vectorized path is merged. Otherwise reduce `k_eigenvectors` to 8 during experimentation.
- Local (4090): keep `mixed_precision: false`, `gradient_clip: 0.5–1.0`, `num_workers: 0` if WSL2; otherwise `2–4` on Linux.
- Modal (A100): `mixed_precision: true` is fine once NaNs are controlled; keep `gradient_clip: 0.5–1.0`; large batch sizes help amortize fixed costs.

## Roadmap Alignment
1) P0: Vectorized GNN + static PE buffer (this PR series).
2) P1: Edge temporal stream (learned adjacency) and detector wiring.
3) P2: Optional dynamic PE mode backed by a batched/GPU eigen implementation; keep default static.

## Conclusion
- The current GNN+LPE integration is conceptually aligned but computationally inefficient due to Python loops and repeated PE transforms.
- We can fix it without altering outputs by vectorizing across timesteps and switching to a static PE buffer by default.
- Learned adjacency via an edge stream is the next step toward EvoBrain parity once the core path is performant and test‑covered.
