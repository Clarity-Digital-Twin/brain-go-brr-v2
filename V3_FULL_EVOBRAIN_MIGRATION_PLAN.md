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
    - Returns [(i,j) for i in range(n) for j in range(i+1, n)] for E=n*(n-1)/2 pairs
    - For n=19: returns 171 pairs indexed 0-170
  - `def edge_scalar_series(elec: torch.Tensor, *, metric: str='cosine') -> torch.Tensor`
    - Input: `(B, 19, T, 64)`; Output: `(B, E, T, 1)`
    - Implementation:
      ```python
      pairs = pair_indices_undirected(19)  # 171 pairs
      edge_feats = []
      for i, j in pairs:
          if metric == 'cosine':
              similarity = F.cosine_similarity(elec[:, i, :, :], elec[:, j, :, :], dim=-1)
          edge_feats.append(similarity.unsqueeze(-1))  # (B, T, 1)
      return torch.stack(edge_feats, dim=1)  # (B, E, T, 1)
      ```
  - `def assemble_adjacency(edge_weights: torch.Tensor, *, n_nodes: int=19, top_k: int=3, threshold: float=1e-4, symmetric: bool=True, identity_fallback: bool=True) -> torch.Tensor`
    - Input: `(B, E, T)`; Output: `(B, T, 19, 19)`
    - Build symmetric adjacency from edge weights, apply top-k then threshold per row

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
    - Buffer: `static_pe` `(19, k)` computed once from 10–20 structural graph:
      ```python
      # In __init__:
      structural_adj = self._get_structural_adjacency()  # 10-20 montage adjacency
      edge_index = (structural_adj > 0).nonzero(as_tuple=False).t()
      data = Data(x=torch.randn(19, 1), edge_index=edge_index)
      pe_transform = AddLaplacianEigenvectorPE(k=k_eigenvectors)
      data = pe_transform(data)
      self.register_buffer('static_pe', data.laplacian_eigenvector_pe)  # (19, k)
      ```
    - Vectorized forward when `use_vectorized`:
      ```python
      # Flatten B×T graphs into disjoint batch
      batch_size, n_nodes, seq_len, feat_dim = x.shape
      x_flat = x.permute(0, 2, 1, 3).reshape(-1, n_nodes, feat_dim)  # (B*T, N, D)

      # Build disjoint edge_index with offset for each graph
      edge_list = []
      for t in range(batch_size * seq_len):
          offset = t * n_nodes
          edges = edge_index + offset  # Offset node indices
          edge_list.append(edges)
      edge_index_batch = torch.cat(edge_list, dim=1)

      # Add static PE
      pe_broadcast = self.static_pe.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
      x_with_pe = torch.cat([x_flat, pe_broadcast], dim=-1)

      # Single GNN forward
      out = self.gnn(x_with_pe, edge_index_batch)

      # Reshape back
      out = out.reshape(batch_size, seq_len, n_nodes, -1).permute(0, 2, 1, 3)
      ```

4) Config/schema updates
- File: `src/brain_brr/config/schemas.py`
  - Add new fields:
    ```python
    class GraphConfig(BaseConfig):
        # New v3 fields
        edge_features: Literal['cosine', 'correlation'] = 'cosine'
        edge_top_k: int = 3
        edge_threshold: float = 1e-4
        edge_temperature: float = 0.1
        use_vectorized: bool = True  # Enable vectorized GNN path
        use_static_pe: bool = True   # Use static Laplacian PE

        # Existing fields (keep for compatibility)
        k_eigenvectors: int = 16
        alpha: float = 0.05
        n_layers: int = 2
        dropout: float = 0.1
        use_residual: bool = True

        # Deprecated (map to new names)
        similarity: Optional[str] = None  # -> edge_features
        top_k: Optional[int] = None       # -> edge_top_k
        threshold: Optional[float] = None # -> edge_threshold
        temperature: Optional[float] = None # -> edge_temperature

        def __post_init__(self):
            # Handle deprecated params
            if self.similarity is not None:
                warnings.warn(
                    f"'similarity' is deprecated, use 'edge_features' instead",
                    DeprecationWarning,
                    stacklevel=2
                )
                self.edge_features = self.similarity
            # Similar for top_k, threshold, temperature...
    ```
  - Add `architecture: Literal['v2', 'v3'] = 'v2'` to ModelConfig for gradual rollout.

## TDD Checklist (must pass)

Unit — edge pipeline (tests/unit/models/test_edge_features.py)
- `test_pair_indices_undirected()` → 171 pairs for N=19; verify (0,1), (0,2)... (17,18) ordering.
- `test_edge_scalar_series_cosine_shape()` → `(B=2,19,T=10,64)` → `(2,171,10,1)`; all finite.
- `test_edge_scalar_series_gradient_flow()` → Verify gradients flow through cosine similarity.
- `test_assemble_adjacency_topk_threshold_symmetry()` → top‑k per row then threshold; symmetric adjacency; identity fallback for empty rows; handles small T and all‑zeros.
- `test_assemble_adjacency_sparse_output()` → Verify sparsity level matches top_k * n_nodes.

Unit — GNN vectorization & PE (tests/unit/models/test_gnn_pyg_vectorized.py)
- `test_gnn_vectorized_preserves_shape()` for `(B=2,N=19,T=5,D=64)` and sparse symmetric adjacency.
- `test_gnn_static_pe_buffer_shape()` → buffer `(19,k)` on correct device; broadcast OK.
- `test_gnn_static_pe_computation()` → Verify PE computed from structural 10-20 graph.
- `test_gnn_bypass_edge_transform_flag()` avoids double transforms when upstream Softplus is used.
- `test_gnn_disjoint_batch_construction()` → Verify edge indices properly offset for each graph.
- `test_gnn_vectorized_vs_loop_equivalence()` → Output matches between vectorized and loop paths.

Unit — Mamba configurations (tests/unit/models/test_mamba_v3.py)
- `test_node_mamba_shape_preservation()` → `(B*19, 64, 960)` → `(B*19, 64, 960)`.
- `test_edge_mamba_single_channel()` → `(B*171, 1, 960)` → `(B*171, 1, 960)`.
- `test_mamba_gradient_flow()` → Verify gradients flow through both Mamba streams.

Integration — detector v3 (tests/integration/models/test_detector_v3.py)
- `test_v3_forward_no_nan_and_shape()` — `(B=2,19,15360)` → `(B,15360)` logits, no NaNs.
- `test_v3_graph_disabled_matches_temporal_only_shape()` sanity path when graph is disabled.
- `test_v3_memory_usage_vs_v2()` → Verify memory reduction vs current implementation.
- `test_v3_speed_improvement()` → Verify >10x speedup on forward pass.
- `test_v3_training_stability()` → 10 steps without NaN/inf with focal loss.

Performance benchmarks (tests/performance/test_v3_benchmarks.py)
- `test_forward_pass_speed()` → <1s for batch_size=12 on GPU (was 30-40s).
- `test_memory_efficiency()` → <8GB VRAM for batch_size=12 (current: 12-20GB).
- `test_gradient_computation_speed()` → Full backward pass <2s.

## Defaults (match EvoBrain + our constraints)
- Node Mamba: `d_model=64`, `n_layers=6`, `d_state=16`, `d_conv=4`, `expand=2` (using Mamba2).
- Edge Mamba: `d_model=1`, `n_layers=2`, `d_state=8`, `d_conv=4`, `expand=2` (using Mamba2).
- GNN: `SSGConv` with `alpha=0.05`, `K=2`, `n_layers=2`, `k_eigenvectors=16`.
- Adjacency: `edge_top_k=3`, `edge_threshold=1e-4`, symmetric with identity fallback.
- Laplacian PE: static buffer default; `dynamic_pe=False` for performance.
- Note: We use Mamba2 (improved) vs EvoBrain's Mamba1 for better performance.

## Environment/Setup Notes
- PyG must match Torch/CUDA; install via data.pyg.org wheels. Local 4090: AMP off initially, gradient_clip 0.5–1.0; WSL2 `num_workers=0`. A100: enable AMP once stable; larger batches amortize overhead.
- Mamba: use `conv_kernel=4` (CUDA supports {2,3,4}); set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d fallback for debug.
- Channel order must match canonical montage in `src/brain_brr/constants.py` when mapping edge pairs.

## Rollout Strategy (with xfail progression)

Phase 1: Edge pipeline foundation
1) Implement `edge_features.py` with pair_indices, edge_scalar_series, assemble_adjacency.
2) Add unit tests (all pass immediately).
3) Verify edge features are differentiable and sparse adjacency is valid.

Phase 2: Detector V3 scaffold
1) Create `detector_v3.py` with dual-stream architecture.
2) Add integration tests with `@pytest.mark.xfail(reason="V3 not complete")` initially.
3) Wire up node/edge Mamba streams, verify shapes.

Phase 3: GNN vectorization
1) Extend `gnn_pyg.py` with vectorized path behind `use_vectorized` flag.
2) Add static PE buffer computation and caching.
3) Remove xfail from integration tests as features complete.
4) Benchmark memory and speed improvements.

Phase 4: Config and deployment
1) Update configs with `architecture: v3` option (default: v2 for safety).
2) Add smoke test with v3 architecture on small data.
3) Run full training for 10 epochs to verify stability.
4) Enable v3 as default after validation.

Phase 5: Cleanup
1) Remove heuristic `graph_builder.py` after v3 proven stable.
2) Remove old GNN per-timestep loop path.
3) Update documentation with v3 as canonical architecture.

## Go/No‑Go Criteria (must ALL pass before v3 becomes default)

✅ Functional correctness
- All unit tests pass (edge features, GNN vectorization, Mamba configs).
- All integration tests pass (v3 forward, memory, speed).
- V3 output shape matches V2: `(B, 15360)` for 60s windows.
- No NaN/inf in 100 forward passes with random input.

✅ Performance improvements
- Forward pass: <1s for batch_size=12 (current: 30-40s) - **>30x speedup required**.
- Memory usage: <8GB VRAM for batch_size=12 (current: 12-20GB) - **>33% reduction required**.
- Training step: <5s total (forward + backward + optimizer).

✅ Training stability
- 10 epochs on smoke dataset without NaN/inf (focal loss, balanced sampling).
- Loss decreases monotonically for first 5 epochs.
- Seizure/non-seizure samples both contribute to gradients.
- Learning rate 1e-4 stable (no gradient explosion).

✅ Compatibility
- Config flag `architecture: v2/v3` works correctly.
- Can switch between v2/v3 mid-training via checkpoint.
- V3 can load v2 encoder weights (TCN unchanged).

⛔ Blockers (any failure = no-go)
- NaN/inf in any forward or backward pass.
- Memory usage higher than v2.
- Speed improvement <10x.
- Cannot reproduce v2 metrics within 5% on validation set.

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

