# V3 Architecture: As Implemented (Ground Truth)

This document describes the actual V3 model pipeline executed when `model.architecture: v3` is set. It is aligned to the current codebase and references concrete modules and shapes.

## High-Level Flow

```
Input EEG → TCN Encoder → Dual-Stream (Node/Edge) → GNN → Back-Projection → Upsample → Detection
                         ├─ Node Stream (BiMamba2)
                         └─ Edge Stream (BiMamba2 → learned adjacency)
```

## Tensor Shapes (Summary)

- Input: `(B, 19, 15360)`
- TCN out: `(B, 512, 960)`
- Electrode features: `(B, 19, 960, 64)`
- Node stream out: `(B, 19, 960, 64)`
- Edge features: `(B, 171, 960, 1)`
- Edge weights: `(B, 171, 960)`
- Adjacency: `(B, 960, 19, 19)`
- GNN out: `(B, 19, 960, 64)`
- Bottleneck merged: `(B, 512, 960)`
- Upsampled: `(B, 19, 15360)`
- Output logits: `(B, 15360)`

## Detailed Pipeline

### 1) Input

- Raw EEG windows are 60s at 256 Hz in the 10-20 montage: `(B, 19, 15360)`.

### 2) TCN Encoder — `src/brain_brr/models/tcn.py`

- Module: `TCNEncoder`
- Config: 8 layers, channels `[64, 128, 256, 512]`, kernel size 7, `stride_down=16`.
- Output: `(B, 512, 960)` after downsampling 16×.
- Reference: `src/brain_brr/models/tcn.py:99` (class), `src/brain_brr/models/tcn.py:118` (init), `src/brain_brr/models/tcn.py:141` (forward).

### 3) Dual-Stream (V3) — `src/brain_brr/models/detector.py`

The V3 path is gated by `SeizureDetector.architecture == "v3"` and the presence of V3 components.

Reference: `src/brain_brr/models/detector.py:171` (V3 forward branch).

#### 3.1) Electrode Projection

- `proj_to_electrodes`: `Conv1d(512 → 19*64)` applied to `(B, 512, 960)` gives `(B, 19, 960, 64)` after reshape/permute.
- Reference: `src/brain_brr/models/detector.py:188`.

#### 3.2) Node Stream (per‑electrode temporal modeling)

- `node_mamba`: `BiMamba2(d_model=64, num_layers=6, d_state=16, d_conv=4, expand=2, headdim=8)`.
- Reshape `(B, 19, 960, 64)` → `(B*19, 64, 960)` and ensure contiguous before Mamba.
- Output reshaped back to `(B, 19, 960, 64)`.
- References: `src/brain_brr/models/detector.py:193-201`, constructor in `src/brain_brr/models/detector.py:296-306`.

#### 3.3) Edge Stream (learned adjacency)

1) Edge features — `edge_scalar_series`
   - Metric: `cosine` (default) or `correlation`.
   - Input `(B, 19, 960, 64)` → `(B, 171, 960, 1)` (171 = 19·18/2 pairs).
   - Reference: `src/brain_brr/models/edge_features.py:39`.

2) Temporal edge processing — BiMamba2 over edges
   - Learned lift: `edge_in_proj: Conv1d(1 → D)` with `D = edge_mamba_d_model` (default 16).
   - `edge_mamba`: `BiMamba2(d_model=D, num_layers=edge_mamba_layers (default 2), d_state=edge_mamba_d_state (default 8), d_conv=4, expand=2, headdim=4)`.
   - Projection back: `edge_out_proj: Conv1d(D → 1)` then `Softplus` to ensure non‑negative weights.
   - Output: `(B, 171, 960)`.
   - References: `src/brain_brr/models/detector.py:206-217`, config assertions at `src/brain_brr/models/detector.py:313-321`.

3) Adjacency assembly — `assemble_adjacency`
   - Map edge weights to `adj ∈ (B, 960, 19, 19)`.
   - Top‑k sparsification per row (`edge_top_k`, default 3), threshold pruning (`edge_threshold`, default 1e‑4).
   - Symmetrize by averaging and apply identity fallback for disconnected nodes.
   - Reference: `src/brain_brr/models/edge_features.py:86`.

### 4) GNN with Laplacian PE — `src/brain_brr/models/gnn_pyg.py`

- Module: `GraphChannelMixerPyG(use_vectorized=True, use_dynamic_pe=False, bypass_edge_transform=True)`.
- Vectorized processing: flatten to a single disjoint batch of `B*960` graphs and process in one pass.
- Static Laplacian PE: computed once from structural 10‑20 adjacency and concatenated to node features in the first layer.
- Architecture: 2× `SSGConv` (α=0.05), LayerNorm, dropout, residuals on layers > 1.
- Input `(B, 19, 960, 64)`, adjacency `(B, 960, 19, 19)` → output `(B, 19, 960, 64)`.
- References: class at `src/brain_brr/models/gnn_pyg.py:23`, vectorized forward at `src/brain_brr/models/gnn_pyg.py:94`, static PE at `src/brain_brr/models/gnn_pyg.py:94-120`.

### 5) Back‑Projection, Upsample, Detection — `src/brain_brr/models/detector.py` and `src/brain_brr/models/tcn.py`

- Back‑projection to bottleneck: `proj_from_electrodes: Conv1d(19*64 → 512)` → `(B, 512, 960)`.
- Projection head: `ProjectionHead(512 → 19)` then nearest‑neighbor upsample `960 → 15360`.
- Detection head: `Conv1d(19 → 1)` → output logits `(B, 15360)`.
- References: back‑projection `src/brain_brr/models/detector.py:232-236`, head `src/brain_brr/models/tcn.py:160-189`, final conv `src/brain_brr/models/detector.py:268-271`.

## Configuration (relevant keys)

YAML under `model:`

```yaml
architecture: v3

graph:
  enabled: true
  edge_features: cosine           # or correlation
  edge_top_k: 3                   # sparsity per node
  edge_threshold: 1.0e-4          # pruning cutoff
  edge_mamba_layers: 2            # temporal edge stack
  edge_mamba_d_state: 8           # SSM state dim
  edge_mamba_d_model: 16          # multiple of 8 required

  n_layers: 2                     # GNN layers
  dropout: 0.1
  use_residual: true
  alpha: 0.05                     # SSGConv mixing
  k_eigenvectors: 16              # Laplacian PE dim
```

Notes:

- Node stream is fixed at `d_model=64`, `num_layers=6`, `headdim=8`.
- Edge stream dimensions (`D`) are configurable via `edge_mamba_d_model` (default 16); assertions enforce `D % 8 == 0`.

## BiMamba2 Details — `src/brain_brr/models/mamba.py`

- Each bidirectional layer validates `((d_model * expand) / headdim)` is an integer and warns if not multiple of 8.
- CUDA kernels are used when `mamba-ssm` is available; otherwise a depthwise `Conv1d` fallback is used (primarily for CPU/tests).
- V3 fixes the common CUDA‑alignment issue by choosing `headdim=8` for node (`(64*2)/8=16`) and `headdim=4` for edge when `D=16` (`(16*2)/4=8`).
- Reference: class `BiMamba2` and `BiMamba2Layer` at `src/brain_brr/models/mamba.py:1`.

Clarification: The headdim choices eliminate kernel‑alignment fallbacks. Fallbacks can still occur if `mamba-ssm` is not available or is explicitly disabled via `SEIZURE_MAMBA_FORCE_FALLBACK=1`.

## GNN PE and Structural Graph

- Structural adjacency for 10‑20 montage is defined in `get_structural_adjacency` and used to compute static Laplacian PE once per run.
- References: `src/brain_brr/models/edge_features.py:172` (structural graph), PE computation in `src/brain_brr/models/gnn_pyg.py:94-120`.

## Behavior Compared to V2 (TCN path)

- V2 path: `TCN → BiMamba2(512) → Head`. Optional heuristic dynamic GNN can be enabled for V2 only.
- V3 path: replaces the global 512‑dim BiMamba2 with dual‑stream node/edge processing and learned adjacency + vectorized GNN.
- References: V2 branch in `forward` at `src/brain_brr/models/detector.py:238-241`; V2 heuristic builder only attached when `architecture != "v3"` (`src/brain_brr/models/detector.py:351-363`).

## Validation Status (unit tests present)

- Detector from config initializes with V3 components present.
- V3 forward shapes and NaN checks pass.
- V3 works with and without GNN enabled.
- V2 compatibility maintained.
- References: `tests/unit/models/test_detector_v3.py` and `tests/unit/models/test_gnn_pyg_vectorized.py`.

## Performance Characteristics (observed/expected)

- Memory: batch size 8 typically fits 12–20GB (RTX 4090); larger batches on A100.
- Complexity: O(N) in sequence length for TCN, both Mamba streams, and GNN per timestep (batched over all timesteps).

## Source Pointers

- Detector (V3 branch and construction): `src/brain_brr/models/detector.py:171`, `src/brain_brr/models/detector.py:291`.
- Edge features and adjacency: `src/brain_brr/models/edge_features.py:39`, `src/brain_brr/models/edge_features.py:86`.
- GNN (vectorized, PE): `src/brain_brr/models/gnn_pyg.py:23`, `src/brain_brr/models/gnn_pyg.py:94`.
- BiMamba2: `src/brain_brr/models/mamba.py:1`.
- TCN and ProjectionHead: `src/brain_brr/models/tcn.py:99`, `src/brain_brr/models/tcn.py:160`.

