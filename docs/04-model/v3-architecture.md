# V3 Architecture (Single Source of Truth)

Flow

- Input `(B,19,15360)` → TCN `(B,512,960)` → Electrode features `(B,19,960,64)`
- Node Mamba: BiMamba2 over `(B*19,64,960)` → `(B,19,960,64)`
- Edge features: cosine/correlation `(B,171,960,1)` → Edge Mamba (1→16→1 + Softplus) → weights `(B,171,960)`
- Assemble adjacency `(B,960,19,19)` with top‑k, threshold, symmetry, identity fallback
- Vectorized GNN (SSGConv×2 + Laplacian PE) over all timesteps → `(B,19,960,64)`
- Back‑project to `(B,512,960)` → ProjectionHead to `(B,19,15360)` → Conv1d(19→1) logits `(B,15360)`

Optional enhancement

- Lightweight time–frequency hybrid: add a 3‑band STFT side‑branch and fuse before `proj_to_electrodes`. See `docs/04-model/time-frequency-hybrid.md`.

Key parameters

- Node Mamba: d_model=64, n_layers=6, d_state=16, d_conv=4, expand=2, headdim=8
- Edge Mamba: d_model=16, n_layers=2, d_state=8, d_conv=4, expand=2, headdim=4; 1→16→1 Conv1d lift/proj; Softplus
- GNN: SSGConv×2, α=0.05, residuals, LayerNorm+Dropout; Laplacian PE (k=16) dynamic by default, static optional
- TCN: 8 layers, channels [64,128,256,512], kernel 7, stride_down 16

Stability notes (dynamic PE)

- Dynamic PE is enabled by default with safeguards (degree clamp, diagonal regularization, NaN/Inf checks, cached‑PE fallback; eigens in fp32).
- On consumer GPUs (e.g., RTX 4090), if you observe NaNs early in training, set `use_dynamic_pe: false` as a temporary fallback and file an issue with logs.
- Edge projection clamping is hardcoded in the V3 path (similarity clamp [-0.99, 0.99], edge projection clamp [-3, 3]). Legacy `BGB_EDGE_CLAMP*` envs have been removed.
- Finite checks: enable `BGB_DEBUG_FINITE=1` to activate `assert_finite` guards in critical tensors (see `src/brain_brr/models/debug_utils.py`). Use only when diagnosing NaNs.
- See `docs/04-model/laplacian-pe.md` for implementation details and configuration knobs, and `docs/08-operations/v3-nan-explosion-resolution.md` for incident context.

Constraints and guards

- CUDA alignment: `(d_model*expand)/headdim` must be integer and multiple of 8 (node headdim=8, edge headdim=4 satisfy this)
- V3 uses vectorized GNN; PE mode configurable (dynamic recommended). `bypass_edge_transform=True` (edge weights already Softplus’ed upstream)
- Identity fallback in adjacency prevents disconnected nodes

Where in code

- Detector (V3 branch): `src/brain_brr/models/detector.py`
- Edge features + adjacency: `src/brain_brr/models/edge_features.py`
- GNN (vectorized; PE configurable): `src/brain_brr/models/gnn_pyg.py`
- Mamba layers: `src/brain_brr/models/mamba.py`
- TCN encoder + head: `src/brain_brr/models/tcn.py`

Status and validated decisions

- Dynamic Laplacian PE: default ON with safeguards; fallback OFF on RTX 4090 if instability is observed.
- Graph sparsity: top‑k=3 validated by EvoBrain (“top‑3 neighbors kept”); threshold prune, symmetry, identity fallback for safety.
- Temporal → spatial order: time‑then‑graph validated by literature; vectorized over timesteps for efficiency.
- Node stream capacity: 64 dims (1216 total) sufficient; can ablate 128 later.
- Bidirectional SSM: BiMamba2 for 60s windows (offline); causal variant optional for streaming.
