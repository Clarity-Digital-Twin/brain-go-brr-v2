# V3 Architecture (Ground Truth)

Canonical reference: `V3_ARCHITECTURE_AS_IMPLEMENTED.md`.

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

- Degree clamp, diagonal regularization, NaN/Inf checks with cached PE fallback are implemented in `gnn_pyg.py` to prevent non‑finite logits.

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
