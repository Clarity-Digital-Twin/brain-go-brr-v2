# V3 Architecture: As Implemented (Ground Truth)

> Archived note: Canonical and up‑to‑date architecture now lives at
> `docs/04-model/v3-architecture.md`. See `docs/ARCHIVE_MAPPING.md`.

## High‑Level Flow

```
Input EEG → TCN Encoder → Dual‑Stream (Node/Edge) → GNN → Back‑Projection → Upsample → Detection
                         ├─ Node Stream (BiMamba2)
                         └─ Edge Stream (BiMamba2 → learned adjacency)
```

## Core Shapes

- Input `(B,19,15360)` → TCN `(B,512,960)` → Electrodes `(B,19,960,64)`
- Node stream `(B,19,960,64)`
- Edge weights `(B,171,960)` → Adjacency `(B,960,19,19)`
- GNN out `(B,19,960,64)` → Merge `(B,512,960)` → Upsample `(B,19,15360)` → Logits `(B,15360)`

## Key Components and References

- TCN: `TCNEncoder` 8 layers, stride_down=16 → src/brain_brr/models/tcn.py:99
- V3 forward branch and construction → src/brain_brr/models/detector.py:171, src/brain_brr/models/detector.py:291
- Node stream: `BiMamba2(d_model=64, layers=6, headdim=8)` → src/brain_brr/models/detector.py:296
- Edge features: `edge_scalar_series` (cosine/correlation) → src/brain_brr/models/edge_features.py:39
- Edge stream: `Conv1d(1→D)` → `BiMamba2(d_model=D, layers=2, headdim=4)` → `Conv1d(D→1)` + Softplus → src/brain_brr/models/detector.py:308
- Adjacency assembly (top‑k, threshold, symmetrize, identity fallback) → src/brain_brr/models/edge_features.py:86
- GNN: `GraphChannelMixerPyG` vectorized, static Laplacian PE → src/brain_brr/models/gnn_pyg.py:23, src/brain_brr/models/gnn_pyg.py:94
- Projection head and detection → src/brain_brr/models/tcn.py:160, src/brain_brr/models/detector.py:268

## Configuration (relevant YAML)

```yaml
model:
  architecture: v3
  graph:
    enabled: true
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16   # must be multiple of 8
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
```

Notes
- Node stream fixed: `d_model=64`, `num_layers=6`, `headdim=8`.
- Edge `D=edge_mamba_d_model` is configurable (default 16) and must be multiple of 8.
- Mamba fallbacks only occur if `mamba-ssm` is unavailable or forced (`SEIZURE_MAMBA_FORCE_FALLBACK=1`).

For a comprehensive walk‑through (including complexity and test status), see `docs/04-model/v3-architecture.md`.
