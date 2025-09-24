# V3 Architecture (Ground Truth)

Canonical reference: `V3_ARCHITECTURE_AS_IMPLEMENTED.md`.

Components

- TCN encoder: `src/brain_brr/models/tcn.py`
- Dual-stream Mamba (node + edge): `src/brain_brr/models/detector.py`, `src/brain_brr/models/mamba.py`
- Edge features + adjacency: `src/brain_brr/models/edge_features.py`
- Vectorized GNN (PyG): `src/brain_brr/models/gnn_pyg.py`
- Projection/Head + Post-processing

Notes

- Node: d_model=64, layers=6, headdim=8
- Edge: d_model=16, layers=2, headdim=4 (Softplus, top-k, threshold, symmetrize)
