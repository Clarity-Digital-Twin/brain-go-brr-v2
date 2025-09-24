# Architecture Summary

Pipeline: Input EEG → TCN → Dual-Stream (Node Mamba, Edge Mamba) → Vectorized GNN (PyG) → Projection/Detection → Post-process.

Shapes

- TCN out: `(B, 512, 960)`
- Electrode feats: `(B, 19, 960, 64)`
- Edge weights: `(B, 171, 960)` → adjacency `(B, 960, 19, 19)`

Where implemented

- Detector: `src/brain_brr/models/detector.py`
- Edge/Adjacency: `src/brain_brr/models/edge_features.py`
- GNN: `src/brain_brr/models/gnn_pyg.py`
