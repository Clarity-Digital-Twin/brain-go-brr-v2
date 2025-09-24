# Project Overview

Mission: O(N) clinical seizure detection with TCN → Dual-Stream Mamba → Vectorized GNN → Post-processing.

- Input: 19-channel EEG (60s, 256Hz)
- Core: TCN encoder, bidirectional Mamba for nodes/edges, PyG GNN with static Laplacian PE
- Output: Per-sample seizure probabilities with clinical post-processing

Authoritative references

- V3 spec: `V3_ARCHITECTURE_AS_IMPLEMENTED.md`
- Code: `src/brain_brr/models/detector.py`, `src/brain_brr/models/gnn_pyg.py`, `src/brain_brr/models/edge_features.py`

Quick start

- Local smoke: `make s`
- Full local: `make train-local` (see `docs/05-training/local.md`)
