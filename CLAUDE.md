# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🧠 Project Overview

Brain-Go-Brr v2: TCN + Bi-Mamba-2 + GNN for clinical EEG seizure detection — O(N) sequence modeling with bidirectional SSM and optional graph neural networks.

Why this is different:
- Transformers struggle on long EEG (O(N²) cost)
- Pure CNNs miss global temporal context
- Bidirectional Mamba-2 brings O(N) global context efficiently
- Optional GNN with Laplacian PE for spatial electrode relationships

## ⚡ Essential Commands

| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) — RUN AFTER EVERY CHANGE ✅ |
| `make t` | Fast tests (no coverage) |
| `make test` | Full tests with coverage |
| `make test-gpu` | GPU-specific tests |
| `make setup` | Initial setup (uv, hooks) |
| `make train-local` | Smoke test config (1 epoch, small batch) |
| `uv sync -E gpu` | GPU extra (Mamba-SSM) |
| `uv sync -E post,eval` | Extras: post-proc + eval |
| `python -m src train configs/smoke_test.yaml` | Direct training command |

## 🏗️ Architecture

| Component | Specification | Location |
|-----------|--------------|----------|
| Input | 19-channel EEG @ 256 Hz | - |
| TCN Encoder | Multi-scale temporal, ×16 downsample | `src/brain_brr/models/tcn.py` |
| Bi-Mamba-2 | 6 layers, d_model=512, d_state=16 | `src/brain_brr/models/mamba.py` |
| GNN (optional) | PyG SSGConv + Laplacian PE, α=0.05 | `src/brain_brr/models/gnn_pyg.py` |
| Graph Builder | Heuristic cosine/correlation adjacency | `src/brain_brr/models/graph_builder.py` |
| Hysteresis | tau_on=0.86, tau_off=0.78 | `src/brain_brr/post/postprocess.py` |
| Output | Per-timestep probabilities | - |

**Note**: Edge Mamba stream for learned adjacency NOT YET IMPLEMENTED (still using heuristic graph builder)

## 📁 Project Structure

```
src/brain_brr/      # Core modules
├── models/         # Neural network components
│   ├── detector.py # Main SeizureDetector class
│   ├── tcn.py     # TCN encoder (multi-scale temporal)
│   ├── mamba.py   # Bidirectional Mamba-2
│   ├── gnn_pyg.py # PyG GNN with Laplacian PE
│   └── graph_builder.py # Heuristic adjacency builder
├── data/          # EEG preprocessing
│   ├── loader.py  # EDF handling with MNE
│   └── dataset.py # PyTorch Dataset
├── train/         # Training pipeline
├── post/          # Post-processing
├── eval/          # Evaluation (TAES)
├── events/        # Event generation
└── config/        # Pydantic schemas

configs/           # YAML experiments
tests/             # Pytest suite
data/              # Datasets (git-ignored)
results/           # Outputs (git-ignored)
```

## Critical Implementation Details

### Channel Ordering
MUST maintain canonical 10-20 montage order (defined in `src/brain_brr/constants.py`):
```python
["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
 "Fz", "Cz", "Pz",
 "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
```

### Mamba CUDA Dispatch
- Configured d_conv=5, but CUDA kernels only support {2,3,4}
- Internally coerces to 4 for CUDA path
- Set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d fallback

### Post-Processing Pipeline
Hysteresis thresholds (tau_on=0.86, tau_off=0.78) → morphology → duration filtering → event generation

## 🎯 Clinical Targets (TAES)

- 10 FA/24h: >95% sensitivity (current SOTA: ~90%)
- 5 FA/24h: >90% sensitivity (current SOTA: ~85%)
- 1 FA/24h: >75% sensitivity (current SOTA: ~70%)

## 🔧 Development Rules

1. Quality first: run `make q` after EVERY change 🧹
2. Type everything: full type hints required
3. Tests required for new functions (unit/integration markers)
4. No comments unless explicitly requested
5. Follow neighboring file patterns; preserve `src/brain_brr/` APIs

Code style:
- Python 3.11+, 4-space indent, Ruff line length 100
- Imports: stdlib → third-party → first-party (sorted)

## 📊 Data Pipeline

1. Read EDF via MNE; 10-20 montage
2. Bandpass 0.5–120 Hz; 60 Hz notch
3. Resample to 256 Hz
4. Window 60s with 10s stride
5. Per-channel z-score normalization

Note: TUSZ may have malformed headers - fallback repair implemented. Channel synonyms handled (T7→T3, T8→T4, P7→T5, P8→T6).

## 🚀 Training Strategy

- Train: TUH EEG Seizure Corpus
- Validate: CHB-MIT
- Evaluate: epilepsybenchmarks.com
- No pretrained weights (novel architecture)

## ⚠️ Critical Notes

- Caching keys depend on config; edit config to invalidate cache
- WSL tip: `export UV_LINK_MODE=copy` (Makefile sets this by default) ⚙️
- CI uses `uv sync` (no extras) to avoid GPU builds on non-CUDA runners
- Use `num_workers=0` in configs to prevent WSL multiprocessing hangs

---

**Mission**: Shock the world with O(N) clinical seizure detection 🚀