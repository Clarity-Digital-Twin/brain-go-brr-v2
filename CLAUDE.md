# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🧠 Project Overview

Brain-Go-Brr v2.6: TCN + Bi-Mamba + GNN + LPE for clinical EEG seizure detection — O(N) sequence modeling with bidirectional SSM and graph neural networks with Laplacian positional encoding.

Why this architecture works:
- Transformers struggle on long EEG (O(N²) cost)
- Pure CNNs miss global temporal context
- Bidirectional Mamba brings O(N) global context efficiently
- GNN with Laplacian PE captures spatial electrode relationships
- TCN provides multi-scale temporal feature extraction

## ⚡ Essential Commands

| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) — RUN AFTER EVERY CHANGE ✅ |
| `make t` | Fast tests (no coverage) |
| `make test` | Full tests with coverage |
| `make test-gpu` | GPU-specific tests |
| `make setup` | Initial setup (uv, hooks) |
| `make setup-gpu` | Install v2.6 stack (Mamba+PyG+TCN) — REQUIRED |
| `make s` | Smoke test (1 epoch, 3 files) |
| `make train-local` | Full v2.6 training (100 epochs) |
| `uv sync -E gpu` | GPU extra (Mamba-SSM) |
| `uv sync -E post,eval` | Extras: post-proc + eval |
| `python -m src train configs/local/smoke.yaml` | Direct smoke test |
| `python -m src train configs/local/train.yaml` | Direct full training |

## 🏗️ Architecture

| Component | Specification | Location |
|-----------|--------------|----------|
| Input | 19-channel EEG @ 256 Hz | - |
| TCN Encoder | 8 layers, channels [64,128,256,512], ×16 downsample | `src/brain_brr/models/tcn.py` |
| Bi-Mamba | 6 layers, d_model=512, d_state=16, conv_kernel=4 | `src/brain_brr/models/mamba.py` |
| GNN | PyG SSGConv + Laplacian PE (k=16), α=0.05, 2 layers | `src/brain_brr/models/gnn_pyg.py` |
| Graph Builder | Heuristic cosine similarity, top_k=3 | `src/brain_brr/models/graph_builder.py` |
| Hysteresis | tau_on=0.86, tau_off=0.78 | `src/brain_brr/post/postprocess.py` |
| Output | Per-timestep seizure probabilities | - |

**Current v2.6**: Uses heuristic graph builder (cosine similarity)
**Future v3.0**: Will add edge Mamba stream for learned adjacency matrices

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

### Critical Installation Order
1. **PyTorch 2.2.2**: Must be EXACT version with CUDA 12.1
2. **Mamba-SSM 2.2.2**: Compile with `--no-build-isolation`
3. **PyG 2.6.1**: Use pre-built wheels from torch-2.2.0+cu121
4. **TCN 1.2.3**: Pure PyTorch, installs easily

See `INSTALLATION.md` for detailed steps.

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

## 🚀 Training Configuration

### Local (RTX 4090)
```yaml
# configs/local/train.yaml
data:
  cache_dir: cache/tusz  # MUST use existing cache with 3734 files
training:
  batch_size: 12  # Conservative for 24GB VRAM
  mixed_precision: false  # Disabled - causes NaNs on RTX 4090
```

### Modal (A100-80GB)
```yaml
# configs/modal/train.yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
training:
  batch_size: 64  # A100 can handle larger batches
  mixed_precision: true  # A100 tensor cores
```

## ⚠️ Critical Notes

- **v2.6 Stack**: TCN + BiMamba + GNN + LPE (31M parameters)
- **Installation**: Run `make setup-gpu` after base setup
- **Cache**: Local uses `cache/tusz/`, Modal uses `/results/cache/tusz/`
- **Focal Loss**: REQUIRED for 12:1 class imbalance
- **Balanced Sampling**: CRITICAL or batches may have zero seizures
- **WSL**: `export UV_LINK_MODE=copy` prevents permission issues
- **Modal**: Needs 24 CPU cores + 96GB RAM to avoid bottlenecks

---

**Mission**: Shock the world with O(N) clinical seizure detection 🚀