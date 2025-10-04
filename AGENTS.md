# AGENTS.md

This file provides critical project context for AI coding agents when working with this codebase. Claude automatically ingests this file to understand your project requirements, conventions, and workflows.

## 🧠 Project Overview

Brain-Go-Brr v3.6.0 (Modal Training Baseline): Clinical EEG seizure detection using **TCN + BiMamba + GNN + Dynamic LPE** — achieving O(N) complexity with state-space models and graph neural networks. V2 heuristic paths are removed; V3 is the only supported architecture.

**Architecture Stack (31M parameters)**:
- **TCN**: Multi-scale temporal features (8 layers, channels [64,128,256,512])
- **Node Mamba**: Bidirectional SSM for O(N) global context (19×, 6 layers, d_model=64 per electrode)
- **Edge Mamba**: Learned adjacency (171×, 2 layers, d_model=16)
- **GNN**: Spatial electrode relationships via SSGConv (α=0.05, 2 layers)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors, dynamic)

Current Architecture (v3.6.0 - October 3, 2025):
- **V3 dual-stream** → Node (19×) and Edge (171×) parallel processing
- **Edge similarity clamping** → Prevents ±1.0 boundary explosions (margin=0.01)
- **Dynamic Laplacian PE** → Time-evolving graph structure
- **Detached eigenvectors** → Prevents gradient explosion (gnn_pyg.py:205)
- **3-tier NaN protection** → Gradient sanitization + clamping + monitoring
- **Constants centralized** → All clinical thresholds in `src/brain_brr/constants.py`

See V3 details: docs/04-model/v3-architecture.md

## 🚀 Quick Commands

### Essential Development Commands
| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) — **RUN AFTER EVERY CHANGE** ✅ |
| `make t` | Fast tests without coverage |
| `make test` | Full test suite with coverage |
| `make setup` | Initial setup with uv |
| `make setup-gpu` | Install GPU stack (Mamba+PyG+TCN) — **REQUIRED for V3** |
| `make s` | Smoke test (1 epoch, 3 files) |
| `make train-local` | Full training (100 epochs, official train/dev splits) |

### Local Training (RTX 4090)
```bash
# 🚨 CRITICAL: Set NaN protection flags (REQUIRED for PyTorch 2.5.0+)
export BGB_SANITIZE_GRADS=1  # Prevents gradient explosion
export BGB_NAN_DEBUG=1       # Shows NaN warnings

# Smoke test (quick validation)
make s  # or: python -m src train configs/local/smoke.yaml

# Full training in tmux (recommended)
tmux new -s train
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
make train-local  # or: .venv/bin/python -m src train configs/local/train.yaml
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
# List sessions: tmux ls
```

### Modal Cloud Deployment (A100-80GB)
```bash
# One-time cache population from S3 (use --detach!)
modal run --detach deploy/modal/app.py --action populate-cache

# Test Mamba CUDA before training
modal run deploy/modal/app.py --action test-mamba

# Smoke test (quick validation, use --detach)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached for long runs)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor training
modal app list                    # List running apps
modal app logs <app-id>           # Stream logs
modal app stop <app-id>          # Stop training

# Resume from checkpoint
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train.yaml --resume true
```

## 📁 Project Structure

```
src/brain_brr/           # Core implementation
├── models/
│   ├── detector.py      # Main SeizureDetector orchestrator
│   ├── tcn.py          # TCN encoder (8 layers, stride_down=16)
│   ├── mamba.py        # Bidirectional Mamba (6 layers)
│   ├── gnn_pyg.py      # PyG GNN with Laplacian PE (dynamic by default)
│   ├── edge_features.py # Edge similarity + adjacency assembly (clamp at source)
│   └── fusion.py       # Gated/multihead fusion (PR-4)
├── data/               # EEG data pipeline
│   ├── io.py           # EDF I/O and CSV parsing
│   ├── preprocess.py   # Preprocessing (filters, z-score, ±10σ clip)
│   └── datasets.py     # BalancedSeizureDataset and cache integration
├── train/              # Training loop
│   └── loop.py         # Main training orchestrator
├── post/               # Post-processing
│   └── postprocess.py  # Hysteresis + morphology
└── config/             # Pydantic configuration schemas

configs/                 # Training configurations
├── local/              # RTX 4090 optimized
│   ├── smoke.yaml      # 1 epoch, 3 files (BGB_SMOKE_TEST=1)
│   └── train.yaml      # 100 epochs (train/dev official splits)
└── modal/              # A100-80GB optimized
    ├── smoke.yaml      # 1 epoch, 50 files
    └── train.yaml      # 100 epochs (train/dev official splits)

cache/tusz/             # Pre-processed data (local)
├── train/              # 4,667 NPZ files + manifest.json
└── dev/                # 1,832 NPZ files + manifest.json

/results/cache/tusz/    # Modal persistent SSD cache
```

## ⚙️ Critical Configuration

### Local Training (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz          # MUST exist: train (4667) + dev (1832)
  num_workers: 0                  # WSL2 multiprocessing fix
training:
  batch_size: 12                  # Conservative for 24GB VRAM
  mixed_precision: false          # DISABLED - causes NaNs on RTX 4090
  loss: focal                     # REQUIRED for 12:1 imbalance
  use_balanced_sampling: true     # CRITICAL or no seizures in batches
model:
  graph:
    edge_similarity_margin: 0.01  # v3.3.0: Boundary safety
```

### Modal Cloud (A100-80GB)
```yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
  num_workers: 4                  # SAFE: 8 caused overhead (v3.4.1)
  persistent_workers: false       # CRITICAL: Prevents hangs
  prefetch_factor: 2              # SAFE: 4/8 caused OOM
training:
  batch_size: 32                  # v3.4.1: 64 causes OOM (77GB peak)
  gradient_accumulation_steps: 2  # Maintain effective batch=64
  mixed_precision: true           # A100 tensor cores (3.8x faster)
  gradient_clip: 0.5              # Gradient protection
model:
  graph:
    edge_similarity_margin: 0.01  # v3.3.0: Boundary safety
resources:
  cpu: 24                         # Avoid bottlenecks (default: 0.125!)
  memory: 98304                   # 96GB RAM
```

## 🔧 Installation Requirements

### Exact Version Lock (DO NOT CHANGE)
```
PyTorch==2.5.0+cu124      # EXACT version for Mamba+PyG
CUDA Toolkit==12.4        # Must match PyTorch
mamba-ssm==2.2.5          # Includes A100 int64 indexing fix (PR #708)
causal-conv1d==1.5.2      # Latest stable for PyTorch 2.5+
torch-geometric==2.6.1    # Latest for torch 2.5.0
numpy==1.26.4             # 2.x breaks mamba-ssm
```

### Installation Order (CRITICAL)
**PREREQUISITE**: Install CUDA 12.4 toolkit BEFORE running make commands:
```bash
# Ubuntu/WSL2
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4
```

1. Base environment: `make setup`
2. GPU components: `make setup-gpu` (clears caches, builds from source)
3. Verify: `.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"`

**Note**: PyG requires pre-built wheels from https://data.pyg.org/whl/torch-2.5.0+cu124.html

## 🏥 Clinical Specifications

### Data Pipeline
1. **Input**: TUH EEG Seizure Corpus (10-20 montage, 19 channels)
2. **Preprocessing**: Bandpass 0.5-120Hz, 60Hz notch, resample to 256Hz
3. **Windowing**: 60s windows with 10s stride (83% overlap)
4. **Normalization**: Per-channel z-score

### Channel Order (MUST maintain)
```python
["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
 "Fz", "Cz", "Pz",
 "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
```

### Post-Processing
1. **Hysteresis**: τ_on=0.86, τ_off=0.78
2. **Morphology**: Opening(11), Closing(31)
3. **Duration**: 3-600s valid range
4. **Merging**: Events within 2s

### Performance Targets (TAES)
| FA Rate | Target Sensitivity |
|---------|-------------------|
| 10 FA/24h | >95% |
| 5 FA/24h | >90% |
| 1 FA/24h | >75% |

## 🛠️ Development Guidelines

### Code Requirements
- **Python 3.11+** with full type hints
- **Ruff** line length 100, 4-space indent
- **Imports**: stdlib → third-party → first-party (sorted)
- **No comments** unless explicitly requested
- **Follow patterns** from neighboring files

### Testing Strategy
```bash
make t              # Quick tests for development
make test           # Full coverage before commits
make test-gpu       # GPU-specific tests
```

### Environment Variables
```bash
# Debugging (CRITICAL for PyTorch 2.5.0+)
export BGB_SANITIZE_GRADS=1          # RECOMMENDED: Sanitize NaN gradients
export BGB_NAN_DEBUG=1               # Debug NaN losses
export SEIZURE_MAMBA_FORCE_FALLBACK=1 # Force Conv1d fallback
export BGB_FORCE_MANIFEST_REBUILD=1   # Rebuild cache manifest

# Data limits
export BGB_SMOKE_TEST=1              # Limit to 3 files
export BGB_LIMIT_FILES=50            # Custom file limit

# Testing
export BGB_SKIP_GPU_TESTS=1          # Skip GPU tests during training

# WSL2 fixes
export UV_LINK_MODE=copy             # Prevent permission issues
```

## 🚨 Critical Notes

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Symbol mismatch: `_ZN3c104cuda9SetDeviceEab`** | **Rebuild mamba-ssm from source with `--no-binary` flag** |
| **CUDA 12.4 toolkit not found** | **Install: `sudo apt-get install -y cuda-toolkit-12-4`** |
| Cache directory wrong | Local: `cache/tusz/`, Modal: `/results/cache/tusz/` |
| Zero seizures in batches | Enable `use_balanced_sampling: true` |
| NaN losses on RTX 4090 | Set `mixed_precision: false` + `BGB_SANITIZE_GRADS=1` |
| **Non-finite logits** | **Rebuild cache after Sep 26 fix + use `BGB_SANITIZE_GRADS=1`** |
| **Edge similarity explosions** | **v3.3.0: Set `edge_similarity_margin: 0.01` in configs** |
| **Gradient spikes (7.03+)** | **v3.3.1: FIXED - eigenvectors detached in gnn_pyg.py:205** |
| **Modal XID 31 GPU crashes** | **v3.3.1: FIXED - unique Triton cache dirs in deploy/modal/app.py** |
| Modal training stuck | Increase CPU cores (24) and RAM (96GB) |
| PyG installation fails | Use pre-built wheels, not `uv sync -E graph` |
| Mamba CUDA errors | Ensure CUDA 12.4 toolkit installed, rebuild from source |

### Modal-Specific Settings
- **Resources**: 24 CPU cores + 96GB RAM (defaults are too low!)
- **Storage**: Cache on `/results/` (persistent SSD), never S3
- **W&B**: Set entity to team name if using team API key
- **Detached runs**: Use `--detach` for long training sessions

### Key Files to Reference
- Installation: `INSTALLATION.md`
- Architecture evolution: `ARCHITECTURE_EVOLUTION.md`
- Config details: `configs/README.md`
- Modal training: `docs/05-training/modal.md`
- Local training: `docs/05-training/local.md`

## 📊 Expected Performance

### Training Times
- **Local (RTX 4090)**: ~2-3 hours/epoch, ~200-300 hours total
- **Modal (A100)**: ~1 hour/epoch, ~100 hours total (~$319)
- **Smoke test**: ~5 minutes both platforms

### Resource Usage
- **VRAM**: 12-20GB (RTX 4090), 40-60GB (A100)
- **Cache size**: ~50GB processed NPZ files
- **Checkpoint size**: ~125MB per epoch

---

---

**Mission**: Deploy V3 dual-stream architecture with Dynamic LPE for <1 FA/24h clinical seizure detection 🚀

**Current Status (v3.6.0 - October 3, 2025)**:
- ✅ **Constants centralization COMPLETE** - All magic numbers in `constants.py`
- ✅ **Modal training LAUNCHED** - Full 100-epoch run active on A100-80GB (App: ap-BwyQN1PX1prmfzbWGlUDqS)
- ✅ **Smoke test validated** - Zero crashes, W&B integration perfect
- ✅ **Clean code refactoring COMPLETE** - All modules extracted and optimized
- ✅ **Production ready** - PyTorch 2.5.0 + mamba-ssm 2.2.5 (XID 31 crashes resolved)
- ✅ **Training ROCK SOLID** - Zero NaN/Inf issues with gradient sanitization
- 🔴 **1 active P1** - Validation loss weighting (non-blocking, defer to post-training)
