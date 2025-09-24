# CLAUDE.md

This file provides critical project context for Claude Code (claude.ai/code) when working with this codebase. Claude automatically ingests this file to understand your project requirements, conventions, and workflows.

## 🧠 Project Overview

Brain-Go-Brr v2.6 + V3: Clinical EEG seizure detection using **TCN + BiMamba + GNN + LPE** — achieving O(N) complexity with state-space models and graph neural networks.

**Architecture Stack (31M parameters)**:
- **TCN**: Multi-scale temporal features (8 layers, channels [64,128,256,512])
- **BiMamba**: Bidirectional SSM for O(N) global context (6 layers, d_model=512)
- **GNN**: Spatial electrode relationships via SSGConv (α=0.05, 2 layers)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors)

Paths supported:
- **V2 (architecture: tcn)** → heuristic cosine similarity graphs (top_k=3)
- **V3 (architecture: v3)** → dual‑stream with learned adjacency (Edge Mamba) + vectorized GNN
See V3 details: docs/architecture/V3_ACTUAL.md

## 🚀 Quick Commands

### Essential Development Commands
| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) — **RUN AFTER EVERY CHANGE** ✅ |
| `make t` | Fast tests without coverage |
| `make test` | Full test suite with coverage |
| `make setup` | Initial setup with uv |
| `make setup-gpu` | Install GPU stack (Mamba+PyG+TCN) — **REQUIRED for v2.6/V3** |
| `make s` | Smoke test (1 epoch, 3 files) |
| `make train-local` | Full training (100 epochs, 3734 files) |

### Local Training (RTX 4090)
```bash
# Smoke test (quick validation)
make s  # or: python -m src train configs/local/smoke.yaml

# Full training in tmux (recommended)
tmux new -s train
make train-local  # or: .venv/bin/python -m src train configs/local/train.yaml
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
# List sessions: tmux ls
```

### Modal Cloud Deployment (A100-80GB)
```bash
# Test Mamba CUDA before training
modal run deploy/modal/app.py --action test-mamba

# Smoke test (quick validation)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

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
│   ├── gnn_pyg.py      # PyG GNN with Laplacian PE
│   └── graph_builder.py # Heuristic adjacency builder
├── data/               # EEG data pipeline
│   ├── loader.py       # EDF processing with MNE
│   └── dataset.py      # PyTorch Dataset with balanced sampling
├── train/              # Training loop
│   └── loop.py         # Main training orchestrator
├── post/               # Post-processing
│   └── postprocess.py  # Hysteresis + morphology
└── config/             # Pydantic configuration schemas

configs/                 # Training configurations
├── local/              # RTX 4090 optimized
│   ├── smoke.yaml      # 1 epoch, 3 files (BGB_SMOKE_TEST=1)
│   └── train.yaml      # 100 epochs, 3734 files
└── modal/              # A100-80GB optimized
    ├── smoke.yaml      # 1 epoch, 50 files
    └── train.yaml      # 100 epochs, 3734 files

cache/tusz/             # Pre-processed data (local)
├── train/              # 3734 NPZ files + manifest.json
└── val/                # 933 NPZ files

/results/cache/tusz/    # Modal persistent SSD cache
```

## ⚙️ Critical Configuration

### Local Training (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz          # MUST exist with 3734 files!
  num_workers: 0                  # WSL2 multiprocessing fix
training:
  batch_size: 12                  # Conservative for 24GB VRAM
  mixed_precision: false          # DISABLED - causes NaNs
  loss: focal                     # REQUIRED for 12:1 imbalance
  use_balanced_sampling: true     # CRITICAL or no seizures in batches
```

### Modal Cloud (A100-80GB)
```yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
  num_workers: 8                  # A100 handles parallel IO
training:
  batch_size: 64                  # Larger batch for 80GB
  mixed_precision: true           # A100 tensor cores
resources:
  cpu: 24                         # Avoid bottlenecks (default: 0.125!)
  memory: 98304                   # 96GB RAM
```

## 🔧 Installation Requirements

### Exact Version Lock (DO NOT CHANGE)
```
PyTorch==2.2.2+cu121      # EXACT version for Mamba+PyG
CUDA Toolkit==12.1        # Must match PyTorch
mamba-ssm==2.2.2          # Later versions have bugs
causal-conv1d==1.4.0      # 1.5+ needs PyTorch 2.4+
torch-geometric==2.6.1    # Latest for torch 2.2.2
numpy==1.26.4             # 2.x breaks mamba-ssm
```

### Installation Order (CRITICAL)
1. Base environment: `make setup`
2. GPU components: `make setup-gpu`
3. Verify: `.venv/bin/python -c "from mamba_ssm import Mamba2; print('✅')"`

**Note**: PyG requires pre-built wheels from https://data.pyg.org/whl/torch-2.2.0+cu121.html

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
# Debugging
export BGB_NAN_DEBUG=1               # Debug NaN losses
export SEIZURE_MAMBA_FORCE_FALLBACK=1 # Force Conv1d fallback
export BGB_FORCE_MANIFEST_REBUILD=1   # Rebuild cache manifest

# Data limits
export BGB_SMOKE_TEST=1              # Limit to 3 files
export BGB_LIMIT_FILES=50            # Custom file limit

# WSL2 fixes
export UV_LINK_MODE=copy             # Prevent permission issues
```

## 🚨 Critical Notes

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Cache directory wrong | Local: `cache/tusz/`, Modal: `/results/cache/tusz/` |
| Zero seizures in batches | Enable `use_balanced_sampling: true` |
| NaN losses on RTX 4090 | Set `mixed_precision: false` |
| Modal training stuck | Increase CPU cores (24) and RAM (96GB) |
| PyG installation fails | Use pre-built wheels, not `uv sync -E graph` |
| Mamba CUDA errors | Ensure CUDA 12.1 toolkit installed |

### Modal-Specific Settings
- **Resources**: 24 CPU cores + 96GB RAM (defaults are too low!)
- **Storage**: Cache on `/results/` (persistent SSD), never S3
- **W&B**: Set entity to team name if using team API key
- **Detached runs**: Use `--detach` for long training sessions

### Key Files to Reference
- Installation: `INSTALLATION.md`
- Architecture evolution: `ARCHITECTURE_EVOLUTION.md`
- Config details: `configs/README.md`
- Modal deployment: `docs/03-deployment/modal/deploy.md`
- Local setup: `docs/03-deployment/local/setup.md`

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

**Mission**: Deploy O(N) clinical seizure detection that beats transformer baselines 🚀
