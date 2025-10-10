# CLAUDE.md

This file provides critical project context for Claude Code (claude.ai/code) when working with this codebase. Claude automatically ingests this file to understand your project requirements, conventions, and workflows.

## 🧠 Project Overview

Brain-Go-Brr v3.10.0 (Auto-Restart & Checkpoint Fix): Clinical EEG seizure detection using **TCN + BiMamba + GNN + Dynamic LPE** with stable eigendecomposition — achieving O(N) complexity with state-space models and graph neural networks. **Hands-free Modal A100 training** with auto-restart, bulletproof checkpoints, and checkpoint resume fix saving $616 over 100 epochs.

**Architecture Stack (31M parameters)**:
- **TCN**: Multi-scale temporal features (8 layers, channels [64,128,256,512])
- **BiMamba**: Bidirectional SSM for O(N) global context (6 layers, d_model=64 per electrode)
- **GNN**: Spatial electrode relationships via SSGConv (α=0.05, 2 layers)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors)

Current Architecture (v3.10.0 - October 10, 2025):
- **V3 dual-stream** → Node (19×) and Edge (171×) parallel processing
- **Memory-mapped cache (NPY)** → <1 GB RAM vs 387 GB for NPZ, 99.6% faster startup
- **Auto-restart training** → Hands-free 100-epoch training via modal.Period(hours=23) with overlap protection
- **Checkpoint resume fix** → Saves epoch+1 (next to train) instead of epoch (completed), prevents 14h waste per restart
- **Bulletproof checkpoints** → Atomic saves (temp + fsync + rename), AMP scaler + RNG capture, every 30min
- **Timeout guard** → 23h wall-clock limit with 1h safety margin, graceful exit before Modal kill
- **Complete tensor safety** → All 3 datasets use copy-on-read tensors for read-only mmap safety (no PyTorch warnings)
- **Edge similarity clamping** → Prevents ±1.0 boundary explosions (PR-5)
- **Dynamic Laplacian PE** → Time-evolving graph structure, fully dynamic every timestep
- **Detached eigenvectors** → Prevents gradient explosion through eigendecomposition (gnn_pyg.py:205)
- **3-tier NaN protection** → Gradient sanitization + clamping + monitoring
- **Modal 1.0 compatible** → Updated max_containers parameter for future compatibility
- **Zero technical debt** → All P0/P1/P2/P3 issues resolved, production training LIVE

## 🚀 Quick Commands

### Essential Development Commands
| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) — **RUN AFTER EVERY CHANGE** ✅ |
| `make t` | Fast tests without coverage |
| `make ts` | Training-safe tests (CPU only) — **USE DURING TRAINING** 🏃 |
| `make test` | Full test suite with coverage |
| `make setup` | Initial setup with uv |
| `make setup-gpu` | Install GPU stack (Mamba+PyG+TCN) — **REQUIRED for V3** |
| `make smoke-bimamba` | BiMamba2 smoke test (1 epoch, 3 files) |
| `make smoke-fla` | FLA smoke test (1 epoch, 3 files) |
| `make train-bimamba` | BiMamba2 full training (100 epochs) |
| `make train-fla` | FLA full training (100 epochs) |

### Local Training (RTX 4090)
```bash
# NaN Protection (gradient clipping is always applied from config)
# Optional debugging (not required for normal training):
# export BGB_SANITIZE_GRADS=1  # Debug: Log where NaNs occur
export BGB_NAN_DEBUG=1         # Enable NaN warnings

# BiMamba2 smoke test (3 files, ~5 min - fast pipeline validation)
make smoke-bimamba  # or: python -m src train configs/local/smoke_bimamba.yaml

# FLA smoke test (3 files, ~5 min)
make smoke-fla  # or: python -m src train configs/local/smoke_fla.yaml

# BiMamba2 full training in tmux (recommended)
tmux new -s train-bimamba
export BGB_NAN_DEBUG=1
make train-bimamba  # or: .venv/bin/python -m src train configs/local/train_bimamba.yaml

# FLA full training in tmux
tmux new -s train-fla
export BGB_NAN_DEBUG=1
make train-fla  # or: .venv/bin/python -m src train configs/local/train_fla.yaml
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
# List sessions: tmux ls
```

**NOTE**: See `docs/04-model/v3-stability-evolution.md` for gradient explosion fix details and `docs/08-operations/nan-prevention-complete.md` for complete NaN protection documentation.

### Docker Training (GPU-accelerated containers)
```bash
# Smoke test (3 files, ~5 min - fast validation)
docker compose up smoke-test

# Integration test (50 files, ~60 min - deeper validation)
docker compose up integration-test

# Full training (100 epochs)
docker compose up train

# Development shell
docker compose run dev
```

**Smoke Test Standards**:
- **Local/Docker**: 3 files (BGB_SMOKE_TEST=1)
- **Modal**: 50 files (BGB_LIMIT_FILES=50)
- Both use same architecture, just different scale

**Monitoring**:
- **W&B**: Set `WANDB_API_KEY` in environment (see `docs/05-training/docker.md`)
- **TensorBoard**: `docker compose up tensorboard` → http://localhost:6006

### Modal Cloud Deployment (A100-80GB)
```bash
# Test Mamba CUDA before training
modal run deploy/modal/app.py --action test-mamba

# BiMamba2 smoke test (50 files, ~10 min - quick validation)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml

# FLA smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml

# BiMamba2 full training - Manual (requires manual resume every 23h)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# BiMamba2 full training - Auto-Restart (RECOMMENDED: hands-free to 100 epochs)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml

# FLA full training (ALWAYS use --detach for long runs)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_fla.yaml

# Monitor training
modal app list                    # List running apps
modal app logs <app-id>           # Stream logs
modal app stop brain-go-brr-v2    # Stop auto-restart training

# Resume from checkpoint (manual, for one-off runs)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume
```

**NOTE**: Modal automatically sets `BGB_NAN_DEBUG=1` via `deploy/modal/app.py`. Gradient clipping (0.5) provides primary NaN protection.

## 📁 Project Structure

```
src/brain_brr/           # Core implementation
├── models/
│   ├── detector.py      # Main SeizureDetector orchestrator
│   ├── tcn.py          # TCN encoder (8 layers, stride_down=16)
│   ├── mamba.py        # Bidirectional Mamba (6 layers)
│   ├── gnn_pyg.py      # PyG GNN with Dynamic Laplacian PE
│   ├── edge_features.py # Edge similarity computation with margin
│   └── fusion.py       # Multi-head gated fusion (PR-4)
├── data/               # EEG data pipeline
│   ├── io.py           # EDF I/O and CSV parsing
│   ├── preprocess.py   # Preprocessing (filters, z-score, ±10σ clip)
│   └── datasets.py     # Balanced dataset and cache integration
├── train/              # Training loop
│   └── loop.py         # Main training orchestrator
├── post/               # Post-processing
│   └── postprocess.py  # Hysteresis + morphology
└── config/             # Pydantic configuration schemas

configs/                 # Training configurations
├── local/              # RTX 4090 optimized
│   ├── smoke_bimamba.yaml  # BiMamba2: 1 epoch, 3 files
│   ├── train_bimamba.yaml  # BiMamba2: 100 epochs (official train/dev)
│   ├── smoke_fla.yaml      # FLA: 1 epoch, 3 files
│   └── train_fla.yaml      # FLA: 100 epochs (official train/dev)
└── modal/              # A100-80GB optimized
    ├── smoke_bimamba.yaml  # BiMamba2: 1 epoch, 50 files
    ├── train_bimamba.yaml  # BiMamba2: 100 epochs (official train/dev)
    ├── smoke_fla.yaml      # FLA: 1 epoch, 50 files
    └── train_fla.yaml      # FLA: 100 epochs (official train/dev)

cache/tusz_mmap/        # Pre-processed data (local, memory-mapped NPY)
├── train/              # 4667 NPY files (data + labels) + manifest.json
└── dev/                # 1832 NPY files (data + labels) + manifest.json

/results/cache/tusz_mmap/  # Modal persistent SSD volume (preferred; not S3)
```

## ⚙️ Critical Configuration

### Local Training (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz_mmap     # Memory-mapped NPY cache: train (4667) + dev (1832)
  num_workers: 0                  # WSL2 multiprocessing fix
training:
  batch_size: 8                   # OPTIMIZED: 2x faster than batch=4 (~20GB VRAM)
  mixed_precision: false          # DISABLED - causes NaNs
  loss: focal                     # REQUIRED for 12:1 imbalance
  use_balanced_sampling: true     # CRITICAL or no seizures in batches
model:
  graph:
    edge_similarity_margin: 0.01  # v3.3.0: Safety margin from ±1 boundaries
```

### Modal Cloud (A100-80GB)
```yaml
data:
  cache_dir: /results/cache/tusz_mmap  # Persistent SSD volume (Modal, mmap NPY)
  num_workers: 4                       # SAFE: 8 caused overhead
  prefetch_factor: 2              # SAFE: 4/8 caused OOM
training:
  batch_size: 48                  # EXPERIMENT: ~58GB peak (testing if faster than 32×2)
  gradient_accumulation_steps: 1  # Effective batch=48 (no accumulation)
  mixed_precision: true           # A100 tensor cores (3.8x faster)
model:
  graph:
    edge_similarity_margin: 0.01  # v3.3.0: Safety margin from ±1 boundaries
resources:
  cpu: 24                         # Avoid bottlenecks (default: 0.125!)
  memory: 98304                   # 96GB RAM
```

**🚨 CRITICAL A100 Memory Lessons (Oct 2025)**:
- `batch_size=64` + `gradient_accumulation_steps=1` → **77GB peak → OOM ❌**
- `batch_size=32` + `gradient_accumulation_steps=2` → **50GB peak → SAFE ✅** (effective batch=64)
- `batch_size=48` + `gradient_accumulation_steps=1` → **~58GB peak → EXPERIMENTING** (current config)
- Key insight: **batch_size controls peak memory**, **grad_accum splits backward** into chunks
- See `configs/README.md` for full OOM analysis

## 🔧 Installation Requirements

### Exact Version Lock (DO NOT CHANGE)
```
PyTorch==2.5.0+cu124      # EXACT version for Mamba+PyG
CUDA Toolkit==12.4        # Must match PyTorch
mamba-ssm==2.2.5          # Includes A100 int64 indexing fix
causal-conv1d==1.5.2      # Latest stable for PyTorch 2.5+
torch-geometric==2.6.1    # Latest for torch 2.5.0
numpy==1.26.4             # 2.x breaks mamba-ssm
```

### Installation Order (CRITICAL)
**PREREQUISITE**: Install CUDA 12.4 toolkit BEFORE running make commands:
```bash
# Ubuntu/WSL2 - Install CUDA 12.4 toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Verify
/usr/local/cuda-12.4/bin/nvcc --version
```

**Why?** PyTorch 2.5.0+cu124 includes CUDA 12.4 **runtime** but NOT the **toolkit**. The toolkit is required to compile mamba-ssm from source.

1. Install CUDA 12.4 toolkit (see above)
2. Base environment: `make setup`
3. GPU components: `make setup-gpu` (clears caches, builds from source)
4. Verify: `.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"`

**Note**: PyG requires pre-built wheels from https://data.pyg.org/whl/torch-2.5.0+cu124.html

## 🏥 Clinical Specifications

### Data Pipeline
1. **Input**: TUH EEG Seizure Corpus (10-20 montage, 19 channels)
2. **Preprocessing**: Bandpass 0.5-120Hz, 60Hz notch, resample to 256Hz
3. **Windowing**: 60s windows with 10s stride (83% overlap)
4. **Normalization**: Per-channel z-score + **outlier clipping to ±10σ**

### Channel Order (MUST maintain)
```python
["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
 "Fz", "Cz", "Pz",
 "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
```

### CRITICAL: Naming Convention
- **We use `dev` NOT `val`** for validation split to match TUSZ official naming
- Cache structure: `cache/tusz_mmap/{train,dev}/` NOT `{train,val}/`
- This prevents confusion when reading TUSZ documentation

### CRITICAL: Dataset Strategy (This is CORRECT, not a bug!)
- **Training**: Uses `BalancedSeizureDataset` with manifest to oversample seizures (8% → ~30% in batches)
  - Requires: `train/manifest.json` (auto-created if missing)
  - Why: Model needs enough seizures to learn patterns effectively
- **Validation**: Uses `ValidationDataset` with natural distribution (~8% seizures)
  - Requires: `dev/manifest.json` (instant loading, no NPZ scan)
  - Why: Measures real-world performance with fast startup (99.6% faster)
- **This is standard ML practice**: Train on balanced data, validate on real distribution

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
make ts             # Safe tests during training (CPU only)
make test           # Full coverage before commits
make test-gpu       # GPU-specific tests (stop training first)
```

**CRITICAL: Testing During Training**
- Training uses ~20GB GPU memory on RTX 4090 (batch_size=8)
- Performance tests (`make test-performance`) will OOM during training
- **Solution**: Use `make ts` or `make test-cpu` for concurrent testing
- Test suite auto-detects training and limits GPU usage to 3GB
- Set `BGB_SKIP_GPU_TESTS=1` to force-skip all GPU tests

### Environment Variables
```bash
# Debugging
export BGB_NAN_DEBUG=1               # Debug NaN losses
# export BGB_SANITIZE_GRADS=1        # Debug: Log gradient NaNs (not required)
export SEIZURE_MAMBA_FORCE_FALLBACK=1 # Force Conv1d fallback
export BGB_FORCE_MANIFEST_REBUILD=1   # Rebuild cache manifest

# Data limits (smoke testing)
export BGB_SMOKE_TEST=1              # Auto-limit to 3 files (fast validation)
export BGB_LIMIT_FILES=50            # Override: use N files instead

# Testing
export BGB_SKIP_GPU_TESTS=1          # Skip GPU tests during training

# WSL2 fixes
export UV_LINK_MODE=copy             # Prevent permission issues
```

## 🚨 Critical Notes

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Symbol mismatch: `_ZN3c104cuda9SetDeviceEab`** | **Rebuild mamba-ssm from source with `--no-binary` flag (see INSTALLATION.md#1)** |
| **CUDA 12.4 toolkit not found** | **Install: `sudo apt-get install -y cuda-toolkit-12-4`** |
| Cache directory wrong | Local: `cache/tusz_mmap/`, Modal: `/results/cache/tusz_mmap/` |
| Zero seizures in batches | Enable `use_balanced_sampling: true` |
| NaN losses on RTX 4090 | Set `mixed_precision: false` |
| **Non-finite logits** | **Rebuild cache after Sep 26 fix (gradient clipping handles it)** |
| **Edge similarity explosions** | **v3.3.0: Set `edge_similarity_margin: 0.01` in configs** |
| **Gradient spikes (7.03+)** | **v3.3.1: FIXED - eigenvectors detached in gnn_pyg.py:205 (see docs/04-model/v3-stability-evolution.md)** |
| **Modal XID 31 GPU crashes** | **v3.3.1: FIXED - unique Triton cache dirs in deploy/modal/app.py:539-546** |
| Modal training stuck | Increase CPU cores (24) and RAM (96GB) |
| PyG installation fails | Use pre-built wheels, not `uv sync -E graph` |
| Mamba CUDA errors | Ensure CUDA 12.4 toolkit installed, rebuild from source |
| CI/CD test failures | Tests properly skip when PyG not installed (v3.3.0+) |

### Modal-Specific Settings
- **Resources**: 24 CPU cores + 96GB RAM (defaults are too low!)
- **Storage**: Cache on `/results/cache/tusz_mmap/` (persistent SSD), outputs to `/results/`; avoid S3 for training hot path
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
- **Smoke test**: ~5 minutes (3 files local/Docker, 50 files Modal)

### Resource Usage
- **VRAM**: 12-20GB (RTX 4090), 40-60GB (A100)
- **Cache size**: ~50GB processed NPZ files
- **Checkpoint size**: ~125MB per epoch

### GPU-Specific Test Adjustments

Due to hardware differences, integration tests have adjusted thresholds:

| Test Type | RTX 4090 (Local) | A100 (CI/Modal) |
|-----------|------------------|-----------------|
| Batch Size | 2 (24GB VRAM) | 4-8 (80GB VRAM) |
| TCN Speed (10 batches) | <1.5s | <0.5s |
| Memory Usage | <4.0GB | <8.0GB |

**Environment Variables:**
- `BGB_TCN_SPEED_TARGET`: Override speed threshold (default: 1.5s local, 0.5s CI)
- `BGB_TCN_MEM_MAX`: Override memory threshold (default: 4.0GB)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Reduce VRAM fragmentation

---

**Mission**: Deploy V3 dual-stream architecture with Dynamic LPE for <1 FA/24h clinical seizure detection 🚀

**Current Status (v3.10.0 - October 10, 2025 - Auto-Restart & Checkpoint Fix)**:
- ✅ **Zero technical debt** - All P0/P1/P2/P3 issues RESOLVED across all priority levels
- ✅ **Auto-restart training** - Hands-free 100-epoch training via modal.Period(hours=23), zero manual intervention
- ✅ **Checkpoint resume fix** - Saves epoch+1 instead of epoch, prevents 14h waste per restart ($616 net savings)
- ✅ **Bulletproof checkpoints** - Atomic saves every 30min, AMP scaler + RNG capture, verified integrity
- ✅ **Timeout guard** - 23h wall-clock limit, 1h safety margin, graceful exit before Modal kill
- ✅ **Comprehensive validation** - PRE_TRAINING_VALIDATION.md, metrics pipeline verified from first principles
- ✅ **Test suite enhanced** - Manifest validation, checkpoint robustness, 75%+ coverage maintained
- ✅ **BiMamba2 baseline training RUNNING** - Modal A100-80GB (App: ap-ik2xwlXmuQMvPyhSfrZJfi), Step 2 of 3-step plan
- ✅ **FLA research complete** - BiGatedDeltaNet implemented, all smoke tests passed
- ✅ **Research comparison strategy** - Train both BiMamba2 and FLA stacks independently, document results for both
- ✅ **Modal 1.0 migration complete** - Updated max_containers parameter, deprecation warnings fixed
- 📊 **Research goal** - Empirical comparison on full TUSZ dataset; both results publishable regardless of outcome
- 📊 **Next**: Enable auto-restart after first manual resume completes (~23h), then hands-free to 100 epochs
- 📚 **See**: `FLA_ROADMAP.md` for complete strategy, `MODAL_CLI_REFERENCE.md` for updated commands
