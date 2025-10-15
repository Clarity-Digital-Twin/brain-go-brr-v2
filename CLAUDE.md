# CLAUDE.md

This file provides critical project context for Claude Code (claude.ai/code) when working with this codebase. Claude automatically ingests this file to understand your project requirements, conventions, and workflows.

## 🧠 Project Overview

Brain-Go-Brr v4.0.0 (FLA Production + WSL2 Fix): Clinical EEG seizure detection with **dual production stacks** — BiMamba2 (baseline) and FLA (BiGatedDeltaNet research). Both use **TCN + SSM + GNN + Dynamic LPE** achieving O(N) complexity. **Hands-free Modal A100 training** with auto-restart, exact mid-epoch resume via StatefulDataLoader, and **local FLA training now works** (WSL2 SIGBUS fix).

**Architecture Stack (31M parameters)**:
- **TCN**: Multi-scale temporal features (8 layers, channels [64,128,256,512])
- **BiMamba**: Bidirectional SSM for O(N) global context (6 layers, d_model=64 per electrode)
- **GNN**: Spatial electrode relationships via SSGConv (α=0.05, 2 layers)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors)

Current Architecture (v4.0.0 - October 12, 2025):
- **V3 dual-stream** → Node (19×) and Edge (171×) parallel processing
- **Memory-mapped cache (NPY)** → <1 GB RAM vs 387 GB for NPZ, 99.6% faster startup
- **Auto-restart training** → Hands-free 100-epoch training via modal.Period(hours=23) with overlap protection
- **StatefulDataLoader** → Exact mid-epoch resume with PyTorch official dataloader state management ($150+ savings)
- **Three checkpoint fixes** → All resume bugs eliminated:
  1. Resume fix: Saves epoch+1 (next) not epoch (completed), prevents 14h waste per restart
  2. Buffer fix: Handle `register_buffer(None)` timing bug, enables mid-epoch checkpoints
  3. RNG fix: Force RNG states to CPU before restoration, enables GPU resume
- **Bulletproof checkpoints** → Atomic saves (temp + fsync + rename), AMP scaler + RNG capture, every 30min
- **Timeout guard** → 23h wall-clock limit with 1h safety margin, graceful exit before Modal kill
- **Complete tensor safety** → All 3 datasets use copy-on-read tensors for read-only mmap safety (no PyTorch warnings)
- **Edge similarity clamping** → Prevents ±1.0 boundary explosions (PR-5)
- **Dynamic Laplacian PE** → Time-evolving graph structure, fully dynamic every timestep
- **Detached eigenvectors** → Prevents gradient explosion through eigendecomposition (gnn_pyg.py:205)
- **3-tier NaN protection** → Gradient sanitization + clamping + monitoring
- **Modal 1.0 compatible** → Updated max_containers parameter for future compatibility
- **Zero technical debt** → All P3 items resolved (Pydantic schemas correct, .gitkeep files added), production ready
- **WSL2 SIGBUS fix** → Local FLA training now works (cache must be on native ext4 filesystem, not Windows drives)
- **Dual production stacks** → BiMamba2 (Modal A100) + FLA (local RTX 4090) both training simultaneously

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

**🚨 CRITICAL: Choose the RIGHT action for your use case!**

| Use Case | Action | Auto-Restart? | When to Use |
|----------|--------|---------------|-------------|
| **Full Training (100 epochs)** | `schedule-training` | ✅ YES | **ALWAYS use this for production!** |
| Smoke test (validation) | `train` | ❌ NO | Quick tests (<1 hour) |
| One-off experiment | `train` | ❌ NO | Single run experiments |

**WHY THIS MATTERS:**
- `--action train` → Runs ONCE, exits after 23h, **requires manual restart** ❌
- `--action schedule-training` → Runs FOREVER with 23h auto-restart until 100 epochs ✅

```bash
# Test Mamba CUDA before training
modal run deploy/modal/app.py --action test-mamba

# ============================================================================
# SMOKE TESTS (use --action train, no auto-restart needed)
# ============================================================================
# BiMamba2 smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml

# FLA smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml

# ============================================================================
# FULL TRAINING (use --action schedule-training for auto-restart!)
# ============================================================================
# BiMamba2 full training - Auto-Restart (CORRECT ✅)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml

# FLA full training - Auto-Restart (CORRECT ✅)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_fla.yaml

# ============================================================================
# MANUAL MODE (use ONLY for experiments, NOT production!)
# ============================================================================
# BiMamba2 manual (runs once, exits after 23h, requires manual resume)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml --resume

# ============================================================================
# MONITORING & CONTROL
# ============================================================================
modal app list                    # List running apps
modal app logs <app-id>           # Stream logs
modal app stop brain-go-brr-v2    # Stop auto-restart scheduler
```

**🔍 How to verify you started the RIGHT job:**
```bash
modal app list
# Look for: 2 tasks running (scheduler + train) ✅
# NOT: 1 task (manual train only) ❌
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
4. FLA library (OPTIONAL): `make setup-fla` (for Gated DeltaNet research stack)
5. Verify: `.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"`

**Note**: PyG requires pre-built wheels from https://data.pyg.org/whl/torch-2.5.0+cu124.html

**Triton Warning**: FLA installation may show "Triton 3.1.0 below recommended 3.2.0" - this is expected and harmless. Triton 3.2.0 requires PyTorch 2.6+, which breaks mamba-ssm compatibility. FLA requirement (Triton >=3.0) is satisfied.

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
| **WSL2 SIGBUS crash (FLA training)** | **Cache on Windows drives causes mmap page evictions. Move cache to native ext4 filesystem (see INSTALLATION.md#6, SIGBUS_CRASH_ANALYSIS.md)** |
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
| FLA Triton warning (3.1.0 vs 3.2.0) | **EXPECTED**: FLA requires Triton >=3.0 (satisfied). 3.2.0 needs PyTorch 2.6+ which breaks mamba-ssm |

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

## 📊 Actual Performance (MEASURED from production training)

### Training Times (LOCAL FLA - RTX 4090)
**MEASURED from 6 complete epochs (Epochs 1-6):**
- **Training**: ~4.1h per epoch (7702 batches @ ~2.1s/batch)
- **Validation**: ~5.5h per epoch (18528 batches, disk-backed)
- **Total**: ~9.6h per epoch average
- **100 epochs**: ~960 hours (40 days)
- **Current status**: Epoch 7/100 in progress (7% complete)

**Breakdown by epoch:**
- Epoch 1: 7.20h (1.62h train + 5.58h val) - faster due to warmup
- Epochs 2-6: 10.1h avg (4.6h train + 5.5h val) - consistent performance

**CRITICAL FINDING**: Validation takes **1.3× longer** than training (57.2% vs 42.8% of epoch time)

### Training Times (MODAL BiMamba2 - A100-80GB)
**DOCUMENTED** (not measured - training paused at Epoch 6):
- Training: ~1-2h per epoch (documented)
- Validation: ~5.8h per epoch (documented)
- Total: ~7-12h per epoch (documented)
- **Status**: PAUSED due to high costs

### Cost Comparison
| Platform | 6 Epochs | Cost/Epoch | 100 Epochs | Notes |
|----------|----------|------------|------------|-------|
| **Local FLA (RTX 4090)** | FREE | $0 | **$0** | Only electricity (~negligible) |
| **Modal BiMamba2 (A100)** | $1,118 | **$186** | **$18,600** | Training PAUSED |

**Modal A100-80GB Cost Breakdown**:
- GPU: $2.50/hr + 24 CPU cores: $1.13/hr + 96GB RAM: $0.77/hr
- **Total: $4.40/hour**
- **Per epoch (7-12h)**: $31-$53
- **Actual cost**: $186/epoch (from 6 epochs measured)

### Smoke Tests
- **Local/Docker**: ~5 minutes (3 files)
- **Modal**: ~5-10 minutes (50 files)

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

**Current Status (v4.0.0 - October 12, 2025 - FLA Production + WSL2 Fix)**:
- ✅ **MAJOR: Dual production stacks** - BiMamba2 (Modal A100) + FLA (local RTX 4090) both training simultaneously
- ✅ **MAJOR: WSL2 SIGBUS fix** - Local FLA training now works after cache migration to native ext4 filesystem
- ✅ **MAJOR: FLA stack validated** - Training verified past previous crash point (batch 5401 vs crash at 2890)
- ✅ **Zero technical debt** - All items resolved (P0/P1/P2/P3), production ready
- ✅ **Auto-restart training** - Hands-free 100-epoch training via modal.Period(hours=23), zero manual intervention
- ✅ **StatefulDataLoader integrated** - Exact mid-epoch resume via PyTorch official dataloader state management
- ✅ **Checkpoint resume fix** - Saves epoch+1 instead of epoch, prevents 14h waste per restart
- ✅ **Mid-epoch checkpoint robustness** - Saves exact batch position, eliminates 1-2h wasted compute per restart ($150+ savings)
- ✅ **Pydantic v2 warning fix** - Clean Annotated pattern for forward references, zero warnings in production logs
- ✅ **Bulletproof checkpoints** - Atomic saves every 30min, AMP scaler + RNG + DataLoader state capture
- ✅ **Backward compatibility** - Old checkpoints still work (logs warning, restarts from epoch start)
- ✅ **Timeout guard** - 23h wall-clock limit, 1h safety margin, graceful exit before Modal kill
- ✅ **Modal 1.0 migration complete** - Updated max_containers parameter, deprecation warnings fixed
- 📊 **Current training**: BiMamba2 (Modal, PAUSED at Epoch 6) + FLA (Local, Epoch 7/100) - local training progressing normally
- 📚 **See**: `docs/08-operations/wsl2-sigbus-fix.md` for WSL2 details, `docs/archive_v4/` for incident analysis
