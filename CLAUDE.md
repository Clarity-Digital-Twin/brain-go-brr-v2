# CLAUDE.md

**Brain-Go-Brr v4.0** - Clinical EEG seizure detection with dual production stacks (BiMamba2 + FLA). TCN + SSM + GNN + Dynamic LPE achieving O(N) complexity for <1 FA/24h detection.

## Quick Reference

### Essential Commands
```bash
make q                  # Quality check (REQUIRED after changes)
make t                  # Fast tests
make ts                 # CPU-only tests (use during training)
make setup              # Base install
make setup-gpu          # GPU stack (Mamba+PyG+TCN)
make setup-fla          # FLA stack (optional)

make smoke-bimamba      # BiMamba2: 1 epoch, 3 files (~5min)
make smoke-fla          # FLA: 1 epoch, 3 files (~5min)
make train-bimamba      # BiMamba2: Full training (100 epochs)
make train-fla          # FLA: Full training (100 epochs)
```

### Training Quick Start
```bash
# Local (RTX 4090)
export BGB_NAN_DEBUG=1
tmux new -s train-fla
make train-fla          # Detach: Ctrl+B D

# Modal (A100-80GB) - Production
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_bimamba.yaml
```

## Project Structure

```
src/brain_brr/
├── cli/                # Command-line interface
│   └── services/       # Service implementations
├── config/             # Pydantic schemas
├── data/               # EEG pipeline (io.py, preprocess.py, datasets.py)
├── eval/               # Evaluation metrics
│   └── helpers/        # Scoring utilities
├── events/             # Event detection
├── models/             # Neural networks
│   ├── builders/       # Model constructors
│   ├── detector.py     # Main orchestrator
│   ├── tcn.py          # TCN encoder (8 layers)
│   ├── mamba.py        # BiMamba (6 layers)
│   ├── gnn_pyg.py      # GNN + Dynamic LPE
│   ├── edge_features.py # Edge similarity
│   └── fusion.py       # Multi-head gated fusion
├── post/               # Post-processing (hysteresis, morphology)
├── streaming/          # Real-time inference
├── train/              # Training loop (loop.py)
└── utils/              # Helpers

configs/
├── local/              # RTX 4090 configs
│   ├── smoke_bimamba.yaml
│   ├── train_bimamba.yaml
│   ├── smoke_fla.yaml
│   └── train_fla.yaml
└── modal/              # A100-80GB configs
    └── [same structure]

tests/
├── clinical/           # Clinical validation
├── integration/        # Integration tests
│   ├── data/
│   ├── eval/
│   └── post/
├── performance/        # Benchmarks
├── train/              # Training tests
└── unit/               # Unit tests
    ├── cli/
    ├── data/
    ├── eval/
    ├── events/
    ├── models/
    ├── post/
    ├── train/
    └── utils/

cache/tusz_mmap/        # Local: Memory-mapped NPY
├── train/              # 4667 files + manifest.json
└── dev/                # 1832 files + manifest.json

/results/cache/tusz_mmap/  # Modal: Persistent SSD
```

## Architecture (31M params)

- **TCN**: 8 layers, [64,128,256,512] channels, stride_down=16
- **BiMamba**: 6 layers, d_model=64 per electrode, O(N) SSM
- **GNN**: SSGConv α=0.05, 2 layers, spatial relationships
- **LPE**: 16 Laplacian eigenvectors, dynamic per timestep
- **Dual-stream**: Node (19×) + Edge (171×) parallel processing

## Critical Configurations

### Local (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz_mmap
  num_workers: 0                  # WSL2 fix
training:
  batch_size: 8                   # 2x faster than 4 (~20GB VRAM)
  mixed_precision: false          # Causes NaNs
  loss: focal                     # 12:1 imbalance
  use_balanced_sampling: true     # CRITICAL
model:
  graph:
    edge_similarity_margin: 0.01  # ±1.0 boundary safety
```

### Modal (A100-80GB)
```yaml
data:
  cache_dir: /results/cache/tusz_mmap
  num_workers: 4
  prefetch_factor: 2
training:
  batch_size: 48                  # ~58GB peak
  gradient_accumulation_steps: 1
  mixed_precision: true
model:
  graph:
    edge_similarity_margin: 0.01
resources:
  cpu: 24                         # Default 0.125 too low!
  memory: 98304                   # 96GB RAM
```

## Installation

### Version Lock (DO NOT CHANGE)
- PyTorch==2.5.0+cu124
- CUDA Toolkit==12.4
- mamba-ssm==2.2.5
- causal-conv1d==1.5.2
- torch-geometric==2.6.1
- numpy==1.26.4

### Installation Order
```bash
# 1. Install CUDA 12.4 toolkit
sudo apt-get install -y cuda-toolkit-12-4

# 2. Base environment
make setup

# 3. GPU stack (builds from source)
make setup-gpu

# 4. FLA (optional, for research)
make setup-fla

# 5. Verify
.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"
```

### 🚨 UV Dependency Warning

**NEVER run `uv sync` after GPU setup** - it deletes mamba-ssm/FLA/PyG!

```bash
# ✅ SAFE
uv sync --no-prune

# ❌ DANGEROUS
uv sync

# If packages deleted
make setup-gpu && make setup-fla
```

**Why**: GPU packages require CUDA toolkit, PyTorch, and `--no-build-isolation`. Not in pyproject.toml by design.

## Clinical Specs

### Data Pipeline
1. **Input**: TUH EEG Corpus (19 channels, 10-20 montage)
2. **Preprocessing**: 0.5-120Hz bandpass, 60Hz notch, 256Hz resample
3. **Windowing**: 60s windows, 10s stride (83% overlap)
4. **Normalization**: Per-channel z-score + ±10σ clip

### Channel Order (MUST maintain)
```python
["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
 "Fz", "Cz", "Pz",
 "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
```

### Naming: Use `dev` NOT `val`
- Cache: `cache/tusz_mmap/{train,dev}/`
- Matches TUSZ official naming

### Dataset Strategy
- **Training**: `BalancedSeizureDataset` (oversamples seizures 8%→30%)
- **Validation**: `ValidationDataset` (natural distribution ~8%)
- Standard ML: Train balanced, validate real-world

### Post-Processing
- Hysteresis: τ_on=0.86, τ_off=0.78
- Morphology: Opening(11), Closing(31)
- Duration: 3-600s valid
- Merge: Events within 2s

### Performance Targets
- 10 FA/24h → >95% sensitivity
- 5 FA/24h → >90% sensitivity
- 1 FA/24h → >75% sensitivity

## Development Guidelines

### Code Style
- Python 3.11+, full type hints
- Ruff: 100 line length, 4-space indent
- Imports: stdlib → third-party → first-party (sorted)
- No comments unless requested
- Follow neighboring file patterns

### Testing
- `make t` - Development
- `make ts` - During training (CPU only)
- `make test` - Full coverage pre-commit
- `make test-gpu` - GPU tests (stop training first)

### Environment Variables
```bash
# Debugging
export BGB_NAN_DEBUG=1              # NaN warnings
export BGB_SANITIZE_GRADS=1         # Gradient NaN logging
export SEIZURE_MAMBA_FORCE_FALLBACK=1 # Conv1d fallback

# Data limits
export BGB_SMOKE_TEST=1             # 3 files
export BGB_LIMIT_FILES=50           # Override

# Testing
export BGB_SKIP_GPU_TESTS=1         # Skip GPU tests
```

## Common Issues

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError: mamba_ssm/fla** | UV deleted packages! `make setup-gpu && make setup-fla`. See UV warning above. |
| **Symbol mismatch** | Rebuild mamba-ssm: `INSTALLATION.md#1` |
| **CUDA 12.4 not found** | `sudo apt-get install -y cuda-toolkit-12-4` |
| **WSL2 SIGBUS** | Move cache to ext4 (not Windows drives): `INSTALLATION.md#6` |
| **Zero seizures in batches** | `use_balanced_sampling: true` |
| **NaN losses (RTX 4090)** | `mixed_precision: false` |
| **Edge similarity explosion** | `edge_similarity_margin: 0.01` |

## Modal-Specific

### Actions
- **Production**: `--action schedule-training` (auto-restart every 23h until 100 epochs)
- **Smoke tests**: `--action train` (single run)
- **Cache ops**: `populate-cache`, `check-cache`, `clean_stray_npz.py`

### Verify Auto-Restart
```bash
modal app list
# Should see: 2 tasks (scheduler + train) ✅
```

### Commands
```bash
# Populate cache from S3 to Modal SSD (first-time setup)
modal run --detach deploy/modal/app.py --action populate-cache

# Verify cache health + manifests
modal run deploy/modal/app.py --action check-cache

# Clean stray NPZ files (if check-cache warns)
modal run deploy/modal/clean_stray_npz.py --confirm

# Test CUDA/Mamba
modal run deploy/modal/app.py --action test-mamba

# Smoke test (50 files, ~10min)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/smoke_bimamba.yaml

# Production (auto-restart)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_bimamba.yaml

# Resume after timeout (uses timeout_exit.pt)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume true

# Monitor
modal app list
modal app logs <app-id>
modal app stop brain-go-brr-v2
```

### Modal-Specific Config
```yaml
data:
  persistent_workers: true     # Keeps mmap pages warm
training:
  mid_checkpoint_interval_s: 1800  # Checkpoint every 30min
  mid_epoch_keep: 3                # Keep last 3 mid-epoch checkpoints
```

### Timeout Guard
- Exits at 23h (1h before Modal 24h kill limit)
- Writes `timeout_exit.pt` with full state
- Resume with `--resume true` to continue from timeout checkpoint
- Env: `BGB_WALL_CLOCK_LIMIT_S=82800` (set automatically by Modal)

## Key Documentation

**Installation**: `INSTALLATION.md`
**Training**: `STATUS.md` (current progress)
**Methodology**: `docs/05-training/training-methodology.md`
**Troubleshooting**: `docs/08-operations/troubleshooting.md`

## Performance Metrics

**Local FLA (RTX 4090)**:
- ~9.6h/epoch (4.1h train, 5.5h val)
- 100 epochs: ~40 days
- Cost: $0 (electricity only)

**Modal BiMamba2 (A100-80GB)**:
- ~7-12h/epoch
- Cost: $4.40/hour = $186/epoch
- Status: PAUSED

**Smoke tests**: 5 min (local), 10 min (modal)

---

**Mission**: <1 FA/24h clinical seizure detection with V3 dual-stream architecture 🚀

**Status (v4.0.0)**: BiMamba2 (Modal PAUSED Epoch 6) + FLA (Local Epoch 13/100, patience 4/5)
