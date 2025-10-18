# Modal Deployment - v4.0.0

## 🎯 Overview

Modal cloud deployment for Brain-Go-Brr with dual production stacks:
- **BiMamba2** (baseline): State-space model on A100-80GB
- **FLA** (research): Gated DeltaNet on A100-80GB

Both use Modal SSD volumes for cache (memory-mapped NPY format) and persistent training outputs.

## 📁 Storage Architecture

### Modal SSD Volume (`brain-go-brr-results`)
**Purpose**: Fast local storage for cache AND training outputs

```
/results/                          # Modal SSD volume
├── cache/tusz_mmap/               # Memory-mapped NPY cache (529GB)
│   ├── train/                     # 4,667 files (data + labels)
│   │   ├── *_data.npy             # Uncompressed for mmap
│   │   ├── *_labels.npy           # Uncompressed for mmap
│   │   └── manifest.json          # BalancedSeizureDataset index
│   └── dev/                       # 1,832 files (data + labels)
│       ├── *_data.npy             # Uncompressed for mmap
│       ├── *_labels.npy           # Uncompressed for mmap
│       └── manifest.json          # ValidationDataset index
├── smoke/                         # Smoke test outputs
│   ├── checkpoints/
│   ├── tensorboard/
│   └── wandb/
└── train/                         # Full training outputs
    ├── checkpoints/
    ├── tensorboard/
    └── wandb/
```

### S3 Bucket (`brain-go-brr-eeg-data-20250919`)
**Purpose**: Cold storage for raw EDF files (read-only mount)

```
/data/edf/                         # S3 mount (raw data)
├── train/                         # Training EDF files
└── dev/                           # Validation EDF files
```

**Cache Strategy**: Cache copied from S3 to Modal SSD once via `populate-cache`, then reused forever.

## 🚀 Commands

### One-Time Setup
```bash
# Copy cache from S3 to Modal SSD (run ONCE, takes ~6 hours)
modal run --detach deploy/modal/app.py --action populate-cache

# Verify cache completeness
modal run deploy/modal/app.py --action check-cache

# Test Mamba CUDA kernels
modal run deploy/modal/app.py --action test-mamba
```

### Smoke Tests (Fast Validation)
```bash
# BiMamba2 smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml

# FLA smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml
```

### Full Training

**🚨 CRITICAL**: Choose the RIGHT action!

| Use Case | Action | Auto-Restart? | When to Use |
|----------|--------|---------------|-------------|
| **Full Training (100 epochs)** | `schedule-training` | ✅ YES | **ALWAYS use for production!** |
| Smoke test | `train` | ❌ NO | Quick tests (<1 hour) |
| One-off experiment | `train` | ❌ NO | Single run experiments |

```bash
# BiMamba2 full training - Auto-Restart (CORRECT ✅)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml

# FLA full training - Auto-Restart (CORRECT ✅)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_fla.yaml

# Manual mode (ONLY for experiments, NOT production!)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml --resume
```

**Auto-Restart Timeline**:
- T=0h: Run 1 starts
- T=22h 50m: Timeout guard triggers → checkpoint saved → exits
- T=23h: Period triggers → Run 2 starts (10 min idle)
- Repeats until 100 epochs complete or manually stopped

### Monitoring
```bash
# List running apps
modal app list

# Stream logs
modal app logs brain-go-brr-v2 --follow

# Stop scheduled training
modal app stop brain-go-brr-v2
```

### Diagnostics
```bash
# Inspect volume contents
modal run deploy/modal/inspect_volume.py

# Check cache readiness
modal run deploy/modal/app.py --action check-cache
```

## 📝 Files in This Directory

| File | Purpose | Type |
|------|---------|------|
| `app.py` | Main Modal deployment script | Production |
| `inspect_volume.py` | Volume diagnostics tool | Utility |
| `patch_mamba_pr708.py` | XID 31 GPU crash fix (PR #708) | Build |
| `README.md` | This file | Documentation |

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Cache missing | Run `populate-cache` action once |
| Training stuck | Check `modal app logs` for hangs |
| GPU XID 31 crash | `patch_mamba_pr708.py` applied at build time |
| Out of memory | Reduce `batch_size` in config |

## 📚 See Also

- `CLAUDE.md` - Full project overview and command reference
- `docs/05-training/modal.md` - Detailed Modal training guide
- `configs/README.md` - Configuration file documentation

