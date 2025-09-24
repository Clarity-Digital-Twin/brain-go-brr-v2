# Deployment Documentation

## Overview

Brain-Go-Brr v2 supports multiple deployment targets, optimized for different use cases.

## Deployment Options

### 1. Local Development
- **Location**: `local/`
- **Purpose**: Development, debugging, small-scale experiments
- **Hardware**: RTX 4090 or similar consumer GPU
- **Model switch**: set `model.architecture: tcn` (baseline) or `v3` (dual‑stream)
- **Performance**: environment‑dependent (seconds per batch on FP32); use smoke config first

### 2. Modal Cloud (Production)
- **Location**: `modal/`
- **Purpose**: Full training runs, production deployment
- **Hardware**: A100‑80GB GPU
- **Model switch**: `model.architecture: v3` by default in modal configs
- **Performance**: FP16 with vectorized GNN yields sub‑second to few‑seconds per batch (dataset/config dependent)
- **Costs**: usage‑based; monitor with Modal dashboard

### 3. Operations
- **Location**: `operations/`
- **Purpose**: Monitoring, maintenance, troubleshooting

## Quick Start

### Local Training
```bash
# Smoke test
python -m src train configs/local/smoke.yaml

# Full training (if you have 100+ hours)
python -m src train configs/local/train.yaml
```

### Modal Training (Recommended)
```bash
# Deploy and run on A100
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml

# Monitor progress
modal app logs <app-id>
```

## Key Documents

- **[modal/storage.md](modal/storage.md)**: Modal storage architecture
- **[modal/PERFORMANCE_OPTIMIZATION.md](modal/PERFORMANCE_OPTIMIZATION.md)**: A100 optimization details
- **[modal/deploy.md](modal/deploy.md)**: Modal deployment guide
- **[troubleshooting.md](troubleshooting.md)**: Common issues and fixes
- **[operations/smoke-tests.md](operations/smoke-tests.md)**: Fast pipeline and unit smoke tests

## Performance Notes

- Vectorized GNN + static Laplacian PE (v3) reduces CPU overhead dramatically.
- A100 (FP16) is significantly faster than local FP32; exact speed depends on cache, I/O, and batch size.
- Use `configs/modal/smoke.yaml` to validate environment, then `configs/modal/train.yaml`.

## Current Status

✅ **Modal A100 training optimized**
- Mixed precision (FP16) enabled
- Cache on Modal SSD (`/results/cache/tusz`, not S3)
- W&B logging configured in configs
