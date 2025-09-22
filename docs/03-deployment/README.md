# Deployment Documentation

## Overview

Brain-Go-Brr v2 supports multiple deployment targets, optimized for different use cases.

## Deployment Options

### 1. Local Development
- **Location**: `local/`
- **Purpose**: Development, debugging, small-scale experiments
- **Hardware**: RTX 4090 or similar consumer GPU
- **Performance**: ~3s/batch with FP32

### 2. Modal Cloud (Production)
- **Location**: `modal/`
- **Purpose**: Full training runs, production deployment
- **Hardware**: A100-80GB GPU
- **Performance**: ~5s/batch with FP16
- **Cost**: ~$319 for 100 epochs

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
  --config configs/modal/train_a100.yaml

# Monitor progress
modal app logs <app-id>
```

## Key Documents

- **[modal/storage.md](modal/storage.md)**: Modal storage architecture
- **[modal/PERFORMANCE_OPTIMIZATION.md](modal/PERFORMANCE_OPTIMIZATION.md)**: A100 optimization details
- **[modal/deploy.md](modal/deploy.md)**: Modal deployment guide
- **[troubleshooting.md](troubleshooting.md)**: Common issues and fixes

## Performance Summary

| Environment | GPU | Batch Size | Precision | Speed | Cost |
|-------------|-----|------------|-----------|-------|------|
| Local | RTX 4090 | 32 | FP32 | ~3s/batch | Electricity |
| Modal | A100-80GB | 128 | FP16 | ~5s/batch | $3.19/hr |

## Current Status

✅ **Modal A100 training optimized and running**
- Mixed precision enabled (FP16)
- Batch size 128 (utilizing full VRAM)
- Cache on Modal SSD (not S3)
- W&B integration working
- Expected completion: ~100 hours ($319)