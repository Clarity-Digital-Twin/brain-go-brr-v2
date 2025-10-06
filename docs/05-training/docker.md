# Docker Training Guide

**Last Updated**: October 4, 2025 (v3.6.1)
**Status**: ✅ Production Ready

---

## Overview

Docker provides GPU-accelerated local training in isolated containers, matching the exact environment used in Modal cloud deployment. This guide covers the V2 implementation with proper volume mounting strategy.

## Quick Start

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

---

## Volume Mount Strategy

Docker V2 mounts **THREE directories** to match local/Modal environments exactly:

### 1. Original EDFs (Read-Only, 109GB)

```yaml
- ./data_ext4:/app/data_ext4:ro
```

**Purpose**: Enables smoke tests with EEGWindowDataset
**Used by**: `data_dir: data_ext4/tusz/edf` in configs
**Contains**: Original EDF files + CSV labels (train/dev/eval)

### 2. Preprocessed Cache (Read-Only, 449GB)

```yaml
- ./cache/tusz:/app/cache/tusz:ro
```

**Purpose**: Primary data source for training
**Used by**: `cache_dir: cache/tusz` in configs
**Contains**:
- `train/` - 4667 NPZ files + `manifest.json`
- `dev/` - 1832 NPZ files + `manifest.json`

### 3. Results (Read-Write)

```yaml
- ./results:/app/results:rw
```

**Purpose**: Checkpoints, logs, outputs
**Written by**: Training loop, evaluation, etc.

**Total mounted**: ~560GB (all read-only except results)

---

## Dataset Strategy

Docker uses the SAME dataset strategy as local/Modal:

### Smoke Test (3 files)

- **Dataset**: `EEGWindowDataset` (original EDFs)
- **Requires**: Both `data_ext4/` AND `cache/` mounts
- **Environment**: `BGB_SMOKE_TEST=1` (auto-limits to 3 files)
- **Duration**: ~5 minutes

### Integration Test (50 files)

- **Dataset**: `EEGWindowDataset` (original EDFs)
- **Requires**: Both `data_ext4/` AND `cache/` mounts
- **Environment**: `BGB_LIMIT_FILES=50`
- **Duration**: ~60 minutes

### Full Training (4667 files)

- **Dataset**: `BalancedSeizureDataset` (preprocessed cache)
- **Requires**: Only `cache/` mount (but both are mounted)
- **Uses**: `train/manifest.json` for instant loading
- **Duration**: ~100 hours (RTX 4090)

---

## Configuration

### Smoke Test Service

```yaml
services:
  smoke-test:
    image: brain-go-brr:latest
    command: train configs/local/smoke.yaml
    environment:
      - BGB_SMOKE_TEST=1
      - BGB_NAN_DEBUG=1
    volumes:
      - ./data_ext4:/app/data_ext4:ro
      - ./cache/tusz:/app/cache/tusz:ro
      - ./results:/app/results:rw
      - ./configs:/app/configs:ro
```

### Integration Test Service

```yaml
services:
  integration-test:
    image: brain-go-brr:latest
    command: train configs/local/smoke.yaml
    environment:
      - BGB_LIMIT_FILES=50
      - BGB_NAN_DEBUG=1
    volumes:
      - ./data_ext4:/app/data_ext4:ro
      - ./cache/tusz:/app/cache/tusz:ro
      - ./results:/app/results:rw
      - ./configs:/app/configs:ro
```

### Full Training Service

```yaml
services:
  train:
    image: brain-go-brr:latest
    command: train configs/local/train.yaml
    environment:
      - BGB_NAN_DEBUG=1
    volumes:
      - ./data_ext4:/app/data_ext4:ro
      - ./cache/tusz:/app/cache/tusz:ro
      - ./results:/app/results:rw
      - ./configs:/app/configs:ro
```

---

## File Dependency Matrix

| Use Case | Dataset | data_dir (EDFs) | cache_dir (NPZ) | manifest.json |
|----------|---------|-----------------|-----------------|---------------|
| **Smoke (3 files)** | EEGWindowDataset | ✅ REQUIRED | ✅ REQUIRED | ❌ |
| **Integration (50 files)** | EEGWindowDataset | ✅ REQUIRED | ✅ REQUIRED | ❌ |
| **Full Training** | BalancedSeizureDataset | ❌ | ✅ REQUIRED | ✅ REQUIRED |
| **Validation** | ValidationDataset | ❌ | ✅ REQUIRED | ✅ REQUIRED |

---

## Troubleshooting

### Split Directory Not Found

**Error**: `Split directory not found: data_ext4/tusz/edf/train`

**Cause**: Missing `data_ext4/` volume mount

**Solution**: Ensure docker-compose.yml includes:
```yaml
- ./data_ext4:/app/data_ext4:ro
```

> **Why the original container failed:** the first revision mounted only the preprocessed cache, assuming EDFs were unnecessary once NPZ files existed. The smoke test still uses `EEGWindowDataset`, which reads EDFs directly, so the container aborted as soon as it tried to locate `data_ext4/tusz/edf/train`. Mounting all three directories (EDFs + cache + results) restores parity with the local and Modal setups.

### Slow Training Performance

**Symptoms**: Training slower than local (outside Docker)

**Causes**:
1. Docker Desktop resource limits too low
2. Volume mounts on slow storage
3. Not using native Linux (WSL2 adds overhead)

**Solutions**:
- Increase Docker Desktop CPU/RAM allocations
- Move data to SSD/NVMe
- Use native Linux if possible

### Cache Not Found

**Error**: `Cache directory not found: cache/tusz/train`

**Cause**: Cache not built or mount path wrong

**Solution**:
1. Build cache locally first (see below)
2. Verify mount path matches config

### Building Cache for Docker

```bash
# Run cache build OUTSIDE Docker (faster)
make setup
make setup-gpu
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/train \
  --cache-dir cache/tusz/train \
  --split train

python -m src build-cache \
  --data-dir data_ext4/tusz/edf/dev \
  --cache-dir cache/tusz/dev \
  --split dev
```

---

## Monitoring

### View Logs

```bash
# Stream logs
docker compose logs -f smoke-test

# View specific service
docker compose logs train
```

### Monitor GPU Usage

```bash
# From host (outside Docker)
watch -n 1 nvidia-smi

# Inside container
docker compose exec train nvidia-smi
```

### TensorBoard

```bash
# Start TensorBoard (maps to host port 6006)
docker compose up tensorboard

# Open browser
open http://localhost:6006
```

### Weights & Biases

Docker automatically uses W&B if you set:

```yaml
environment:
  - WANDB_API_KEY=${WANDB_API_KEY}
  - WANDB_ENTITY=your-team
  - WANDB_PROJECT=brain-go-brr
```

Then in your shell:
```bash
export WANDB_API_KEY=your_key_here
docker compose up train
```

---

## Validation Checklist

Before running Docker training:

```bash
# 1. Verify data_ext4/ exists
ls -lh data_ext4/tusz/edf/train/ | head -5
# Should show: 4667 directories (patients)

# 2. Verify cache exists with manifests
ls -lh cache/tusz/train/manifest.json
ls -lh cache/tusz/dev/manifest.json
# Should show: 27MB and 13MB files

# 3. Verify NPZ files
ls -1 cache/tusz/train/*.npz | wc -l  # Should be 4667
ls -1 cache/tusz/dev/*.npz | wc -l    # Should be 1832

# 4. Check disk space for volume mounts
df -h .
# Need: 560GB available (109GB data_ext4 + 449GB cache + headroom)
```

---

## Success Criteria

✅ **Docker V2 is working when:**

1. `docker compose up smoke-test` completes without "Split directory not found"
2. Smoke test uses 3 files (BGB_SMOKE_TEST=1)
3. Full training uses BalancedSeizureDataset (61,616 windows)
4. Validation uses ValidationDataset with manifest (instant load)
5. Performance matches local training (99%+ speed)
6. No adhoc hacks or workarounds

---

## Comparison: Docker vs Local vs Modal

| Feature | Local | Docker | Modal |
|---------|-------|--------|-------|
| **GPU** | RTX 4090 | RTX 4090 | A100-80GB |
| **Batch Size** | 12 | 12 | 32 |
| **Mixed Precision** | No (FP32) | No (FP32) | Yes (FP16) |
| **Cache Location** | cache/tusz/ | cache/tusz/ | /results/cache/tusz/ |
| **Data Location** | data_ext4/ | data_ext4/ | /data/edf/ |
| **Isolation** | None | Full | Full |
| **Setup Time** | Minutes | Seconds | 10-15 min |

---

## Related Documentation

- Volume mounting: `docs/archive/DOCKER_IMPLEMENTATION_PLAN_V2.md`
- Local training: `docs/05-training/local.md`
- Modal training: `docs/05-training/modal.md`
- Smoke tests: `docs/05-training/smoke-tests.md`

---

**Key Takeaway**: Docker V2 mounts BOTH original EDFs and preprocessed cache to match local/Modal environments exactly. This enables full parity across all training modes.
