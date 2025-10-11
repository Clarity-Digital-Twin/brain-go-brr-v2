# Smoke Tests

**Purpose**: Fast end-to-end pipeline validation (1-5 minutes)
- **NOT for training dynamics** - just validates no crashes
- **Industry standard**: Minimal data to exercise all code paths

## Smoke Test Standard (3 Files)

All platforms use **3 files** for consistent fast validation:

| Platform | Command | Duration | Files |
|----------|---------|----------|-------|
| **Local** | `make s` | ~5 min | 3 files |
| **Docker** | `docker compose up smoke-test` | ~5 min | 3 files |
| **Modal** | `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml` | ~5 min | 50 files* |

*Modal smoke uses 50 files due to cloud startup overhead - still <10 min

## Local Smoke Test

```bash
# Quick smoke (3 files, ~5 min)
make s

# Or manually:
export BGB_SMOKE_TEST=1 BGB_NAN_DEBUG=1
python -m src train configs/local/smoke.yaml
```

**How it works**:
- `BGB_SMOKE_TEST=1` automatically limits to 3 files (hardcoded default)
- No need to set `BGB_LIMIT_FILES` - it's implied
- Uses `configs/local/smoke.yaml` (1 epoch, batch_size=8)

## Docker Smoke Test

```bash
# Fast smoke (3 files, ~5 min) - RECOMMENDED
docker compose up smoke-test

# Deeper validation (50 files, ~60 min)
docker compose up integration-test
```

**Environment variables set automatically**:
- `BGB_SMOKE_TEST=1` - enables 3-file limit
- `BGB_NAN_DEBUG=1` - enables NaN warnings

## Modal Smoke Test

```bash
# Smoke test (50 files, ~10 min)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
# Optional: validate cache first
modal run deploy/modal/app.py --action check-cache
```

Modal uses 50 files instead of 3 due to:
- Cloud startup overhead (~2-3 min)
- Worth validating more data once environment is up

## Environment Variables

**Smoke mode** (automatic 3-file limit):
```bash
BGB_SMOKE_TEST=1  # Enables smoke shortcuts (3 files default)
```

**Custom file limits** (override default):
```bash
BGB_LIMIT_FILES=10  # Override to use N files
BGB_SMOKE_TEST=1    # Still enables smoke shortcuts
```

**Other useful flags**:
- `BGB_NAN_DEBUG=1` - Enable NaN warnings (recommended)
- `BGB_DISABLE_TQDM=1` - Disable progress bars
- `BGB_FORCE_MANIFEST_REBUILD=1` - Rebuild cache manifest

## What to Look For

✅ **Success indicators**:
- No crashes during data loading
- Model forward/backward pass succeeds
- Gradients are finite (check logs for NaN warnings)
- Checkpoints saved if enabled
- W&B/TensorBoard logs created

❌ **Failure indicators**:
- Dataset loading crashes
- NaN/Inf losses or gradients
- CUDA out-of-memory errors
- File permission issues

## Troubleshooting

**Empty manifest in smoke mode**:
```bash
# Training auto-falls back to EEGWindowDataset
# Or manually rebuild:
python -m src scan-cache --cache-dir cache/tusz_mmap/train
```

**WSL2 multiprocessing issues**:
```yaml
# In configs/local/smoke.yaml
data:
  num_workers: 0  # Use 0 for WSL2 stability
```

**RTX 4090 stability**:
```yaml
# In configs/local/smoke.yaml
training:
  mixed_precision: false  # Disable for stability
```
