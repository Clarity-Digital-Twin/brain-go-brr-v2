# 5-Minute Quickstart

**Last Updated**: October 1, 2025
**Codebase Version**: v3.4.1
**Time Required**: 5 minutes

---

## What You'll Do

Run a **smoke test** to verify your installation and see the model training in action. This uses 3 files and runs for 1 epoch (~5 minutes).

## Prerequisites

- ✅ RTX 4090 or A100 GPU
- ✅ Installation complete (see `../01-installation/env-setup.md`)
- ✅ Cache directory exists: `cache/tusz/train/` with NPZ files

---

## Local Smoke Test (RTX 4090)

### Step 1: (Optional) Enable NaN Debugging

```bash
export BGB_NAN_DEBUG=1         # Show NaN warnings if they occur
# export BGB_SANITIZE_GRADS=1  # Debug: zero-out non-finite gradients while investigating
```

> Gradient clipping (`training.gradient_clip: 0.5`) is always active and provides the primary NaN protection. Enable `BGB_SANITIZE_GRADS` only when you want extra debugging visibility.

### Step 2: Run Smoke Test

```bash
make s
```

**What this does**:
- Sets `BGB_SMOKE_TEST=1` (limits to 3 files)
- Runs 1 epoch with batch_size=12
- Uses config: `configs/local/smoke.yaml`

### Step 3: Verify Output

You should see:

```
[INFO] Using BalancedSeizureDataset (manifest found)
[INFO] Training samples: 3 files
[INFO] Batch 1/X: loss=0.693, lr=1.0e-04
[INFO] Large grad norm: 5.72e+00 (clipped to 0.5)  ← NORMAL!
[INFO] Epoch 1 complete - avg loss: 0.65
```

✅ **Success indicators**:
- No `NaN loss detected` errors
- `Large grad norm` messages are normal (gradient clipping working)
- Loss decreasing slightly
- Checkpoint saved (if enabled)

---

## Modal Cloud Smoke Test (A100-80GB)

### Step 1: Verify Modal Setup

```bash
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

### Step 2: Run Smoke Test

```bash
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

**What this does**:
- Spins up A100-80GB instance
- Runs 1 epoch with 50 files
- Enables `BGB_NAN_DEBUG=1` for extra logging

### Step 3: Monitor

```bash
# In another terminal
modal app logs <app-id>
```

Expected runtime: ~5-10 minutes

---

## Common Issues

### ❌ "Cache directory not found"

**Solution**: Build cache first:
```bash
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train
python -m src scan-cache --cache-dir cache/tusz/train
```

### ❌ "NaN loss detected at batch X"

**Solution**:
1. Check `BGB_SANITIZE_GRADS=1` is set
2. Verify cache was built **after** September 26 (preprocessing fix)
3. If cache is old: delete and rebuild

```bash
rm -rf cache/tusz/train
make setup-cache  # Rebuilds from scratch
```

### ❌ "GPU OOM (out of memory)"

**Solution**: Reduce batch size in `configs/local/smoke.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 12
```

### ❌ WSL2: Dataloader hangs

**Solution**: Set workers to 0 in `configs/local/smoke.yaml`:
```yaml
data:
  num_workers: 0
```

---

## What's Next?

### ✅ Smoke test passed → Ready for full training

**Local (RTX 4090)**:
```bash
export BGB_NAN_DEBUG=1          # Optional: extra NaN logging
# export BGB_SANITIZE_GRADS=1   # Optional: debugging helper
make train-local                # 100 epochs, ~200-300 hours
```

**Modal (A100)**:
```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

### 📚 Learn More

- **Full training walkthrough**: `first-run.md`
- **Understanding logs**: `understanding-logs.md`
- **Local training guide**: `../05-training/local.md`
- **Modal training guide**: `../05-training/modal.md`
- **Troubleshooting**: `../08-operations/troubleshooting.md`

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `make s` | Local smoke test (3 files, 1 epoch) |
| `make train-local` | Full local training (100 epochs) |
| `modal run deploy/modal/app.py --action test-mamba` | Test Mamba CUDA |
| `python -m src validate configs/local/smoke.yaml` | Validate config |

**Environment variables**:
- `BGB_NAN_DEBUG=1` - Show NaN warnings (recommended)
- `BGB_SANITIZE_GRADS=1` - Optional debugging helper (see gradient protection guide)
- `BGB_SMOKE_TEST=1` - Limit to 3 files (auto-set by `make s`)

---

**Status**: ✅ If smoke test completes without NaN errors, you're ready for production training!
