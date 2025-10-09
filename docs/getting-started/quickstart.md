# 5-Minute Quickstart

**Last Updated**: October 8, 2025  
**Codebase Version**: v3.9.1 (Validation OOM Fix)  
**Time Required**: ~5 minutes

---

## Goal

Run the **smoke test** to confirm your environment, cache, and model all wire together before launching longer jobs.

---

## Prerequisites

- ✅ GPU available (RTX 4090 for local or Modal A100 for cloud)
- ✅ Environment set up (`make setup && make setup-gpu`)
- ✅ Memory-mapped cache present:
  - Local: `cache/tusz_mmap/{train,dev}` with `_data.npy` / `_labels.npy` pairs and `manifest.json`
  - Modal: `/results/cache/tusz_mmap/{train,dev}` populated via `populate-cache`

> **Need to convert an old NPZ cache?**  
> ```bash
> python scripts/convert_cache_to_mmap.py --source cache/tusz/train --dest cache/tusz_mmap/train
> python scripts/convert_cache_to_mmap.py --source cache/tusz/dev   --dest cache/tusz_mmap/dev
> python -m src scan-cache --cache-dir cache/tusz_mmap/train
> python -m src scan-cache --cache-dir cache/tusz_mmap/dev
> ```

---

## Local Smoke Test (RTX 4090)

1. **Optional debugging flags**
   ```bash
   export BGB_NAN_DEBUG=1          # Extra NaN logging
   # export BGB_SANITIZE_GRADS=1   # Zero/log non-finite grads while debugging
   ```

2. **Run the smoke test**
   ```bash
   make s
   ```
   - Automatically sets `BGB_SMOKE_TEST=1` (3 files, 1 epoch)
   - Uses `configs/local/smoke.yaml` (batch size 8, mixed precision disabled)

3. **Confirm success**
   - BalancedSeizureDataset loads (manifest found)  
   - Logs show `Large grad norm … clipped to 0.5` (expected)  
   - Loss decreases slightly; no NaN/Inf warnings

---

## Modal Smoke Test (A100-80GB)

1. **Verify Modal auth and cache**
   ```bash
   modal token set --token-id ... --token-secret ...
   modal run deploy/modal/app.py --action check-cache   # Confirms mmap cache + manifests
   ```

2. **Run the smoke job (50 files)**
   ```bash
   modal run --detach deploy/modal/app.py \
     --action train \
     --config configs/modal/smoke.yaml
   ```
   - Automatically sets `BGB_LIMIT_FILES=50`, `BGB_NAN_DEBUG=1`, `mixed_precision=true`
   - Startup (~5 min) + 1 epoch (<10 min total)

3. **Monitor**
   ```bash
   modal app logs <app-id>
   ```

---

## Common Issues

### ❌ “Cache directory not found”
Ensure the mmap cache exists:
```bash
python -m src scan-cache --cache-dir cache/tusz_mmap/train
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
```

### ❌ NaN or Inf detected
- Keep `training.gradient_clip: 0.5`
- Confirm cache built after September 2025 preprocessing fixes
- Optional: `export BGB_SANITIZE_GRADS=1` to zero/log offending gradients

### ❌ CUDA OOM
Reduce `training.batch_size` in `configs/local/smoke.yaml` (e.g., 6) or disable additional debugging features.

### ❌ WSL2 dataloader hangs
Set `data.num_workers: 0` in the smoke config.

---

## Next Steps

- ✅ Smoke test passes → move to [Your First Training Run](first-run.md) for full 100‑epoch guidance.
- 🔄 Smoke test fails → check `docs/08-operations/troubleshooting.md` and rerun after fixing the issue.

### Quick Reference

| Command | Purpose |
|---------|---------|
| `make s` | Local smoke (3 files, 1 epoch) |
| `modal run --detach … smoke.yaml` | Modal smoke (50 files, ~10 min) |
| `modal run deploy/modal/app.py --action check-cache` | Validate Modal cache & manifests |
| `python -m src validate configs/local/smoke.yaml` | Confirm config integrity |

**Important**: The smoke test is a gate—only proceed to long training once it completes without NaNs or crashes.
