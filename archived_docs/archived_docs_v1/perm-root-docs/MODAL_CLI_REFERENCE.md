# Modal CLI Reference

**Last Updated:** October 10, 2025 (Modal 1.0 migration)

Quick reference for Modal cloud deployment commands used in this project.

---

## 🚀 Common Commands

### Training (BiMamba2)

```bash
# Smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_bimamba.yaml

# Full training (4667 files, 100 epochs)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml

# Resume from last.pt
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume
```

### Auto-Restart Training (23h intervals)

```bash
# Step 1: Deploy the app (registers scheduled functions)
modal deploy deploy/modal/app.py

# Step 2: Start auto-restart training
modal run --detach deploy/modal/app.py --action schedule-training \
  --config configs/modal/train_bimamba.yaml

# Monitor logs
modal app logs brain-go-brr-v2

# Stop auto-restart
modal app stop brain-go-brr-v2
```

### Cache Management

```bash
# ONE-TIME: Populate cache from S3 to Modal SSD
modal run --detach deploy/modal/app.py --action populate-cache

# Check cache status
modal run deploy/modal/app.py::check_cache

# Clean contaminated cache
modal run deploy/modal/app.py --action clean-cache
```

### Testing

```bash
# Test Mamba CUDA kernels
modal run deploy/modal/app.py --action test-mamba
```

---

## 📋 Monitoring & Management

### View Running Apps

```bash
# List all running apps
modal app list

# Show app details
modal app logs <app-id>

# Stop app
modal app stop <app-id>
```

### Storage Management

```bash
# List volumes
modal volume ls

# View volume contents
modal volume ls brain-go-brr-results

# Download file from volume
modal volume get brain-go-brr-results /results/v3_full_training/checkpoints/last.pt
```

---

## 🔧 Modal 1.0 Migration Notes

**Deprecated (will be removed Feb 2025):**
- `concurrency_limit=N` → Use `max_containers=N`

**Why it matters:**
- Modal 1.0 renamed autoscaling parameters for clarity
- Old code still works (deprecation warning only)
- Should update to avoid breaking changes in Feb 2025

**Updated in our code:**
- ✅ `deploy/modal/app.py:1141` - Changed `concurrency_limit=1` → `max_containers=1`

---

## ⚠️ Important Flags

### --detach

**CRITICAL:** Use `--detach` for long-running jobs (>5 min)!

```bash
# ✅ CORRECT (for long training runs)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# ❌ WRONG (terminal must stay open entire training run)
modal run deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml
```

**What `--detach` does:**
- Returns app ID immediately
- Training continues in cloud even if terminal closes
- View logs with `modal app logs <app-id>`

### --resume

**Usage:** Resume training from last checkpoint

```bash
# ✅ Boolean flag (no value needed)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume

# ❌ WRONG (don't use "true" value)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume true
```

**How it works:**
- Loads `/results/<output_dir>/checkpoints/last.pt`
- Falls back to `best.pt` if `last.pt` missing
- Priority: `mid_epoch_*.pt` > `last.pt` > `best.pt`

---

## 🎯 Workflow Examples

### Fresh Training Start

```bash
# 1. Ensure cache exists (one-time)
modal run deploy/modal/app.py::check_cache

# 2. Start training
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml

# 3. Get app ID from output
# Example: ap-ik2xwlXmuQMvPyhSfrZJfi

# 4. Monitor logs
modal app logs ap-ik2xwlXmuQMvPyhSfrZJfi
```

### Resume After Timeout

```bash
# Training timed out at 23h, now resume
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume

# Checkpoint priority:
# 1. mid_epoch_002_001234.pt (if exists - crash recovery)
# 2. last.pt (normal resume)
# 3. best.pt (fallback)
```

### Auto-Restart Setup (Hands-Free Training)

```bash
# 1. Deploy app (registers scheduled functions)
modal deploy deploy/modal/app.py

# 2. Start auto-restart
modal run --detach deploy/modal/app.py --action schedule-training \
  --config configs/modal/train_bimamba.yaml

# 3. Monitor progress
modal app logs brain-go-brr-v2

# 4. Check status
modal app list

# 5. Stop when done
modal app stop brain-go-brr-v2
```

**Timeline:**
```
T=0h:       Run 1 starts
T=22h50m:   Timeout guard → saves checkpoint → exits
T=23h:      Run 2 starts (10 min idle gap)
T=45h50m:   Timeout guard → saves checkpoint → exits
T=46h:      Run 3 starts (10 min idle gap)
...repeat until 100 epochs or manual stop
```

---

## 💾 Storage Architecture

**S3 Bucket:** `brain-go-brr-eeg-data-20250919`
- `/tusz/edf/` → Raw EDF files (266GB)
- `/cache/tusz_mmap/` → Preprocessed NPY cache (507GB)

**Modal Mounts:**
- `/data/` → S3 bucket (`tusz/edf/`) - read-only
- `/results/` → Modal SSD volume - read/write

**Modal SSD Volume:** `brain-go-brr-results`
- `/results/cache/tusz_mmap/` → NPY cache (copied from S3 once)
- `/results/smoke/` → Smoke test outputs
- `/results/v3_full_training/` → Full training outputs

**Key insight:** Cache lives on Modal SSD (not S3!) for fast mmap access.

---

## 🐛 Debugging

### View Full Logs

```bash
# Stream logs in real-time
modal app logs <app-id>

# View logs from specific time
modal app logs <app-id> --since 1h

# Save logs to file
modal app logs <app-id> > training.log
```

### Check GPU Usage

```bash
# Modal dashboard shows GPU utilization
# Visit: https://modal.com/apps/<your-workspace>/<app-name>/<app-id>
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `concurrency_limit` deprecation | Change to `max_containers` in app.py:1141 |
| `Got unexpected extra argument (true)` | Use `--resume` (no value), not `--resume true` |
| Training hangs during cache build | Normal for first run (~45 min), use `BGB_DISABLE_TQDM=1` |
| OOM during validation | Fixed in v3.9.2 with disk-backed validation |

---

## 📚 Related Documentation

- **Modal Docs:** https://modal.com/docs
- **Migration Guide:** https://modal.com/docs/guide/modal-1-0-migration
- **Our Training Docs:** `docs/05-training/modal.md`
- **Auto-Restart Strategy:** `docs/archive_v2/MODAL_AUTO_RESTART_STRATEGY.md`

---

## ✅ Quick Checklist

**Before first training:**
- [ ] Cache populated: `modal run deploy/modal/app.py::check_cache`
- [ ] W&B secret configured: `modal secret create wandb-secret`
- [ ] AWS S3 secret configured: `modal secret create aws-s3-secret`

**For each training run:**
- [ ] Use `--detach` for long runs
- [ ] Use correct config path (`configs/modal/...`)
- [ ] Check app ID and save it
- [ ] Monitor logs to confirm startup

**For auto-restart:**
- [ ] Deploy first: `modal deploy deploy/modal/app.py`
- [ ] Start scheduled training: `--action schedule-training`
- [ ] Verify in dashboard: https://modal.com/apps
- [ ] Remember to stop when done: `modal app stop brain-go-brr-v2`

---

**Last Updated:** October 10, 2025 - Modal 1.0 migration complete ✅
