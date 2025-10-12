# Checkpoint Strategy Guide

## Overview
Professional checkpoint strategy for long-running EEG training on both local (RTX 4090) and Modal (A100).

## Checkpoint Sizes
- **V3 architecture**: ~195MB per checkpoint
- **Storage overhead**: Minimal (< 2GB total with rotation)
- **Professional practice**: YES, this is standard for deep learning

## Local Training (RTX 4090)

### Configuration (RECOMMENDED - v3.6+)
Set mid-epoch checkpointing in your config YAML:
```yaml
training:
  mid_checkpoint_interval_s: 1800  # Save every 30 minutes
  mid_epoch_keep: 3                # Keep last 3 snapshots (rotating)
```

### Environment Variables (LEGACY - Deprecated)
For backward compatibility only:
```bash
# DEPRECATED: Use config fields above instead
export BGB_MID_EPOCH_MINUTES=30  # Save every 30 minutes
export BGB_MID_EPOCH_KEEP=5      # Keep last 5 snapshots (rotating)
```
**Note:** Config fields take precedence over env vars.

### What Gets Saved
1. **Every epoch**: `last.pt` (for resume; since Oct 10 2025 the metadata stores `epoch+1`, so resumes jump to the next epoch instead of replaying the previous one)
2. **Best model**: `best.pt` (when validation improves)
3. **Mid-epoch**: `mid_epoch_XXX_YYYYYY.pt` (every 30 min, keeps 5)
4. **Periodic**: `checkpoint_epoch_XXX.pt` (every epoch now)
5. **Timeout guard**: `timeout_exit.pt` (written when the Modal wall-clock guard triggers at ~23 h)
- **v3.11.0 metadata**: All checkpoint types persist `global_step`. Mid-epoch and timeout snapshots also carry `batch_idx` and `StatefulDataLoader.state_dict()` so resumes continue at the exact batch with the correct warmup state and W&B step counters.

### Storage Requirements
- **Per epoch**: ~400MB (last + best)
- **Mid-epoch**: ~1GB rolling (5 × 195MB)
- **Total**: < 2GB active storage

### Resume Command
```bash
# Resume from latest checkpoint (BiMamba2 baseline)
.venv/bin/python -m src train configs/local/train_bimamba.yaml --resume

# Or FLA research variant
.venv/bin/python -m src train configs/local/train_fla.yaml --resume

# Automatically picks up from:
# 1. Latest mid-epoch checkpoint (if exists)
# 2. timeout_exit.pt (if present from timeout guard)
# 3. last.pt (if no mid-epoch/timeout snapshots)
# 4. Fresh start (if no checkpoints)

# On resume you should see logs such as:
# [RESUME] ✅ Exact mid-epoch resume: epoch 3, batch 2527, global_step 2527
# [WARMUP] Batch 2527 focal_gamma=2.000  (no schedule reset)
```

## Modal Training (A100)

### Configuration (Already Optimal)
- Saves every epoch (`checkpoint_interval: 1`)
- Persistence volume handles all checkpoints
- Automatic resume with `--resume`
- Timeout guard writes `timeout_exit.pt` roughly one hour before Modal’s 24 h limit; relaunch with `--resume` to continue.

### Modal Commands
```bash
# Start training (with automatic checkpointing)
modal run --detach deploy/modal/app.py \
    --action train \
    --config configs/modal/train_bimamba.yaml

# Resume from checkpoint
modal run --detach deploy/modal/app.py \
    --action train \
    --config configs/modal/train_bimamba.yaml \
    --resume
```

### Atomic Save Mechanics (v3.9.1)

- `save_checkpoint()` writes to `<name>.pt.tmp`, calls `os.fsync()`, then atomically renames to `<name>.pt`. Partial or corrupted files cannot appear even if Modal terminates mid-save.
- Checkpoints include AMP scaler state and RNG seeds for Python, NumPy, torch CPU, and torch CUDA so resumes reproduce the exact batch order after timeouts.
- `load_checkpoint()` handles legacy checkpoints gracefully—missing scaler/RNG fields trigger a warning but training continues.

### 2025 Incident Fixes (Now Baked In)

- **Dynamic buffer mismatch** (`gnn.last_valid_pe`): Graph positional encoding now registers a placeholder tensor so every checkpoint contains the buffer. Loader skips legacy keys safely before `load_state_dict()`, preventing the “Unexpected key gnn.last_valid_pe” crash described in the 2025-10 checkpoint audit.
- **RNG device mismatch**: Checkpoint restores leave RNG tensors on CPU before `torch.set_rng_state`, fixing the `torch.ByteTensor` device error when resuming on CUDA. See `tests/unit/train/test_checkpoint_rng_device.py` for regression coverage.
- **StatefulDataLoader parity**: Mid-epoch and timeout snapshots persist the loader state (`dataloader_state_dict`) plus `global_step`/`batch_idx` so resumes continue at the exact batch with correct warmup scheduling—no more 1–2 hour replays after Modal timeouts.
- **Resume priority order**: Loader always prefers newest `mid_epoch_*.pt`, then `timeout_exit.pt`, then `last.pt`, mirroring the “Clear Path Forward” mitigation plan. This guarantees deterministic recovery after interruptions.

### Legacy checkpoints (pre-Oct 10 2025)

- Checkpoints written before Oct 10 saved the **current** epoch number. When loaded, the training loop restarted the same epoch, so the first resume after the fix will replay Epoch 2 (~14 h, ~$56) before writing a corrected `last.pt`.
- Let that resume complete—once the new checkpoint lands, subsequent restarts use the updated `epoch+1` metadata and continue without replay.
- If you must avoid the replay, delete the old `mid_epoch_*.pt` and `last.pt` files and relaunch without `--resume`, but you will lose all progress since the start of the last completed epoch.

## Best Practices

### DO's
✅ **Always set mid-epoch checkpointing** for local training
✅ **Keep 3-5 mid-epoch checkpoints** (rotating)
✅ **Save every epoch** (`checkpoint_interval: 1`)
✅ **Use --resume flag** when restarting (after Modal timeout guard, this will load `timeout_exit.pt` automatically)

### DON'Ts
❌ Don't use `checkpoint_interval > 1` for long training
❌ Don't disable mid-epoch saves for multi-hour epochs
❌ Don't worry about storage (< 2GB is trivial)

## Storage Management

### Clean Old Results
```bash
# Remove outdated smoke test results
rm -rf results/smoke*

# Remove failed/incomplete runs
rm -rf results/v2.6_*

# Keep only active training
# - results/v3_full_training/ (current)
# - results/{experiment_name}/ (new runs)
```

### Checkpoint Files
```
results/v3_full_training/checkpoints/
├── best.pt                        # Best validation score
├── last.pt                        # Latest epoch (resume point, stores epoch+1)
├── mid_epoch_001_001234.pt        # Mid-epoch snapshots
├── mid_epoch_001_001567.pt        # (rotating, keeps 5)
├── timeout_exit.pt                # Saved by timeout guard ~10 min before Modal hard limit
└── checkpoint_epoch_005.pt        # Periodic saves
```

## Critical Settings Summary

### Local (configs/local/train_bimamba.yaml or train_fla.yaml)
```yaml
training:
  checkpoint_interval: 1  # Save every epoch
  mid_checkpoint_interval_s: 1800  # Save every 30 minutes
  mid_epoch_keep: 3  # Keep last 3 snapshots
```

### Modal (configs/modal/train_bimamba.yaml)
```yaml
training:
  checkpoint_interval: 1  # Already set correctly
```

## Why This Matters

1. **Long epochs**: Each epoch takes 2-3 hours locally
2. **Failure recovery**: Never lose more than 30 minutes
3. **Experimentation**: Easy to try different hyperparameters
4. **Standard practice**: All professional ML projects do this

## Current Training Status

If training is running without mid-epoch saves:
1. **Stop it**: `tmux kill-session -t v3_train`
2. **Set environment variables** (see above)
3. **Restart with resume**: Will continue from any existing checkpoints

---

**Storage Cost**: < 2GB total (0.1% of typical 2TB NVMe)
**Time Saved**: Potentially days of retraining
**Verdict**: 1000% DESIRED AND PROFESSIONAL
