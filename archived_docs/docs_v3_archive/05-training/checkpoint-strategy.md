# Checkpoint Strategy Guide

## Overview
Professional checkpoint strategy for long-running EEG training on both local (RTX 4090) and Modal (A100).

## Checkpoint Sizes
- **V3 architecture**: ~195MB per checkpoint
- **Storage overhead**: Minimal (< 2GB total with rotation)
- **Professional practice**: YES, this is standard for deep learning

## Local Training (RTX 4090)

### Environment Variables (REQUIRED)
```bash
# Set BEFORE starting training
export BGB_MID_EPOCH_MINUTES=30  # Save every 30 minutes
export BGB_MID_EPOCH_KEEP=5      # Keep last 5 snapshots (rotating)
```

### What Gets Saved
1. **Every epoch**: `last.pt` (for resume)
2. **Best model**: `best.pt` (when validation improves)
3. **Mid-epoch**: `mid_epoch_XXX_YYYYYY.pt` (every 30 min, keeps 5)
4. **Periodic**: `checkpoint_epoch_XXX.pt` (every epoch now)

### Storage Requirements
- **Per epoch**: ~400MB (last + best)
- **Mid-epoch**: ~1GB rolling (5 × 195MB)
- **Total**: < 2GB active storage

### Resume Command
```bash
# Resume from latest checkpoint
.venv/bin/python -m src train configs/local/train.yaml --resume

# Automatically picks up from:
# 1. Latest mid-epoch checkpoint (if exists)
# 2. last.pt (if no mid-epoch)
# 3. Fresh start (if no checkpoints)
```

## Modal Training (A100)

### Configuration (Already Optimal)
- Saves every epoch (`checkpoint_interval: 1`)
- Persistence volume handles all checkpoints
- Automatic resume with `--resume true`

### Modal Commands
```bash
# Start training (with automatic checkpointing)
modal run --detach deploy/modal/app.py \
    --action train \
    --config configs/modal/train.yaml

# Resume from checkpoint
modal run --detach deploy/modal/app.py \
    --action train \
    --config configs/modal/train.yaml \
    --resume true
```

## Best Practices

### DO's
✅ **Always set mid-epoch checkpointing** for local training
✅ **Keep 3-5 mid-epoch checkpoints** (rotating)
✅ **Save every epoch** (`checkpoint_interval: 1`)
✅ **Use --resume flag** when restarting

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
# - results/full_training/ (current)
# - results/{experiment_name}/ (new runs)
```

### Checkpoint Files
```
results/full_training/checkpoints/
├── best.pt                        # Best validation score
├── last.pt                        # Latest epoch (resume point)
├── mid_epoch_001_001234.pt        # Mid-epoch snapshots
├── mid_epoch_001_001567.pt        # (rotating, keeps 5)
└── checkpoint_epoch_005.pt        # Periodic saves
```

## Critical Settings Summary

### Local (configs/local/train.yaml)
```yaml
training:
  checkpoint_interval: 1  # Save every epoch
  # Set via environment:
  # BGB_MID_EPOCH_MINUTES=30
  # BGB_MID_EPOCH_KEEP=5
```

### Modal (configs/modal/train.yaml)
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