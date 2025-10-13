# Modal BiMamba2 Training Backup - Epoch 6

**Status**: Paused to control costs
**Date Stopped**: October 13, 2025
**Training Progress**: 5 complete epochs + 50% of epoch 6 (batch 647/1284)
**Modal Spend**: $1,118 (includes debugging/smoke tests, ~$600 actual training)

## Why Paused?

Budget-conscious decision as independent researcher:
- BiMamba2 is baseline comparison
- FLA (Gated DeltaNet) is primary research hypothesis
- Remaining cost: ~$900 to complete epochs 6-100
- Strategy: Finish FLA first (free local training), reassess if BiMamba2 comparison needed

## What's Saved?

### On Modal SSD (Primary, Free Storage)
Modal volume: `brain-go-brr-results`
Path: `v3_full_training/checkpoints/`

Files:
- `epoch_001.pt` through `epoch_005.pt` (5 complete epochs)
- `best.pt` (best validation metric)
- `last.pt` (most recent complete epoch = epoch 5)
- `mid_epoch_006_000647.pt` (stopped at 50% through epoch 6)

### Local Backup (This Directory, Insurance Policy)
Downloaded checkpoints:
- `best.pt` (~125MB) - Best validation performance
- `last.pt` (~125MB) - Most recent complete epoch
- `epoch_005.pt` (~125MB) - Last fully trained epoch

**Purpose**: Insurance against Modal account issues. Modal storage is free and persistent, but local backup provides safety net.

## How to Resume Training

### Option 1: Resume on Modal (Same Account)
```bash
# Checkpoints already on Modal SSD, just resume
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume true

# Training will auto-load last.pt and continue from epoch 6
```

### Option 2: Resume on Modal (New Account / Restore from Local)
```bash
# 1. Upload local backup to new Modal volume
modal volume create brain-go-brr-results-restored
modal volume put brain-go-brr-results-restored \
  backups/modal_bimamba2_epoch6/last.pt \
  v3_full_training/checkpoints/last.pt

# 2. Resume training (will load uploaded checkpoint)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume true
```

### Option 3: Resume on Local RTX 4090
```bash
# Copy checkpoint to local results directory
cp backups/modal_bimamba2_epoch6/last.pt results/local_bimamba_training/checkpoints/

# Resume training (will auto-detect checkpoint)
export BGB_NAN_DEBUG=1
tmux new -s train-bimamba
.venv/bin/python -m src train configs/local/train_bimamba.yaml --resume
```

**Note**: Local training is ~2-3x slower than Modal A100 but free. Budget ~200-300 hours total.

## Future Training Strategy

**Incremental approach**: $500-1000/month when budget allows
- Complete FLA training first (local, free)
- Assess if BiMamba2 comparison needed based on FLA results
- Resume BiMamba2 in 10-20 epoch chunks if valuable
- Each chunk: ~$90-180 for 10 epochs (~10 hours A100 time)

## Training Configuration

**Modal A100-80GB**:
- Batch size: 48 (58GB peak memory)
- Mixed precision: Enabled (3.8x speedup)
- ~1 hour/epoch average
- ~$9.50/hour A100 cost

**What We Learned**:
- First 5 epochs stable, no NaN issues
- Dynamic LPE working (eigendecomposition detached)
- Edge similarity margin (0.01) prevents boundary explosions
- Mid-epoch checkpoints every 30 min = $150+ savings on resume

## Current Focus: FLA Training

**Local RTX 4090**: Currently on epoch 2, progressing well
- BiGatedDeltaNet architecture (same parameter count as BiMamba2)
- WSL2 SIGBUS issue resolved (cache on ext4 filesystem)
- Zero cost, can run continuously
- Expected completion: ~200-300 hours

**Research Question**: Does delta rule + gating improve over pure gating (BiMamba2) for EEG seizure detection?

---

**Last Updated**: 2025-10-13
**Contact**: Independent research project, Brain-Go-Brr V4.0.0
