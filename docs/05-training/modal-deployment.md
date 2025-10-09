# Modal Cloud Deployment Guide

**Last Updated**: October 8, 2025
**Architecture**: V3 dual-stream (A100-80GB optimized)
**Status**: Production-ready

## Quick Start

### Prerequisites
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Set W&B credentials (optional)
modal secret create wandb WANDB_API_KEY=<your-key>
```

### Deployment Steps

1. **Populate Cache** (one-time only — use --detach!)
   ```bash
   modal run --detach deploy/modal/app.py --action populate-cache
   # Expected: 4667 train + 1832 dev files
   # Time: ~1-2 hours for 450GB
   # NOTE: Intentionally clears existing cache before copying from S3
   ```

2. **Test Mamba CUDA**
   ```bash
   modal run deploy/modal/app.py --action test-mamba
   ```

3. **Verify Cache Health**
   ```bash
   modal run deploy/modal/app.py --action check-cache
   # Confirms train/dev counts, warns about stale manifests or stray NPZ files
   ```

4. **Run Smoke Test**
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
   # Time: ~5 minutes, AUROC: ~0.6-0.7
   ```

5. **Launch Full Training**
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml
   # FLA stack: use configs/modal/train_fla.yaml
   # Exits ~23 h with timeout_exit.pt (timeout guard); expect 4-5 resumes for 100 epochs
   ```

6. **Monitor & Resume**
   ```bash
   # List running apps
   modal app list

   # Stream logs
   modal app logs <app-id>

   # Stop if needed
   modal app stop <app-id>

   # Resume after timeout
   modal run --detach deploy/modal/app.py --action train \
      --config configs/modal/train_bimamba.yaml --resume true
   ```

## Modal Architecture

### Resource Allocation
- **GPU**: A100-80GB (required for Mamba CUDA kernels)
- **CPU**: 24 cores (avoid default 0.125!)
- **RAM**: 96GB (avoid bottlenecks)
- **Storage**: 500GB SSD volume

### Volume Structure
```
/results/                        # Persistent SSD volume
├── cache/
│   └── tusz_mmap/              # Memory-mapped NPY cache (train/dev pairs)
│       ├── train/              # 4,667 *_data.npy + *_labels.npy files + manifest.json
│       └── dev/                # 1,832 *_data.npy + *_labels.npy files + manifest.json
├── checkpoints/                # Training checkpoints
├── tensorboard/                # TensorBoard logs
└── wandb/                      # Weights & Biases artifacts
```

### Cache Strategy
- **Location**: `/results/cache/tusz_mmap` (memory-mapped uncompressed NPY cache)
- **Health checks**: `modal run deploy/modal/app.py --action check-cache` validates train/dev counts, manifests, and flags stray NPZ files.
- **Cleanup**: If `check-cache` reports NPZ contamination, run `modal run deploy/modal/clean_stray_npz.py --confirm` (script verifies matching `_data.npy/_labels.npy` before deletion).
- **NOT**: Direct S3 streaming — always populate the SSD volume first.
- **Reuse**: Populate once, then reuse across runs (only rerun `populate-cache` when you want a fresh copy).
- **Performance**: <1 ms window access, <2 GB RSS per worker (81× improvement over NPZ).

## Configuration (A100-optimized)

```yaml
# configs/modal/train_bimamba.yaml (actual configuration)
data:
  dataset: tuh_eeg
  data_dir: /data/edf                      # TUSZ EDF mount (train/dev/eval)
  cache_dir: /results/cache/tusz_mmap      # Memory-mapped NPY cache on SSD
  use_balanced_sampling: true              # Required for seizure-heavy batches
  num_workers: 4                           # Stable worker count on Modal
  persistent_workers: true                 # Keep workers alive between epochs
  prefetch_factor: 2                       # Prevent loader memory spikes

training:
  batch_size: 48                          # Optimized for A100-80GB (~58GB peak)
  learning_rate: 8.0e-5                   # Batch-size scaled from 3e-5
  gradient_clip: 0.5                      # NaN protection
  mixed_precision: true                   # A100 tensor cores (3.8x faster)
  gradient_accumulation_steps: 1          # No accumulation with batch_size=48

  scheduler:
    type: cosine
    warmup_ratio: 0.03

  # Mid-epoch checkpointing (critical for 6-7h epochs)
  mid_checkpoint_interval_s: 1800         # Every 30 minutes
  mid_epoch_keep: 3                       # Keep last 3 snapshots

  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    focal_gamma_enabled: true

experiment:
  output_dir: /results/v3_full_training
  cache_dir: /results/cache/tusz_mmap

  wandb:
    enabled: true
    project: seizure-detection-a100
    entity: your-team                     # Replace with your W&B entity
```

## Expected Performance

### Smoke Test (1 epoch, 50 files)
- Time: ~5 minutes
- AUROC: 0.6-0.7
- Loss: Should decrease
- Memory: ~20GB

### Full Training (100 epochs, 6499 files)
- Time: ~100 hours total (~1 hour/epoch)
- AUROC: >0.95
- Sensitivity@10FA: >90%
- Memory: 40-60GB
- Cost: ~$319
- Expect the timeout guard to exit every ~23 h; relaunch with `--resume true` to continue.

## Monitoring & Debugging

### Training Metrics
```python
# Key metrics to watch
- train/loss < 0.01
- val/auroc > 0.95
- val/sensitivity_at_10fa > 0.90
- learning_rate schedule
- gradient norms
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Cache not found | Run `populate-cache`, then `check-cache` to verify counts |
| Dev manifest warnings / 0 validation windows | Delete dev `manifest.json` and relaunch; loader rebuilds automatically |
| NaN losses | Confirm gradient clipping (0.5); optionally enable `BGB_SANITIZE_GRADS=1` while debugging |
| Modal job killed at 24 h | Use v3.9.0+ (timeout guard writes `timeout_exit.pt`); resume with `--resume true` |
| W&B shows new run after resume | Ensure checkpoint directory is writable so `.wandb_run_id` can be updated |
| OOM errors | Keep `batch_size: 48`, `gradient_accumulation_steps: 1`; reverting to 32/2 is the fallback |
| Slow training | Confirm using SSD cache (`/results/cache/tusz_mmap`) instead of S3 |
| Connection lost | Always run with `--detach` or inside `tmux` |

### Debug Environment Variables
```bash
# Enable for debugging
export BGB_NAN_DEBUG=1           # Verbose NaN reporting
# export BGB_SANITIZE_GRADS=1    # Optional: zero/log non-finite gradients
export BGB_DEBUG_FINITE=1        # Check tensor finiteness
export BGB_WALL_CLOCK_LIMIT_S=82800  # (Modal sets automatically) timeout guard writes timeout_exit.pt

# Limit data for testing
export BGB_LIMIT_FILES=50        # Use only 50 files
export BGB_SMOKE_TEST=1          # Skip balanced sampling
```

## Cost Optimization

### Tips to Reduce Costs
1. **Use smoke tests** to validate changes (~$2)
2. **Enable checkpointing** to resume from failures
3. **Use `--detach`** to avoid connection issues
4. **Monitor early** - stop if metrics look wrong
5. **Use spot instances** if available (not yet on Modal)

### Cost Breakdown (Full Training)
- A100-80GB: ~$3.19/hour
- 100 hours: ~$319
- Storage: ~$10/month for 500GB
- Total: ~$330 per full run

## Resume from Checkpoint

```bash
# Resume training from latest checkpoint
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume true
```

## Troubleshooting Checklist

Before training:
- [ ] Modal authenticated (`modal setup`)
- [ ] Cache populated (4667 + 1832 files)
- [ ] Mamba CUDA test passed
- [ ] Smoke test completed successfully
- [ ] W&B credentials set (if using)

During training:
- [ ] Monitor for NaN losses
- [ ] Check AUROC > 0.5 (not collapsed)
- [ ] Verify memory usage < 70GB
- [ ] Watch for gradient explosions
- [ ] Confirm using SSD cache path

## Commands Reference

```bash
# Setup
modal setup
modal secret create wandb WANDB_API_KEY=<key>

# Cache Management
modal run --detach deploy/modal/app.py --action populate-cache
modal run deploy/modal/app.py --action check-cache
modal run deploy/modal/clean_stray_npz.py --confirm

# Testing
modal run deploy/modal/app.py --action test-mamba

# Training
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# Resume
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml --resume true

# Monitoring
modal app list
modal app logs <app-id>
modal app stop <app-id>
```

## Links
- [Modal Volume Architecture](../08-operations/modal-volume-architecture.md)
- [Configuration Guide](../03-configuration/modal-configs.md)
- [NaN Prevention](../08-operations/nan-prevention-complete.md)
- [Local Training](local-training.md)
