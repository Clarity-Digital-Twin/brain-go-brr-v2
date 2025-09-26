# Modal Cloud Deployment Guide

**Last Updated**: September 26, 2025
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

1. **Populate Cache** (one-time only)
   ```bash
   modal run deploy/modal/app.py --action populate-cache
   # Expected: 4667 train + 1832 dev files
   # Time: ~1-2 hours for 450GB
   ```

2. **Test Mamba CUDA**
   ```bash
   modal run deploy/modal/app.py --action test-mamba
   ```

3. **Run Smoke Test**
   ```bash
   modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
   # Time: ~5 minutes, AUROC: ~0.6-0.7
   ```

4. **Launch Full Training**
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
   # Time: ~100 hours, Cost: ~$319
   ```

5. **Monitor Training**
   ```bash
   # List running apps
   modal app list

   # Stream logs
   modal app logs <app-id>

   # Stop if needed
   modal app stop <app-id>
   ```

## Modal Architecture

### Resource Allocation
- **GPU**: A100-80GB (required for Mamba CUDA kernels)
- **CPU**: 24 cores (avoid default 0.125!)
- **RAM**: 96GB (avoid bottlenecks)
- **Storage**: 500GB SSD volume

### Volume Structure
```
/results/                    # Persistent SSD volume
├── cache/                   # Pre-populated cache (450GB)
│   └── tusz/
│       ├── train/          # 4667 NPZ files (~306GB)
│       └── dev/            # 1832 NPZ files (~143GB)
├── checkpoints/            # Training checkpoints
├── tensorboard/            # Training logs
└── wandb/                  # W&B artifacts
```

### Cache Strategy
- **Location**: `/results/cache/tusz` on SSD volume
- **NOT**: `/cache` from S3 mount (too slow!)
- **Method**: One-time population, then reuse across runs
- **Performance**: 10x faster than S3 mount

## Configuration (A100-optimized)

```yaml
# configs/modal/train.yaml
data:
  cache_dir: /results/cache/tusz  # SSD volume, not S3!
  num_workers: 8                   # A100 handles parallel IO

training:
  batch_size: 64                   # Larger batch for 80GB
  learning_rate: 3e-5              # Conservative for stability
  gradient_clip: 0.5               # Strong clipping
  mixed_precision: true            # A100 tensor cores

  scheduler:
    warmup_ratio: 0.03              # 3% warmup

resources:
  distributed: false                # Single GPU training
  mixed_precision: true             # FP16 on A100

experiment:
  output_dir: /results/train/

  wandb:
    enabled: true
    project: "seizure-v3"
    entity: "your-team"             # Set your team name
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
| Cache not found | Run `populate-cache` action first |
| NaN losses | Enable `BGB_SANITIZE_GRADS=1` |
| OOM errors | Reduce batch size to 32 |
| Slow training | Verify using SSD cache, not S3 |
| Connection lost | Use `--detach` for long runs |

### Debug Environment Variables
```bash
# Enable for debugging
export BGB_NAN_DEBUG=1           # Verbose NaN reporting
export BGB_SANITIZE_GRADS=1      # Clean gradients (recommended)
export BGB_DEBUG_FINITE=1        # Check tensor finiteness

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
  --config configs/modal/train.yaml \
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
modal run deploy/modal/app.py --action populate-cache
modal run deploy/modal/app.py --action verify-cache

# Testing
modal run deploy/modal/app.py --action test-mamba

# Training
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Resume
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true

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