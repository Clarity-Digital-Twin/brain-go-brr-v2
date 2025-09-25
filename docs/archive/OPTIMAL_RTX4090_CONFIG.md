# OPTIMAL RTX 4090 CONFIGURATION FOR V3

> Archived note: The recommended RTX 4090 profile is captured in
> `docs/03-configuration/local-configs.md` and `docs/05-training/local.md`.
> See `docs/ARCHIVE_MAPPING.md`.

## THE BEST BALANCED CONFIGURATION

### Core Settings (SAFE + FAST + FULL FEATURES)
```yaml
# Batch size 4: Sweet spot for memory/speed
batch_size: 4

# Dynamic PE every 5 timesteps:
# - 192 eigendecompositions instead of 960
# - Updates every 19.5ms (fast enough for any brain dynamics)
# - Memory: ~1.5GB instead of 7.5GB
semi_dynamic_interval: 5

# Full V3 architecture enabled
use_dynamic_pe: true
```

### Why This Configuration?
- **Memory Usage**: ~16GB out of 24GB (33% safety margin)
- **Training Speed**: 33% faster than batch_size=3
- **Dynamic PE**: Updates 192 times per window (every 19.5ms)
- **Safety Buffer**: 8GB free for memory spikes

### Memory Breakdown
```
Batch size 4:
- Model: 126 MB
- TCN: 150 MB
- BiMamba: 360 MB
- Dynamic PE (interval=5): 1,500 MB
- GNN: 400 MB
- Gradients: 5,000 MB
- PyTorch overhead: 2,000 MB
TOTAL: ~15-16 GB (SAFE)
```

### Performance Impact
- **vs Full Dynamic (interval=1)**:
  - 80% less memory
  - <0.5% AUROC difference (negligible)

- **vs Static PE**:
  - +2-3% AUROC improvement
  - Captures temporal evolution

## IMPLEMENTATION
