# Configuration Verification Report

## All Configs Checked ✅

### Local Configs (WSL2-optimized)
All configs in `configs/local/` have:
- ✅ `num_workers: 0` (prevents WSL2 hangs)
- ✅ `pin_memory: false` (prevents /dev/shm issues)
- ✅ `use_balanced_sampling: true` (handles class imbalance)
- ✅ `batch_size: 8` (fits in RTX 4090 24GB)
- ✅ `cache_dir: cache/tusz` (local cache)

### Modal Configs (Cloud A100)
All configs in `configs/modal/` have:
- ✅ `num_workers: 4` (cloud can handle multiprocessing)
- ✅ `pin_memory: true` (better GPU transfer)
- ✅ `use_balanced_sampling: true` (same algorithm)
- ✅ `batch_size: 16` (A100 has 40GB VRAM)
- ✅ `cache_dir: /results/cache/tusz` (persistent volume)

## Critical Settings Verified

### Data Pipeline
```yaml
data:
  dataset: tuh_eeg
  sampling_rate: 256          # Standardized
  n_channels: 19              # 10-20 montage
  window_size: 60             # 60 second windows
  stride: 10                  # 10 second stride
  validation_split: 0.2       # 80/20 split
  use_balanced_sampling: true # SeizureTransformer approach
```

### Model Architecture
```yaml
model:
  name: seizure_detector
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 5  # Note: CUDA coerces to 4
```

### Post-processing
```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86   # Onset threshold
    tau_off: 0.78  # Offset threshold
```

## Config Hierarchy

1. **smoke.yaml**: Quick test (1 epoch, 5 min)
2. **dev.yaml**: Development (10 epochs, 1 hour)
3. **train.yaml**: Full training (100 epochs, 16-20 hours)
4. **eval.yaml**: Evaluation only (no training)

## Differences: Local vs Modal

| Setting | Local (WSL2) | Modal (Cloud) | Reason |
|---------|-------------|---------------|---------|
| num_workers | 0 | 4 | WSL2 multiprocessing issues |
| pin_memory | false | true | WSL2 /dev/shm limited |
| batch_size | 8 | 16 | GPU memory (24GB vs 40GB) |
| cache_dir | cache/tusz | /results/cache/tusz | Persistent volume |

## Validation Status

All configs:
- ✅ Use balanced sampling (critical for <0.1% seizure data)
- ✅ Have correct paths for their environment
- ✅ Use same model architecture
- ✅ Use same preprocessing pipeline
- ✅ Use same evaluation metrics

## No Issues Found

All configurations are properly set up for their respective environments!