# LOCAL CONFIG CONSISTENCY VERIFICATION ✅

## FIXES APPLIED

### 1. smoke.yaml
- ✅ Added `use_mne: true` (was missing, inconsistent with others)

### 2. dev.yaml
- ✅ Changed `num_workers: 4` → `0` (WSL2-safe)
- ✅ Added `pin_memory: false` (WSL2-safe)
- ✅ Added `persistent_workers: false` (WSL2-safe)
- ✅ Changed `device: auto` → `cuda` (explicit GPU)
- ✅ Updated checkpoint path to match train.yaml output: `results/tcn_full_100ep/checkpoints/best.pt`

### 3. eval.yaml
- ✅ Changed `num_workers: 4` → `0` (WSL2-safe)
- ✅ Added `pin_memory: false` (WSL2-safe)
- ✅ Added `persistent_workers: false` (WSL2-safe)
- ✅ Changed `device: auto` → `cuda` (explicit GPU)
- ✅ Updated checkpoint path to match train.yaml output: `results/tcn_full_100ep/checkpoints/best.pt`

## INTERNALLY CONSISTENT PARAMETERS

### WSL2-Safe Settings (ALL configs now have):
```yaml
num_workers: 0                # Avoid multiprocessing hangs
pin_memory: false             # Prevent /dev/shm issues
persistent_workers: false     # Critical for stability
device: cuda                  # Explicit GPU (auto defaults wrong)
```

### Model Architecture (IDENTICAL across all):
```yaml
model:
  architecture: tcn
  tcn:
    num_layers: 8
    channels: [64, 128, 256, 512]
    kernel_size: 7
    dropout: 0.15
    stride_down: 16
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
    dropout: 0.1
```

### Preprocessing (IDENTICAL across all):
```yaml
preprocessing:
  montage: "10-20"
  bandpass: [0.5, 120]
  notch_freq: 60
  normalize: true
  use_mne: true              # Now consistent!
```

### Post-processing (IDENTICAL across all):
```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
  morphology:
    opening_kernel: 11
    closing_kernel: 31
  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0
  events:
    tau_merge: 2.0
    confidence_method: mean
```

### Data Pipeline (CONSISTENT):
```yaml
data:
  dataset: tuh_eeg
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  use_balanced_sampling: true    # Critical for all!
```

## CONFIG-SPECIFIC SETTINGS (CORRECT)

| Config | Purpose | Data Dir | Cache Dir | Epochs | Batch Size |
|--------|---------|----------|-----------|---------|------------|
| smoke.yaml | Quick test | .../train | cache/smoke | 1 | 8 |
| train.yaml | Full training | .../train | cache/tusz | 100 | 8 |
| dev.yaml | Tuning | .../dev | cache/dev_tuning | 0 | 32 |
| eval.yaml | Final test | .../eval | cache/eval_final | 0 | 32 |

## CHECKPOINT FLOW

1. **train.yaml** → Creates `results/tcn_full_100ep/checkpoints/best.pt`
2. **dev.yaml** → Loads from `results/tcn_full_100ep/checkpoints/best.pt`
3. **eval.yaml** → Loads from `results/tcn_full_100ep/checkpoints/best.pt`

## VALIDATION

All local configs are now:
- ✅ WSL2-safe (num_workers=0, pin_memory=false, etc.)
- ✅ Internally consistent (same model, preprocessing, postprocessing)
- ✅ Properly linked (checkpoint paths match)
- ✅ Cache-separated (unique cache directories)
- ✅ GPU-explicit (device: cuda, not auto)

Ready for future training runs!
