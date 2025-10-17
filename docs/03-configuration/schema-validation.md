# Configuration Schema Validation Guide

**Last Updated**: October 3, 2025
**Schema Source**: `src/brain_brr/config/schemas.py`
**Validation Mode**: STRICT (`extra="forbid"`)

## Critical Rules

1. **Strict Validation**: Any field not listed below will cause validation errors
2. **Fixed Values**: Some fields MUST have specific values (e.g., `sampling_rate: 256`)
3. **Type Safety**: All fields are strongly typed with Pydantic
4. **Deprecation Warnings**: Legacy fields will warn before removal

## Complete Valid Schema

### Root Level
```yaml
data:          # DataConfig
preprocessing: # PreprocessingConfig
model:         # ModelConfig
postprocessing: # PostprocessingConfig
training:      # TrainingConfig
evaluation:    # EvaluationConfig
experiment:    # ExperimentConfig
logging:       # LoggingConfig
# Note: resources field removed in v3.3.1 - mixed_precision moved to training
```

### Data Configuration
```yaml
data:
  dataset: "tuh_eeg"              # ONLY tuh_eeg supported (TUSZ corpus)
  data_dir: path/to/data          # Path to raw EDF files (train/, dev/, eval/)
  cache_dir: path/to/cache        # Path to memory-mapped NPY cache (train/, dev/ subdirs with *_data.npy/*_labels.npy pairs)
  use_balanced_sampling: true     # CRITICAL for seizure detection (oversample seizures 8% → 30%)
  sampling_rate: 256              # MUST be 256 Hz
  n_channels: 19                  # MUST be 19 (10-20 montage)
  window_size: 60                 # MUST be 60 seconds
  stride: 10                      # MUST be 10 seconds
  num_workers: 0                  # 0 for WSL2, 4 for native Linux
  pin_memory: false               # Enable when num_workers > 0 on Linux
  persistent_workers: false       # Requires num_workers > 0 to take effect
  prefetch_factor: 2              # Default per-worker prefetch depth
```

**Official splits are enforced automatically.** The loader scans `<data_dir>/{train,dev,eval}`
and aborts if patient IDs overlap; legacy `split_policy`, `validation_split`, and
`split_seed` fields were removed in V4.

### Model Configuration (V3 Architecture)
```yaml
model:
  name: "seizure_detector"        # MUST be this
  architecture: "v3"              # MUST be "v3" (only option)

  tcn:                            # Temporal Convolutional Network
    num_layers: 8
    kernel_size: 7
    dropout: 0.15
    stride_down: 16               # Downsampling factor

  mamba:                          # Bidirectional Mamba SSM
    n_layers: 6
    d_model: 512                  # MUST be 512
    d_state: 16                   # MUST be 16
    conv_kernel: 4                # CUDA constraint: 2-4
    dropout: 0.1

  graph:                          # Graph Neural Network
    enabled: true

    # Edge stream parameters
    edge_features: "cosine"       # or "correlation"
    edge_similarity_margin: 0.01   # Clamp similarity to [-1+margin, 1-margin]
    edge_top_k: 3                 # Top-k edges per node
    edge_threshold: 0.0001
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16        # Must be multiple of 8

    # GNN architecture
    n_layers: 2
    dropout: 0.1
    alpha: 0.05                   # SSGConv alpha for EEG

    # Laplacian PE
    k_eigenvectors: 16
    use_dynamic_pe: true          # Dynamic positional encoding
    semi_dynamic_interval: 1
    pe_sign_consistency: true
```

### Training Configuration
```yaml
training:
  epochs: 100
  batch_size: 8                   # 8 for RTX 4090, 48 for A100

  # Loss configuration
  loss: "focal"                   # Required for class imbalance (ONLY option as of v3.7.0)
  focal_alpha: 0.5
  focal_gamma: 2.0

  # Optimizer
  learning_rate: 1.0e-4           # 1.0e-4 local, 8.0e-5 modal (batch-size scaled)
  weight_decay: 0.01              # 0.01 local, 0.05 modal
  optimizer: "adamw"              # ONLY option

  # Training control
  gradient_clip: 0.5              # Increased from 0.1 (post eigendecomposition fix)
  mixed_precision: false          # false for RTX 4090, true for A100
  checkpoint_interval: 1
  mid_checkpoint_interval_s: 1800 # Save every 30 min for long epochs
  mid_epoch_keep: 3               # Keep last 3 mid-epoch checkpoints

  scheduler:
    type: "cosine"                # ONLY option
    warmup_ratio: 0.03            # 3% warmup (increased from 0.01)

  # Optional: Warmup schedules for gradient stabilization
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    focal_gamma_enabled: true
```

### Post-processing Configuration
```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86                  # Onset threshold
    tau_off: 0.78                 # Offset threshold (< tau_on)

  morphology:
    opening_kernel: 11            # Must be odd
    closing_kernel: 31            # Must be odd

  duration:
    min_duration_s: 3.0           # Minimum seizure duration
    max_duration_s: 600.0         # Maximum seizure duration

  events:
    tau_merge: 2.0                # Merge events within 2s
```

### Experiment Configuration
```yaml
experiment:
  name: "v3_training"
  seed: 42
  device: "auto"                  # Auto-detect GPU
  output_dir: results/

  wandb:                          # W&B config (NOT under logging!)
    enabled: false
    project: "seizure-v3"
    entity: null
```

### Logging Configuration
```yaml
logging:
  log_every_n_steps: 10
  log_gradients: false
  log_weights: false
  # Note: wandb goes under experiment, not here!
```

## Common Validation Errors

### Error: wandb under wrong section
```yaml
logging:
  wandb: {...}  # ❌ INVALID - should be under experiment
```
**Fix**: Move to `experiment.wandb`

### Error: Invalid architecture
```yaml
model:
  architecture: "tcn"  # ❌ INVALID - must be "v3"
```
**Fix**: Use `architecture: "v3"` (only supported architecture)

### Error: use_balanced_sampling with smoke tests
```yaml
data:
  use_balanced_sampling: true  # ❌ INVALID for smoke tests with BGB_SMOKE_TEST=1
```
**Fix**: Set `use_balanced_sampling: false` for smoke tests (3-50 files)

## Validation Command

Test your configuration:
```bash
python -c "from src.brain_brr.config.schemas import Config; Config.from_yaml('configs/local/train_bimamba.yaml')"
```

If this passes without errors, your config is valid!

## Fixed Values That Cannot Change

These values are hardcoded in the model architecture:
- `data.sampling_rate: 256` - Model expects 256 Hz
- `data.n_channels: 19` - 10-20 montage standard
- `data.window_size: 60` - 60-second windows
- `data.stride: 10` - 10-second stride
- `mamba.d_model: 512` - Mamba dimension
- `mamba.d_state: 16` - Mamba state size
- `model.architecture: "v3"` - Only V3 supported

## Validation Rules

1. **Hysteresis**: `tau_on > tau_off`
2. **Duration**: `max_duration_s >= min_duration_s`
3. **Kernels**: Must be odd numbers
4. **Bandpass**: `bandpass[0] < bandpass[1]`
5. **Conv kernel**: Must be 2-4 (CUDA constraint)
6. **Edge model dimension**: Multiple of 8

## Related Documentation
- [Local Configuration](local-configs.md)
- [Modal Configuration](modal-configs.md)
- [Environment Variables](env-vars.md)
