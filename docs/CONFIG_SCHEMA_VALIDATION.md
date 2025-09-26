# CONFIG SCHEMA VALIDATION - COMPLETE SPECIFICATION

**CRITICAL**: This document defines EVERY valid configuration field for Brain-Go-Brr V3.
Any field NOT listed here will cause validation errors with `extra="forbid"`.

Generated: 2025-09-25
Schema Source: `/src/brain_brr/config/schemas.py`

---

## 🔴 VALIDATION ERRORS FOUND IN CONFIGS

### local/train.yaml VIOLATIONS:
```yaml
logging.wandb  # ❌ INVALID - wandb is under experiment, not logging!
resources.compile_model  # ❌ INVALID - not in ResourcesConfig
resources.tf32_mode  # ❌ INVALID - not in ResourcesConfig
resources.cudnn_benchmark  # ❌ INVALID - not in ResourcesConfig
```

### MUST FIX:
1. Move `wandb` from `logging` to `experiment.wandb`
2. Remove `compile_model`, `tf32_mode`, `cudnn_benchmark` from resources

---

## 📋 COMPLETE VALID SCHEMA

### Root Level (`Config`)
```yaml
data:          # DataConfig
preprocessing: # PreprocessingConfig
model:         # ModelConfig
postprocessing: # PostprocessingConfig
training:      # TrainingConfig
evaluation:    # EvaluationConfig
experiment:    # ExperimentConfig
logging:       # LoggingConfig
resources:     # ResourcesConfig (optional)
```

### `data:` (DataConfig)
```yaml
data:
  dataset: "tuh_eeg"  # or "chb_mit"
  data_dir: path/to/data  # Path
  cache_dir: path/to/cache  # Path
  use_balanced_sampling: true  # bool
  sampling_rate: 256  # MUST be 256
  n_channels: 19  # MUST be 19
  window_size: 60  # MUST be 60
  stride: 10  # MUST be 10
  num_workers: 0  # int, 0-32
  pin_memory: false  # bool
  persistent_workers: false  # bool
  prefetch_factor: 2  # int, >=2
  split_policy: "official_tusz"  # string
  validation_split: 0.2  # float, 0-0.5 (DEPRECATED)
  split_seed: 42  # int
  max_samples: null  # int or null
  max_hours: null  # float or null
```

### `preprocessing:` (PreprocessingConfig)
```yaml
preprocessing:
  montage: "10-20"  # or "standard_1020"
  bandpass: [0.5, 120.0]  # tuple[float, float]
  notch_freq: 60  # 50 or 60
  normalize: true  # bool
  use_mne: true  # bool
```

### `model:` (ModelConfig)
```yaml
model:
  name: "seizure_detector"  # MUST be this
  architecture: "v3"  # MUST be "v3"

  tcn:  # TCNConfig
    num_layers: 8  # int, 4-12
    kernel_size: 7  # int, 3-11
    dropout: 0.15  # float, 0-0.5
    causal: false  # bool
    stride_down: 16  # int
    use_cuda_optimizations: true  # bool

  mamba:  # MambaConfig
    n_layers: 6  # int, 1-12
    d_model: 512  # MUST be 512
    d_state: 16  # MUST be 16
    conv_kernel: 4  # int, 2-4 (CUDA constraint)
    dropout: 0.1  # float, 0-0.5

  graph:  # GraphConfig (optional)
    enabled: true  # bool

    # Edge stream (V3)
    edge_features: "cosine"  # or "correlation"
    edge_top_k: 3  # int, 1-18
    edge_threshold: 0.0001  # float, >=0
    edge_mamba_layers: 2  # int, 1-6
    edge_mamba_d_state: 8  # int, 4-64
    edge_mamba_d_model: 16  # int, 8-64 (multiple of 8)

    # GNN architecture
    n_layers: 2  # int, 1-4
    dropout: 0.1  # float, 0-0.5
    use_residual: true  # bool
    alpha: 0.05  # float, 0-1

    # Laplacian PE
    k_eigenvectors: 16  # int, 1-18

    # Dynamic PE (V3)
    use_dynamic_pe: true  # bool
    semi_dynamic_interval: 1  # int, 1-960
    pe_sign_consistency: true  # bool
```

### `postprocessing:` (PostprocessingConfig)
```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86  # float, 0.5-1.0
    tau_off: 0.78  # float, 0.5-1.0 (< tau_on)
    min_onset_samples: 128  # int, >=1
    min_offset_samples: 256  # int, >=1

  morphology:
    opening_kernel: 11  # int, odd number
    closing_kernel: 31  # int, odd number
    use_gpu: false  # bool

  duration:
    min_duration_s: 3.0  # float, >=0
    max_duration_s: 600.0  # float, >0

  events:
    tau_merge: 2.0  # float, >=0
    confidence_method: "mean"  # "mean", "peak", or "percentile"
    confidence_percentile: 0.75  # float, 0-1

  stitching:
    method: "overlap_add"  # or "overlap_add_weighted", "max"
    window_size: 15360  # int, >=1
    stride: 2560  # int, >=1

  min_duration: 3.0  # DEPRECATED, use duration.min_duration_s
```

### `training:` (TrainingConfig)
```yaml
training:
  epochs: 100  # int, 1-200
  batch_size: 16  # int, 1-256

  # Loss configuration
  loss: "focal"  # or "bce"
  focal_alpha: 0.5  # float, 0-1
  focal_gamma: 2.0  # float, >=0

  # Optimizer
  learning_rate: 0.0003  # float, 1e-6 to 1e-2
  weight_decay: 0.05  # float, 0-0.2
  optimizer: "adamw"  # or "adam", "sgd"

  # Training control
  resume: false  # bool
  gradient_clip: 1.0  # float, >=0 (0=disabled)
  mixed_precision: true  # bool
  checkpoint_interval: 1  # int, 0-100 (0=disabled)
  gradient_accumulation_steps: 1  # int, 1-100

  scheduler:
    type: "cosine"  # or "linear", "constant"
    warmup_ratio: 0.1  # float, 0-0.5

  early_stopping:
    patience: 5  # int, 1-50
    metric: "sensitivity_at_10fa"  # string
    mode: "max"  # or "min"
```

### `evaluation:` (EvaluationConfig)
```yaml
evaluation:
  metrics: ["taes", "sensitivity", "specificity", "auroc"]  # list[str]
  fa_rates: [10, 5, 2.5, 1]  # list[float]
  save_predictions: false  # bool
  save_plots: true  # bool
```

### `experiment:` (ExperimentConfig)
```yaml
experiment:
  name: "my_experiment"  # string
  description: "description"  # string
  seed: 42  # int
  device: "auto"  # or "cuda", "cpu", "mps"
  output_dir: results/  # Path
  cache_dir: cache/  # Path
  log_level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
  save_model: false  # bool
  save_best_only: true  # bool

  wandb:  # WandbConfig (NOT under logging!)
    enabled: false  # bool
    project: "seizure-detection"  # string
    entity: null  # string or null
    tags: []  # list[str]
```

### `logging:` (LoggingConfig)
```yaml
logging:
  log_every_n_steps: 10  # int, >=1
  log_gradients: false  # bool
  log_weights: false  # bool
  # NO wandb here! It's under experiment!
```

### `resources:` (ResourcesConfig) - OPTIONAL
```yaml
resources:
  max_memory_gb: null  # float or null, >0
  distributed: false  # bool
  mixed_precision: true  # bool
  # NO compile_model, tf32_mode, or cudnn_benchmark!
```

---

## 🚨 CRITICAL RULES

1. **STRICT MODE**: All configs use `extra="forbid"` - ANY field not listed above will FAIL
2. **Fixed Values**: Some fields MUST have specific values:
   - `sampling_rate: 256`
   - `n_channels: 19`
   - `window_size: 60`
   - `stride: 10`
   - `mamba.d_model: 512`
   - `mamba.d_state: 16`
   - `model.architecture: "v3"`

3. **Validation Rules**:
   - `tau_on > tau_off` (hysteresis)
   - `max_duration_s >= min_duration_s`
   - Kernel sizes must be odd
   - `bandpass[0] < bandpass[1]`
   - `conv_kernel` must be 2-4 (CUDA constraint)

4. **Deprecated Fields**:
   - `postprocessing.min_duration` → use `duration.min_duration_s`
   - `validation_split` → use `split_policy: "official_tusz"`

---

## ✅ FIXES NEEDED IMMEDIATELY

### 1. Fix local/train.yaml:
```yaml
# REMOVE these from resources:
resources:
  compile_model: false  # ❌ DELETE THIS LINE
  tf32_mode: true  # ❌ DELETE THIS LINE
  cudnn_benchmark: true  # ❌ DELETE THIS LINE

# MOVE wandb from logging to experiment:
logging:
  wandb:  # ❌ DELETE THIS SECTION
    enabled: false
    project: seizure-v3
    entity: null

# ADD to experiment instead:
experiment:
  wandb:  # ✅ PUT IT HERE
    enabled: false
    project: seizure-v3
    entity: null
```

### 2. Check ALL configs for similar issues:
- modal/train.yaml
- modal/smoke.yaml
- local/smoke.yaml

---

## 🔍 VALIDATION COMMAND

```bash
# Test config validation:
python -c "from src.brain_brr.config.schemas import Config; Config.from_yaml('configs/local/train.yaml')"
```

If this passes, the config is VALID. If it fails, check the error against this document.

---

**STATUS**: CRITICAL - Full training BLOCKED until configs are fixed!