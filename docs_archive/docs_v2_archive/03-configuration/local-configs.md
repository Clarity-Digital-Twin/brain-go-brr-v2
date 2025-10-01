# Local Configs (RTX 4090)

**Last Updated**: October 1, 2025 (v3.4.1)
**Target Hardware**: RTX 4090 24GB (WSL2 or native Linux)

## Overview

V3 dual-stream architecture is the **only supported** architecture. V2 heuristic path has been removed.

## Key Recommendations (v3.4.1)

### Data
- `data.cache_dir: cache/tusz` (local SSD)
- `data.num_workers: 0` (WSL2 stability) or `4` (native Linux with `pin_memory: true`)
- `data.use_balanced_sampling: true` (CRITICAL for full training)
- `data.prefetch_factor: 8` (v3.4.0: increased from 4)

### Training
- `training.batch_size: 12` (conservative for 24GB VRAM)
- `training.mixed_precision: false` (DISABLED - causes NaNs on RTX 4090)
- `training.loss: focal` (REQUIRED for 12:1 imbalance)
- `training.focal_gamma: 2.0` (with warmup in v3.4.1)
- `training.gradient_clip: 0.5` (v3.4.0: increased from 0.1)
- `training.learning_rate: 3.0e-5` (baseline for batch_size=12)

### Model - V3 Dual-Stream
- `model.architecture: v3`
- `model.graph.enabled: true`
- `model.graph.use_dynamic_pe: true` (ALWAYS enabled v3.3.1+)
- `model.graph.semi_dynamic_interval: 5` (optimal)
- `model.graph.edge_similarity_margin: 0.01` (v3.3.0 PR-5)

### Warmup Schedules (v3.4.1 - Optional but Recommended)
- `training.warmup_schedule.enabled: true`
- `training.warmup_schedule.warmup_steps: 1000`
- `training.warmup_schedule.adj_temperature_enabled: true`
- `training.warmup_schedule.focal_gamma_enabled: true`

### Environment Variables (CRITICAL)
```bash
export BGB_SANITIZE_GRADS=1  # Gradient NaN protection
export BGB_NAN_DEBUG=1       # Loss monitoring
```

## Complete V3.4.1 Configuration

```yaml
# Local RTX 4090: TCN + Bi-Mamba + GNN Stack (V3)
# From: configs/local/train.yaml

data:
  dataset: tuh_eeg
  data_dir: /mnt/ssd/tuh_eeg_seizure/v2.0.3/edf  # Adjust to your path
  cache_dir: cache/tusz                           # Local: train (4667) + dev (1832)
  split_policy: official_tusz
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  num_workers: 0                   # WSL2: MUST be 0
  pin_memory: false                # WSL2: MUST be false
  persistent_workers: false        # WSL2: MUST be false
  prefetch_factor: 8               # v3.4.0: Increased from 4
  use_balanced_sampling: true      # CRITICAL: Oversample seizures

preprocessing:
  montage: "10-20"
  bandpass: [0.5, 120]
  notch_freq: 60
  normalize: true
  use_mne: true

model:
  architecture: v3                 # V3 dual-stream ONLY

  # PR-1: Boundary Normalization (v3.3.0)
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true

  tcn:
    num_layers: 8
    kernel_size: 7
    dropout: 0.15
    causal: false
    stride_down: 16
    use_cuda_optimizations: true

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
    dropout: 0.1

  graph:
    enabled: true                  # Required for V3

    # PR-2: Bounded Edge Stream (v3.3.0)
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # V3 Edge Configuration
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16
    edge_similarity_margin: 0.01   # PR-5: Safety margin (v3.3.0)

    # PR-3: Adjacency Conditioning (v3.3.0)
    adj_row_softmax: true
    adj_softmax_tau: 1.0           # Warmup in v3.4.1
    adj_ema_beta: 0.95
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

    # GNN Configuration
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16

    # Dynamic PE (v3.3.1+: ALWAYS ENABLED)
    use_dynamic_pe: true           # Eigenvectors detached (gnn_pyg.py:205)
    semi_dynamic_interval: 5       # Update every 5 timesteps (optimal)
    pe_sign_consistency: true

training:
  epochs: 100
  batch_size: 12                   # Conservative for 24GB VRAM
  loss: focal                      # REQUIRED for class imbalance
  focal_alpha: 0.5
  focal_gamma: 2.0                 # Warmup from 1.0 in v3.4.1
  learning_rate: 3.0e-5
  weight_decay: 0.05
  optimizer: adamw
  scheduler:
    type: cosine
    warmup_ratio: 0.03

  # v3.4.1: Warmup Schedules (Optional but Recommended)
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0
    residual_scale_enabled: false

  gradient_clip: 0.5               # v3.4.0: Increased from 0.1
  mixed_precision: false           # DISABLED: NaNs on RTX 4090
  gradient_accumulation_steps: 1
  early_stopping:
    patience: 5
    metric: sensitivity_at_10fa
  checkpoint_interval: 5

experiment:
  name: v3_local_training
  description: "V3 dual-stream on RTX 4090"
  seed: 42
  device: cuda
  output_dir: results/v3_local
  log_level: INFO
  wandb:
    enabled: false               # Set to true with W&B API key
```

## Smoke Testing

Quick validation (3 files, 1 epoch):
```bash
export BGB_SMOKE_TEST=1          # Limit to 3 files
export BGB_SANITIZE_GRADS=1      # NaN protection
export BGB_NAN_DEBUG=1           # Loss monitoring

make s  # or: python -m src train configs/local/smoke.yaml
```

Config adjustments for smoke test (`configs/local/smoke.yaml`):
- `training.epochs: 1`
- `data.use_balanced_sampling: false` (small dataset doesn't need oversampling)

## WSL2 Considerations

**CRITICAL**: WSL2 has multiprocessing issues. The shipped `configs/local/train.yaml` uses WSL2-safe settings:

```yaml
data:
  num_workers: 0                # MUST be 0 on WSL2
  pin_memory: false             # MUST be false on WSL2
  persistent_workers: false     # MUST be false on WSL2
```

**Native Linux**: You can optimize with:
```yaml
data:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

## Expected Performance

| Platform | Batch Size | Time/Epoch | Total Training |
|----------|-----------|------------|----------------|
| RTX 4090 (WSL2) | 12 | ~2-3 hours | ~200-300 hours |
| RTX 4090 (Linux) | 12 | ~1.5-2 hours | ~150-200 hours |

**Smoke test**: ~5 minutes

## VRAM Usage

| Component | VRAM |
|-----------|------|
| Model (31M params) | ~6-8 GB |
| Batch (12 samples) | ~8-10 GB |
| Gradients + Optimizer | ~4-6 GB |
| **Total** | **~18-24 GB** |

**NOTE**: Batch size=12 is conservative. Some users report success with batch_size=16-18.

## Reference Configs

- **Smoke test**: `configs/local/smoke.yaml` (3 files, 1 epoch)
- **Full training**: `configs/local/train.yaml` (4667 train + 1832 dev files, 100 epochs)

## Common Issues

### NaN losses
**Solution**: Already addressed in v3.4.1
```bash
export BGB_SANITIZE_GRADS=1
export BGB_NAN_DEBUG=1
```

### Out of memory
**Solution**: Reduce batch size
```yaml
training:
  batch_size: 8  # or 6 if still OOM
```

### Zero seizures in batches
**Solution**: Enable balanced sampling
```yaml
data:
  use_balanced_sampling: true
```

### WSL2 DataLoader hangs
**Solution**: Disable multiprocessing
```yaml
data:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
```
