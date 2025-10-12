# Local Configs (RTX 4090)

**Last Updated**: October 8, 2025 (v3.9.0)
**Target Hardware**: RTX 4090 24GB (WSL2 or native Linux)

## Overview

V3 dual-stream architecture is the **only supported** architecture. V2 heuristic path has been removed.

## Key Recommendations (v3.9.0)

### Data
- `data.cache_dir: cache/tusz_mmap` (memory-mapped NPY cache on local SSD/NVMe)
- `data.num_workers: 0` (WSL2 stability) or `4` (native Linux with `pin_memory: true`)
- `data.use_balanced_sampling: true` (CRITICAL for full training)
- `data.prefetch_factor: 2` (conservative for memory stability)
- Run `scripts/convert_cache_to_mmap.py` + `python -m src scan-cache --cache-dir cache/tusz_mmap/train` if you are migrating from the legacy NPZ cache.

### Training
- `training.batch_size: 8` (measured safe on 24GB VRAM; double throughput vs batch 4)
- `training.mixed_precision: false` (DISABLED - causes NaNs on RTX 4090)
- `training.loss: focal` (REQUIRED for 12:1 imbalance)
- `training.focal_gamma: 2.0` (with warmup in v3.4.1)
- `training.gradient_clip: 0.5` (v3.4.0: increased from 0.1)
- `training.learning_rate: 1.0e-4` (increased from 1e-5 for warmup stability)

### Model - V3 Dual-Stream
- `model.architecture: v3`
- `model.graph.enabled: true`
- `model.graph.use_dynamic_pe: true` (ALWAYS enabled v3.3.1+)
- `model.graph.semi_dynamic_interval: 5` (optimal)
- `model.graph.edge_similarity_margin: 0.01` (v3.3.0 PR-5)

### Warmup Schedules (Optional but Recommended)
- `training.warmup_schedule.enabled: true`
- `training.warmup_schedule.warmup_steps: 1000`
- `training.warmup_schedule.adj_temperature_enabled: true`
- `training.warmup_schedule.focal_gamma_enabled: true`

### Optional Environment Variables
```bash
export BGB_NAN_DEBUG=1         # Additional NaN logging
# export BGB_SANITIZE_GRADS=1  # Optional debugging helper
```

## Complete v3.9.0 Configuration

```yaml
# Local RTX 4090: TCN + Bi-Mamba + GNN Stack (V3)
# From: configs/local/train_bimamba.yaml

data:
  dataset: tuh_eeg
  data_dir: /mnt/ssd/tuh_eeg_seizure/v2.0.3/edf  # Adjust to your path
  cache_dir: cache/tusz_mmap                      # Memory-mapped cache (train/dev pairs)
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  num_workers: 0                   # WSL2: MUST be 0 (set to 4 on native Linux)
  pin_memory: true                 # Fast GPU transfer
  persistent_workers: false        # Keep false when num_workers=0 (set true if >0 workers)
  prefetch_factor: 2               # Conservative for memory stability
  use_balanced_sampling: true      # CRITICAL: Oversample seizures

# The loader enforces official TUSZ train/dev splits automatically; legacy
# split_policy/validation_split fields were removed in V4.

preprocessing:
  montage: "10-20"
  bandpass: [0.5, 120]
  notch_freq: 60
  normalize: true

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
  batch_size: 4                    # Conservative for 24GB VRAM
  loss: focal                      # REQUIRED for class imbalance
  focal_alpha: 0.5
  focal_gamma: 2.0                 # Warmup from 1.0 in v3.4.1
  learning_rate: 1.0e-4            # Increased from 1e-5 for warmup stability
  weight_decay: 0.01               # Reduced from 0.05 to prevent weight explosion
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
export BGB_NAN_DEBUG=1           # Loss monitoring (optional)
# export BGB_SANITIZE_GRADS=1    # Optional debugging helper

make smoke-bimamba  # or: python -m src train configs/local/smoke_bimamba.yaml
# FLA stack: make smoke-fla (uses configs/local/smoke_fla.yaml)
```

Config adjustments for smoke test (`configs/local/smoke_bimamba.yaml`; identical in `smoke_fla.yaml`):
- `training.epochs: 1`
- `data.use_balanced_sampling: false` (small dataset doesn't need oversampling)

## WSL2 Considerations

**CRITICAL**: WSL2 has multiprocessing issues. The shipped `configs/local/train_bimamba.yaml` (and `train_fla.yaml`) use WSL2-safe settings:

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
| RTX 4090 (WSL2) | 4 | ~3-4 hours | ~300-400 hours |
| RTX 4090 (Linux) | 4 | ~2-3 hours | ~200-300 hours |

**Smoke test**: ~5 minutes

**Note**: Smaller batch size is more stable but slower. Can experiment with batch_size=8-12 if no OOM occurs.

## VRAM Usage

| Component | VRAM |
|-----------|------|
| Model (31M params) | ~0.12 GB (FP32) |
| Batch (4 samples) | ~2.5-5 GB |
| Gradients + Optimizer | ~5-8 GB |
| **Total** | **~10-16 GB** |

**NOTE**: Batch size=4 leaves plenty of headroom. Can experiment with batch_size=8-12 for faster training if VRAM allows.

## Reference Configs

- **Smoke test**: `configs/local/smoke_bimamba.yaml` (3 files, 1 epoch)  
- **Full training**: `configs/local/train_bimamba.yaml` (4667 train + 1832 dev files, 100 epochs)  
- **FLA variants**: `configs/local/smoke_fla.yaml`, `configs/local/train_fla.yaml`

## Common Issues

### NaN losses
**Solution**: Ensure cache is up to date and gradient clipping remains enabled. Optional debugging:
```bash
export BGB_NAN_DEBUG=1         # Log NaN/Inf tensors
# export BGB_SANITIZE_GRADS=1  # Zero/log non-finite gradients while investigating
```

### Out of memory
**Solution**: Reduce batch size (already conservative at 4)
```yaml
training:
  batch_size: 2  # If batch_size=4 still causes OOM
```

**Alternative**: Increase batch size if you have headroom
```yaml
training:
  batch_size: 8   # or 12 if VRAM allows
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
