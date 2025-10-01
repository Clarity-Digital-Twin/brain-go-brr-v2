# Modal (A100-80GB) Configs

**Last Updated**: October 1, 2025 (v3.4.1)
**Target Hardware**: A100-80GB on Modal cloud

## Overview

V3 dual-stream architecture optimized for A100-80GB with critical XID 31 and hang detection fixes (v3.4.1).

## Key Differences vs Local

| Setting | Local (RTX 4090) | Modal (A100) | Reason |
|---------|------------------|--------------|--------|
| **Batch size** | 12 | 64 | 80GB vs 24GB VRAM |
| **Mixed precision** | false | true | A100 tensor cores (3.8x faster) |
| **Learning rate** | 3.0e-5 | 8.0e-5 | Batch-size scaled |
| **Gradient clip** | 0.5 | 0.5 | Same (both stable) |
| **Num workers** | 0 (WSL2) | 8 | Parallel I/O on cloud |
| **Prefetch factor** | 8 | 8 | v3.4.0: Both increased |
| **CPU cores** | N/A | 24 | CRITICAL: Avoid bottleneck |
| **Memory** | N/A | 98304 MB (96 GB) | CRITICAL: 3x safety margin |

## Resources (deploy/modal/app.py)

```python
@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    memory=98304,  # 96GB RAM (3x for safety)
    cpu=24,        # 24 cores (3 per DataLoader worker)
)
```

**CRITICAL**: Default resources are too low. Always use 24 CPU + 96GB RAM.

## Complete V3.4.1 Configuration

### Data Configuration
```yaml
data:
  dataset: tuh_eeg
  data_dir: /data/edf                    # Read-only dataset mount
  cache_dir: /results/cache/tusz         # Persistent SSD volume
  split_policy: official_tusz            # Patient-disjoint splits
  num_workers: 8                         # Parallel loading
  pin_memory: true                       # Fast GPU transfer
  persistent_workers: true               # Reuse workers
  prefetch_factor: 8                     # v3.4.0: Increased from 4
  use_balanced_sampling: true            # CRITICAL: Oversample seizures
```

### Training Configuration
```yaml
training:
  batch_size: 64                         # A100-80GB optimized
  mixed_precision: true                  # CRITICAL: 3.8x faster
  loss: focal                            # REQUIRED for 12:1 imbalance
  focal_alpha: 0.5
  focal_gamma: 2.0                       # Warmup from 1.0 in v3.4.1
  learning_rate: 8.0e-5                  # Batch-scaled from 3e-5
  gradient_clip: 0.5

  # v3.4.1: Warmup Schedules (Enabled)
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
```

### Model Configuration (V3 Dual-Stream)
```yaml
model:
  architecture: v3

  # PR-1: Boundary Normalization
  norms:
    boundary_norm: layernorm
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true
    layerscale_alpha: 0.1

  graph:
    enabled: true

    # PR-2: Bounded Edge Stream
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # Edge Configuration
    edge_features: cosine
    edge_similarity_margin: 0.01         # PR-5: Safety margin
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16

    # PR-3: Adjacency Conditioning
    adj_row_softmax: true
    adj_softmax_tau: 1.0                 # Warmup in v3.4.1
    adj_ema_beta: 0.95
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

    # Dynamic PE (ALWAYS ENABLED)
    use_dynamic_pe: true                 # Eigenvectors detached
    semi_dynamic_interval: 5             # Optimal for A100
    pe_sign_consistency: true
```

## Critical Fixes (v3.4.1)

### XID 31 GPU Crash Prevention
**Implemented in**: `deploy/modal/app.py:541-551`

```python
# Memory allocator optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Unique Triton cache per run (prevents stale kernels)
run_id = str(uuid.uuid4())[:8]
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_run_{run_id}"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/tii_cache_run_{run_id}"
```

### Hang Detection & Logging
**Implemented in**: `deploy/modal/app.py:722-723`, `src/brain_brr/train/loop.py:440`

```python
# Modal auto-sets these environment variables:
BGB_LOG_EVERY_N_STEPS=10      # Log every 10 batches (vs 50)
BGB_SANITIZE_GRADS=1          # Gradient NaN protection
BGB_NAN_DEBUG=1               # Loss monitoring

# Training loop uses faster heartbeat:
heartbeat_interval = 120       # 2 minutes (vs 5 minutes)
```

## Commands

### Training
```bash
# Smoke test (50 files, quick validation)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke.yaml

# Full training (4667+1832 files, 100 epochs)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train.yaml
```

**CRITICAL**: Always use `--detach` for long-running training. Modal terminates after 8 minutes otherwise.

### Cache Management
```bash
# Populate cache (one-time, from S3)
modal run --detach deploy/modal/app.py --action populate-cache

# Clean cache (if needed)
modal run deploy/modal/app.py --action clean-cache

# Verify cache
modal volume ls seizure-detection-data results/cache/tusz
```

### Monitoring
```bash
# List running apps
modal app list

# Stream logs
modal app logs <app-id>

# Stop training
modal app stop <app-id>
```

## Initialization Timeline

**Total: ~75 minutes** before first epoch starts. This is **NORMAL**, not a hang.

| Phase | Duration | Description |
|-------|----------|-------------|
| Startup | ~1 min | Container launch |
| Train manifest | ~11 min | Load 61,616 windows |
| Dev manifest | **~54 min** | Load 148,224 windows (single-threaded Python) |
| Model init | ~10 sec | Create 31M parameters |
| W&B init | ~3 sec | Connect to dashboard |
| **W&B appears** | **~65 min** | When dashboard shows up |
| Preflight | ~10 min | Test forward/backward |
| **Epoch starts** | **~75 min** | Training begins |

**Patience required**: Don't cancel before 75 minutes!

## Expected Performance

| Metric | Value |
|--------|-------|
| **Time/epoch** | ~1 hour |
| **Total training** | ~100 hours (100 epochs) |
| **Cost** | ~$319 (at $3.19/hour A100) |
| **Smoke test** | ~5 minutes |
| **VRAM usage** | ~40-60 GB (out of 80 GB) |

## Stack Versions

**Pinned in**: `deploy/modal/app.py:16-100`

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.4.0 | Exact match required |
| PyTorch | 2.5.0+cu124 | From PyTorch index |
| mamba-ssm | 2.2.5 | With PR #708 patch |
| causal-conv1d | 1.5.2 | Latest stable |
| PyTorch Geometric | 2.6.1 | Pre-built wheels |
| numpy | <2.0 | 2.x breaks mamba-ssm |

**NOTE**: mamba-ssm is patched at build time with PR #708 int64 casts (fixes XID 31).

## Reference Configs

- **Smoke test**: `configs/modal/smoke.yaml` (50 files, 10 dev, 1 epoch)
- **Full training**: `configs/modal/train.yaml` (4667 train, 1832 dev, 100 epochs)

## Common Issues

### Training appears stuck during initialization
**Expected behavior**: 75-minute initialization is NORMAL
- Check logs: `modal app logs <app-id>`
- Look for: `[DATASET] BalancedSeizureDataset: XXXX windows`
- W&B appears at ~65 min
- See `docs_v2/05-training/modal.md` for detailed timeline

### XID 31 GPU crashes
**Status**: Fixed in v3.4.1
- Memory allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Unique Triton cache per run
- See `docs_v2/RECENT-WORK-SYNTHESIZED/03-incidents/MODAL_XID31_RECURRENCE.md`

### Zero seizures in batches
**Solution**: Ensure balanced sampling enabled
```yaml
data:
  use_balanced_sampling: true
```

### Slow epoch boundaries
**Solution**: Use 24 CPU + 96GB RAM (already default in v3.4.1)

### PyG/Mamba import errors
**Solution**: Check Modal image build logs for compilation errors
- Image rebuilds automatically when code changes
- Mamba patch is applied during build (deploy/modal/app.py:49-91)
