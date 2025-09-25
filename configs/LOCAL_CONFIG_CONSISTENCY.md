# Local Config Consistency (V3) ✅

Canonical stack: TCN + Dual-Stream BiMamba + Vectorized GNN (PyG SSGConv + Static LPE)

Files
- configs/local/smoke.yaml — 1 epoch, 3 files (via BGB_LIMIT_FILES=3), quick validation
- configs/local/train.yaml — 100 epochs, 3734 files, balanced sampling

Shared model
```yaml
model:
  architecture: v3  # V3 dual-stream architecture

  tcn:
    num_layers: 8
    kernel_size: 7
    stride_down: 16
    dropout: 0.15

  mamba:  # Main temporal stream
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
    dropout: 0.1

  graph:
    enabled: true
    # PyG is required; no explicit toggle

    # V3 edge stream config:
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16  # Must be multiple of 8

    # GNN config:
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
```

Shared data
```yaml
data:
  dataset: tuh_eeg
  cache_dir: cache/tusz  # Critical: use existing cache
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
```

Smoke (safe + fast)
```yaml
epochs: 1
batch_size: 1  # Minimal for 3-file smoke test
use_balanced_sampling: false  # MUST be false for BGB_LIMIT_FILES
mixed_precision: false
# Requires: BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1
# Or use: ./run_smoke_test.sh
```

Train (RTX 4090 optimized)
```yaml
epochs: 100
batch_size: 8  # Reduced for V3 dual-stream memory
use_balanced_sampling: true  # Critical for seizure sampling
mixed_precision: false  # RTX 4090 FP16 causes NaNs
learning_rate: 5e-5  # Reduced for V3 stability
gradient_clip: 0.5  # Stronger for NaN protection
```

WSL2 notes
- Must use `num_workers: 0` (multiprocessing issues)
- Keep `pin_memory: false`, `persistent_workers: false`
- Full V3 training takes ~200-300 hours on RTX 4090

Key V3 improvements
- Node BiMamba: d_model=64, headdim=8 → (64*2)/8=16 ✓
- Edge BiMamba: d_model=16, headdim=4 → (16*2)/4=8 ✓
- No Conv1d fallbacks with proper headdim configuration
- Vectorized GNN processes all 960 timesteps in one pass
