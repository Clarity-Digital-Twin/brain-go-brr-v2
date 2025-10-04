# Local Config Consistency (V3.6.1) ✅

Canonical stack: TCN + Dual-Stream BiMamba + Vectorized GNN (PyG SSGConv + Dynamic LPE)

Files
- configs/local/smoke.yaml — 1 epoch, 3 files (via BGB_LIMIT_FILES=3), quick validation
- configs/local/train.yaml — 100 epochs, official train/dev splits, balanced sampling

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
    edge_similarity_margin: 0.01  # v3.2.0: Safety margin from ±1 boundaries

    # GNN config:
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
    use_dynamic_pe: true  # Dynamic PE (recomputed per timestep)
    semi_dynamic_interval: 5  # RTX 4090 optimized interval
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
batch_size: 8  # Same as train for consistency
use_balanced_sampling: false  # MUST be false for BGB_LIMIT_FILES
mixed_precision: false
# Optional: export BGB_NAN_DEBUG=1 for debugging
# Or use: make s (sets flags automatically)
```

Train (RTX 4090 optimized)
```yaml
epochs: 100
batch_size: 8                  # OPTIMIZED: 2× faster than batch=4
use_balanced_sampling: true    # Critical for severe imbalance
mixed_precision: false         # RTX 4090 FP16 can cause NaNs
learning_rate: 1.0e-4          # Conservative for stability
gradient_clip: 0.5             # Increased from 0.1 (eigendecomp fix)
mid_checkpoint_interval_s: 1800  # Save every 30 min
mid_epoch_keep: 3              # Keep last 3 mid-epoch snapshots
```

WSL2 notes
- Use `num_workers: 0` (multiprocessing issues)
- Keep `pin_memory: true`, `persistent_workers: false`
- Mid-epoch checkpoints save every 30 min (critical for 3-hour epochs)
- Full V3 training ~300 hours; prefer Modal for speed (~100 hours)

Key V3 improvements
- Node BiMamba: d_model=64, headdim=8 → (64*2)/8=16 ✓
- Edge BiMamba: d_model=16, headdim=4 → (16*2)/4=8 ✓
- No Conv1d fallbacks with proper headdim configuration
- Vectorized GNN processes all 960 timesteps in one pass
