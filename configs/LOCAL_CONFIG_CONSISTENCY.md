# Local Config Consistency (v4.0.0) ✅

Two dedicated config pairs exist:

```
configs/local/
  smoke_bimamba.yaml   # BiMamba2: 3 files, 1 epoch (BGB_SMOKE_TEST)
  train_bimamba.yaml   # BiMamba2: 100 epochs, official train/dev
  smoke_fla.yaml       # FLA (Gated DeltaNet): smoke
  train_fla.yaml       # FLA (Gated DeltaNet): full run
```

Both stacks share the same TCN → GNN pipeline; only the temporal blocks differ.

## Common Settings (Train Configs)

```yaml
data:
  dataset: tuh_eeg
  cache_dir: cache/tusz_mmap
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  use_balanced_sampling: true   # CRITICAL for imbalanced data
  num_workers: 0                # WSL2 stability
  pin_memory: true              # Faster GPU transfer
  persistent_workers: false     # Must be false when num_workers=0
  prefetch_factor: 2

training:
  batch_size: 8
  learning_rate: 1.0e-4
  gradient_clip: 0.5
  mixed_precision: false     # RTX 4090 FP16 can cause NaNs
  scheduler:
    type: cosine
    warmup_ratio: 0.03

# Checkpoints (train configs only):
mid_checkpoint_interval_s: 1800
mid_epoch_keep: 3
```

**Smoke Config Differences**:
- `use_balanced_sampling: false` (must be false for BGB_LIMIT_FILES to work)
- `pin_memory: false` (optimization not needed for short runs)
- No mid-epoch checkpoints (1 epoch only)

## BiMamba2 Stack (`*_bimamba.yaml`)

- `model.mamba` has **no `temporal_type` overrides** (defaults to BiMamba2).
- `graph.edge_mamba_d_model: 16`.
- Usage:
  - `make smoke-bimamba` → 3 files, 1 epoch
  - `make train-bimamba` → full 100 epochs (≈300h locally)

## FLA Stack (`*_fla.yaml`)

- `model.mamba.temporal_type: gated_deltanet` (applied to both node and edge streams)
- `gdn_fusion_mode: sum`, `gdn_allow_neg_eigval: false`
- `gdn_edge_num_heads: 3`, `gdn_edge_headdim: 8`
- `graph.edge_mamba_d_model: 32` (FLA causal_conv1d requirement)
- All other hyperparameters match the BiMamba2 configs.
- Usage:
  - `make smoke-fla`
  - `make train-fla`

## WSL2 / RTX 4090 Notes
- Always run with `num_workers: 0`.
- Use `BGB_NAN_DEBUG=1` (and optionally `BGB_SANITIZE_GRADS=1`) for debugging.
- Mid-epoch checkpoints prevent >30 min progress loss on 3h epochs.
- Full local runs remain slow (~12.5 days); prefer Modal for long training.

## Architecture Sanity Checks
- Node BiMamba vs GatedDeltaNet share the same `d_model=512`, `n_layers=6`.
- Edge stream uses `d_model=16` (BiMamba2) or `d_model=32` (FLA) with matching head heuristics (`0.75×d_model`).
- Dynamic Laplacian PE enabled (`use_dynamic_pe: true`, `semi_dynamic_interval: 5`).
