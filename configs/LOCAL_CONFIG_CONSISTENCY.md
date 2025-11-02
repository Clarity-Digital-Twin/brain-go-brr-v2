# Local Config Consistency (v4.0.0) ✅

Two dedicated config pairs exist:

```
configs/local/
  smoke_bimamba.yaml       # BiMamba2: 3 files, 1 epoch (BGB_SMOKE_TEST)
  train_bimamba.yaml       # BiMamba2: 100 epochs, official train/dev
  smoke_fla.yaml           # FLA (Gated DeltaNet): smoke
  train_fla.yaml           # FLA (Gated DeltaNet): full run (baseline)
  train_fla_exp1_reg.yaml  # FLA: stronger regularization (Exp1, completed)
  train_fla_exp2_lr.yaml   # FLA: lower LR (Exp2, optional)
  train_fla_exp3_smaller.yaml # FLA: smaller model (Exp3, archived)
  train_fla_exp4_cyclic.yaml  # FLA: cyclic LR restarts (Exp4, next run)
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

### Production Baseline (`train_fla.yaml`)
- `model.mamba.temporal_type: gated_deltanet` (applied to both node and edge streams)
- `gdn_fusion_mode: sum`, `gdn_allow_neg_eigval: false`
- `gdn_edge_num_heads: 3`, `gdn_edge_headdim: 8`
- `graph.edge_mamba_d_model: 32` (FLA causal_conv1d requirement)
- `output_dir: results/local_fla_training`
- Usage: `make train-fla` or `.venv/bin/python -m src train configs/local/train_fla.yaml`

### Hyperparameter Experiments (Research)
All experiments maintain FLA stack settings above but vary regularization/capacity:

**Exp1 - Stronger Regularization (`train_fla_exp1_reg.yaml`)**
- `model.tcn.dropout: 0.20` (UP from 0.15)
- `model.mamba.dropout: 0.2` (UP from 0.1)
- `model.graph.dropout: 0.2` (UP from 0.1)
- `training.weight_decay: 0.05` (UP from 0.01)
- `output_dir: results/local_fla_exp1_reg` ✅ ISOLATED
- Usage: `.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml`

**Exp2 - Lower Learning Rate (`train_fla_exp2_lr.yaml`)**
- `training.learning_rate: 5.0e-5` (DOWN from 1e-4)
- `training.scheduler.warmup_ratio: 0.05` (UP from 0.03)
- `output_dir: results/local_fla_exp2_lr` ✅ ISOLATED
- Usage: `.venv/bin/python -m src train configs/local/train_fla_exp2_lr.yaml`

**Exp3 - Smaller Model (`train_fla_exp3_smaller.yaml`)**
- `model.mamba.n_layers: 4` (DOWN from 6)
- `model.mamba.d_model: 384` (DOWN from 512)
- `model.graph.n_layers: 1` (DOWN from 2)
- `graph.edge_mamba_d_model: 32` **UNCHANGED** (FLA requirement)
- Reduces 31M → 17M parameters
- `output_dir: results/local_fla_exp3_smaller` ✅ ISOLATED
- Usage: `.venv/bin/python -m src train configs/local/train_fla_exp3_smaller.yaml`
- **Status**: Archived (capacity reduction deemed unnecessary)

**Exp4 - Cyclic LR (`train_fla_exp4_cyclic.yaml`)**
- `training.scheduler.type: cosine_restarts` (SGDR with warm restarts)
- `training.scheduler.t_initial: 10`, `t_mult: 2`, `eta_min: 1.0e-6`
- `training.early_stopping.patience: 15` (faster verdict with restarts)
- `output_dir: results/local_fla_exp4_cyclic` ✅ ISOLATED
- Usage: `.venv/bin/python -m src train configs/local/train_fla_exp4_cyclic.yaml`
- **Status**: Ready to launch when baseline early-stops (~epoch 36)

**Current Status (Nov 2025):**
- Baseline (`train_fla.yaml`): Running, plateaued at 0.257 (best 0.284 @ epoch 9)
- Exp1: Completed (negative result, confirmed model not overfitting)
- Exp4: Validated and scheduled immediately after baseline completes
- Exp2: Optional follow-up if Exp4 fails to improve plateau
- Exp3: Archived (no longer under consideration)

## WSL2 / RTX 4090 Notes
- Always run with `num_workers: 0`.
- Use `BGB_NAN_DEBUG=1` (and optionally `BGB_SANITIZE_GRADS=1`) for debugging.
- Mid-epoch checkpoints prevent >30 min progress loss on 3h epochs.
- Full local runs remain slow (~12.5 days); prefer Modal for long training.

## Architecture Sanity Checks
- Node BiMamba vs GatedDeltaNet share the same `d_model=512`, `n_layers=6`.
- Edge stream uses `d_model=16` (BiMamba2) or `d_model=32` (FLA) with matching head heuristics (`0.75×d_model`).
- Dynamic Laplacian PE enabled (`use_dynamic_pe: true`, `semi_dynamic_interval: 5`).
