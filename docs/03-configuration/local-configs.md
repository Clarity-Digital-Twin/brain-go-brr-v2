# Local Configs (RTX 4090)

Defaults to V3 dual‑stream (configs may still list V2 for ablations).

Key recommendations

- `data.cache_dir: cache/tusz`
- `data.num_workers: 0` (WSL2 stability); if native Linux, try 4 with `pin_memory: true` and `persistent_workers: true`.
- `training.mixed_precision: false` (RTX 4090 NaN stability)
- `data.use_balanced_sampling: true` for full runs; false for smoke (BGB_LIMIT_FILES)
- `model.graph.use_dynamic_pe: true` (recommended for V3)

Optimal V3 (RTX 4090) snippet

```yaml
training:
  batch_size: 4
  mixed_precision: false
  gradient_clip: 0.5
  warmup_ratio: 0.10
model:
  architecture: v3
  graph:
    enabled: true
    use_dynamic_pe: true
    semi_dynamic_interval: 5
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
```

Minimal V3 snippet

```yaml
model:
  architecture: v3
  tcn:
    num_layers: 8
    kernel_size: 7
    stride_down: 16
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
  graph:
    enabled: true
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
```

Smoke testing

- Use `BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1` and set `data.use_balanced_sampling: false`.

WSL2 note

- The shipped `configs/local/train.yaml` currently sets `num_workers: 4` as a starting point. If you observe hangs or deadlocks on WSL2, set `num_workers: 0` and rerun.

Reference configs

- Smoke: `configs/local/smoke.yaml`
- Full: `configs/local/train.yaml`
