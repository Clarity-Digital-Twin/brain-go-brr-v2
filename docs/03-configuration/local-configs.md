# Local Configs (RTX 4090)

Recommended

- `data.cache_dir: cache/tusz`
- `data.num_workers: 0` (WSL2 fix)
- `training.batch_size: 12`
- `training.mixed_precision: false`
- `training.use_balanced_sampling: true`

Configs

- Smoke: `configs/local/smoke.yaml`
- Full: `configs/local/train.yaml`
