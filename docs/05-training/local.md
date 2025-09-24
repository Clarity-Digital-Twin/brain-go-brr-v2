# Local Training (RTX 4090)

Quick commands

- Smoke: `make s`
- Full: `make train-local`

Recommendations

- `mixed_precision: false`
- `use_balanced_sampling: true`
- `batch_size: 12`
- WSL2: `num_workers: 0`

Monitoring

- Use tmux: `tmux new -s train` → run training → detach (Ctrl+B, D)

Quick fixes (if local training gets stuck or unstable)

- Dataloader hangs (WSL2): set `data.num_workers: 0` in your config.
- NaN losses (RTX 4090): set `training.mixed_precision: false`, reduce `batch_size`, consider lowering `learning_rate`.
- GPU OOM: reduce `batch_size` or enable gradient accumulation.
