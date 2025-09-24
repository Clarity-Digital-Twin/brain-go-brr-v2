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
