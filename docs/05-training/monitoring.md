# Monitoring and Storage

Local

- tmux: `tmux attach -t train` to view logs; detach with `Ctrl+B, D`.
- TensorBoard: `make tensorboard` (logs under `results/<run>/tensorboard`).

Modal

- List apps: `modal app list`
- Stream logs: `modal app logs <app-id>`
- Stop run: `modal app stop <app-id>`

Storage

- Modal cache at `/results/cache/tusz_mmap/` (persistent SSD volume; no S3 mount). Run `modal run deploy/modal/app.py --action check-cache` after populate-cache to validate counts and manifests.
- Checkpoints under `/results/<run>/checkpoints` (persistent volume for outputs only). Look for `mid_epoch_*.pt`, `last.pt`, and `timeout_exit.pt` after the timeout guard triggers.

W&B (optional)

- Enable in config: `experiment.wandb.enabled: true`, set `project`, `entity`.
- Modal: attach a `wandb-secret` in app; local: export `WANDB_API_KEY`.

Mid-epoch checkpoints

- **Recommended (v3.6+)**: Set in config YAML:
  - `training.mid_checkpoint_interval_s` — interval (seconds)
  - `training.mid_epoch_keep` — retention count
- **Legacy (deprecated)**: Env vars `BGB_MID_EPOCH_MINUTES`, `BGB_MID_EPOCH_KEEP` (backward compatible, config takes precedence)
