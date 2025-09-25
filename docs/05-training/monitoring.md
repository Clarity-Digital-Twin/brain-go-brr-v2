# Monitoring and Storage

Local

- tmux: `tmux attach -t train` to view logs; detach with `Ctrl+B, D`.
- TensorBoard: `make tensorboard` (logs under `results/<run>/tensorboard`).

Modal

- List apps: `modal app list`
- Stream logs: `modal app logs <app-id>`
- Stop run: `modal app stop <app-id>`

Storage

- Modal cache on `/cache/` (S3 mount, read-only from `s3://brain-go-brr-eeg-data-20250919/cache/tusz/`).
- Checkpoints under `/results/<run>/checkpoints` (persistence volume for outputs only).

W&B (optional)

- Enable in config: `experiment.wandb.enabled: true`, set `project`, `entity`.
- Modal: attach a `wandb-secret` in app; local: export `WANDB_API_KEY`.

Mid-epoch checkpoints

- Control cadence via env vars:
  - `BGB_MID_EPOCH_MINUTES` — interval (minutes)
  - `BGB_MID_EPOCH_KEEP` — retention count
