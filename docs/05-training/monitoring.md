# Monitoring and Storage

Local

- tmux: `tmux attach -t train` to view logs; detach with `Ctrl+B, D`.
- TensorBoard: `make tensorboard` (logs under `results/<run>/tensorboard`).

Modal

- List apps: `modal app list`
- Stream logs: `modal app logs <app-id>`
- Stop run: `modal app stop <app-id>`

Storage

- Modal cache on `/results/cache/tusz` (persistent SSD volume).
- Checkpoints under `results/<run>/checkpoints` (best.pt, last.pt, mid-epoch snapshots).

W&B (optional)

- Enable in config: `experiment.wandb.enabled: true`, set `project`, `entity`.
- Modal: attach a `wandb-secret` in app; local: export `WANDB_API_KEY`.

Mid-epoch checkpoints

- Control cadence via env vars:
  - `BGB_MID_EPOCH_MINUTES` — interval (minutes)
  - `BGB_MID_EPOCH_KEEP` — retention count
