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
