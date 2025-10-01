# Smoke Tests

Local

- `make s` — 1 epoch, 3 files (`BGB_SMOKE_TEST=1`)

Modal

- `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`

Environment variables

- `BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1` — limit data and enable smoke shortcuts.
- Keep `training.mixed_precision: false` for stability on RTX 4090.
- Prefer `data.num_workers: 0` on WSL2.

What to look for

- Logs confirm BalancedSeizureDataset usage when manifest exists, or fallback to EEGWindowDataset.
- No NaNs; batch loop progresses; small checkpoints written if enabled.

Troubleshooting

- If manifest empty in smoke: the training loop falls back automatically; you can also run `python -m src scan-cache --cache-dir cache/tusz/train`.
