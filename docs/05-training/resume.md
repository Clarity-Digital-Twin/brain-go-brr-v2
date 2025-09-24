# Checkpointing and Resume

- Resume: add `--resume true` to training command
- Checkpoints: `results/<run>/` (`best.pt`, `last.pt`)

Tmux tips

- Attach: `tmux attach -t train` ; Detach: `Ctrl+B, D` ; Stop: `Ctrl+C`

Details

- The training loop prefers mid-epoch checkpoints named `mid_epoch_*.pt` when `training.resume: true`.
- If no mid-epoch snapshot exists, it loads `last.pt`; `best.pt` is by metric.
- Mid-epoch cadence and retention are controlled via env vars:
  - `BGB_MID_EPOCH_MINUTES` — minutes between mid-epoch saves
  - `BGB_MID_EPOCH_KEEP` — number of mid-epoch snapshots to retain

Examples

- Local: `python -m src train configs/local/train.yaml --resume`
- Modal: `modal run deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true`
