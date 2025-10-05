# Checkpointing and Resume

- Resume: add `--resume true` to training command
- Checkpoints: `results/<run>/` (`best.pt`, `last.pt`)

Tmux tips

- Attach: `tmux attach -t train` ; Detach: `Ctrl+B, D` ; Stop: `Ctrl+C`

Details

- The training loop prefers mid-epoch checkpoints named `mid_epoch_*.pt` when `training.resume: true`.
- If no mid-epoch snapshot exists, it loads `last.pt`; `best.pt` is by metric.
- **Mid-epoch cadence and retention** (v3.6+):
  - **Recommended**: Set in config: `training.mid_checkpoint_interval_s` (seconds), `training.mid_epoch_keep` (count)
  - **Legacy (deprecated)**: Env vars `BGB_MID_EPOCH_MINUTES`, `BGB_MID_EPOCH_KEEP` (backward compatible)
  - Config fields take precedence over env vars

Examples

- Local: `python -m src train configs/local/train.yaml --resume`
- Modal: `modal run deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true`
