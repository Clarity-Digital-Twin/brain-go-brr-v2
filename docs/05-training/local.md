# Local Training (RTX 4090)

Quick commands

- Smoke: `make s`
- Full: `make train-local`

Validate config before running

- `python -m src validate configs/local/train.yaml`
- Optional: `--phase data|model|training`

Recommendations

- `mixed_precision: false`
- `use_balanced_sampling: true`
- `batch_size: 12`
- WSL2: `num_workers: 0`

Cache and manifest

- Ensure `cache/tusz/train` contains NPZs and `manifest.json`.
- Build via CLI if needed:
  - `python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train`
  - `python -m src scan-cache --cache-dir cache/tusz/train`

Monitoring

- Use tmux: `tmux new -s train` → run training → detach (Ctrl+B, D)

Quick fixes (if local training gets stuck or unstable)

- Dataloader hangs (WSL2): set `data.num_workers: 0` in your config.
- NaN losses (RTX 4090): set `training.mixed_precision: false`, reduce `batch_size`, consider lowering `learning_rate`.
- GPU OOM: reduce `batch_size` or enable gradient accumulation.

After crash or restart

- Resume: add `--resume` to `python -m src train ...` or set `training.resume: true` in config.
- Mid-epoch checkpointing (optional): set env vars `BGB_MID_EPOCH_MINUTES` and `BGB_MID_EPOCH_KEEP`.

Pre‑flight checklist (recommended before long runs)

- Run quality and config validation: `make q` and `python -m src validate configs/local/train.yaml`.
- Verify cache and manifest: `python -m src scan-cache --cache-dir cache/tusz/train` → ensure partial>0 or full>0.
- Confirm BalancedSeizureDataset logs appear at startup (see Data docs for expected lines).
- WSL2: set `data.num_workers: 0` if you see dataloader hangs.
- RTX 4090: keep `training.mixed_precision: false` if you see NaNs; reduce LR or batch size if needed.
