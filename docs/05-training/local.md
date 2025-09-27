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
- WSL2: `num_workers: 0`

Recommended V3 profile (RTX 4090, 24GB)

- `training.batch_size: 4`
- `model.graph.use_dynamic_pe: true`
- `model.graph.semi_dynamic_interval: 5`  (compute PE every 5 timesteps)
- `training.gradient_clip: 0.1`
- `training.warmup_ratio: 0.01`

Notes

- The above profile fits comfortably on 24GB VRAM with dynamic PE and has negligible accuracy impact vs full dynamic.
- If you absolutely need full dynamic (`semi_dynamic_interval: 1`) on 4090, reduce `batch_size` to ~3.

Cache and manifest

- Ensure `cache/tusz/train` contains NPZs and `manifest.json`.
- Build via CLI if needed:
  - `python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train`
  - `python -m src scan-cache --cache-dir cache/tusz/train`

Monitoring

- Use tmux: `tmux new -s train` → run training → detach (Ctrl+B, D)

Quick fixes (if local training gets stuck or unstable)

- Dataloader hangs (WSL2): set `data.num_workers: 0` in your config.
- NaN losses (RTX 4090): set `training.mixed_precision: false`, reduce `batch_size`, consider lowering `learning_rate`. For early epochs, you may enable `BGB_SANITIZE_GRADS=1` and (optionally) `BGB_SAFE_CLAMP=1` to absorb initial spikes, then disable once stable.
- GPU OOM: reduce `batch_size` or enable gradient accumulation.

Memory levers for dynamic PE

- Prefer `semi_dynamic_interval: 5–10` before turning dynamic PE off.
- If still tight on memory, reduce `batch_size` (linear memory scaling).
- A chunked PE path can further reduce memory, but is optional (not required for typical runs).

After crash or restart

- Resume: add `--resume` to `python -m src train ...` or set `training.resume: true` in config.
- Mid-epoch checkpointing (optional): set env vars `BGB_MID_EPOCH_MINUTES` and `BGB_MID_EPOCH_KEEP`.

Pre‑flight checklist (recommended before long runs)

- Run quality and config validation: `make q` and `python -m src validate configs/local/train.yaml`.
- Verify cache and manifest: `python -m src scan-cache --cache-dir cache/tusz/train` → ensure partial>0 or full>0.
- Confirm BalancedSeizureDataset logs appear at startup (see Data docs for expected lines).
- WSL2: set `data.num_workers: 0` if you see dataloader hangs.
- RTX 4090: keep `training.mixed_precision: false` if you see NaNs; reduce LR or batch size if needed.
