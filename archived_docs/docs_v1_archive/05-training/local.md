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
- V3.2.0: set `model.graph.edge_similarity_margin: 0.01`

Recommended V3 profile (RTX 4090, 24GB)

- `training.batch_size: 8` (2x faster than batch=4, tested safe ~20GB VRAM)
- `model.graph.use_dynamic_pe: true`
- `model.graph.semi_dynamic_interval: 5`  (compute PE every 5 timesteps)
- `training.gradient_clip: 0.5`
- `training.warmup_ratio: 0.03`

Notes

- The above profile fits comfortably on 24GB VRAM with dynamic PE and has negligible accuracy impact vs full dynamic.
- If you absolutely need full dynamic (`semi_dynamic_interval: 1`) on 4090, reduce `batch_size` to ~4.
- Dataloaders yield dictionaries with `window`, `label`, `file_id`, and `window_start_s`; keep that structure when writing custom training scripts.

Cache and manifest

- Ensure `cache/tusz_mmap/train` contains the memory-mapped `*_data.npy`/`*_labels.npy` pairs plus `manifest.json`.
- Build or refresh via CLI if needed:
  - `python scripts/convert_cache_to_mmap.py --source cache/tusz/train --dest cache/tusz_mmap/train`
  - `python scripts/convert_cache_to_mmap.py --source cache/tusz/dev --dest cache/tusz_mmap/dev`
  - `python -m src scan-cache --cache-dir cache/tusz_mmap/train`
  - `python -m src scan-cache --cache-dir cache/tusz_mmap/dev`

Monitoring

- Use tmux: `tmux new -s train` → run training → detach (Ctrl+B, D)

Quick fixes (if local training gets stuck or unstable)

- Dataloader hangs (WSL2): set `data.num_workers: 0` in your config.
- NaN losses (RTX 4090): set `training.mixed_precision: false`, reduce `batch_size`, consider lowering `learning_rate`. Ensure `training.gradient_clip: 0.5` remains enabled. Optional debugging: `export BGB_NAN_DEBUG=1` (logging) and `export BGB_SANITIZE_GRADS=1` (zero-out/log non-finite gradients).
- GPU OOM: reduce `batch_size` or enable gradient accumulation.

Memory levers for dynamic PE

- Prefer `semi_dynamic_interval: 5–10` before turning dynamic PE off.
- If still tight on memory, reduce `batch_size` (linear memory scaling).
- A chunked PE path can further reduce memory, but is optional (not required for typical runs).

After crash or restart

- Resume: add `--resume` to `python -m src train ...` or set `training.resume: true` in config.
- **Mid-epoch checkpointing** (v3.6+):
  - **Recommended**: Set in config: `training.mid_checkpoint_interval_s: 1800`, `training.mid_epoch_keep: 3`
  - **Legacy (deprecated)**: Env vars `BGB_MID_EPOCH_MINUTES`, `BGB_MID_EPOCH_KEEP` (config takes precedence)

Pre‑flight checklist (recommended before long runs)

- Run quality and config validation: `make q` and `python -m src validate configs/local/train.yaml`.
- Verify cache and manifest: `python -m src scan-cache --cache-dir cache/tusz_mmap/train` → ensure partial>0 or full>0.
- Confirm BalancedSeizureDataset logs appear at startup (see Data docs for expected lines).
- WSL2: set `data.num_workers: 0` if you see dataloader hangs.
- RTX 4090: keep `training.mixed_precision: false` if you see NaNs; reduce LR or batch size if needed.
