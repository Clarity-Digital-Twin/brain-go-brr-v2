# Modal Training (A100-80GB)

Commands

- Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
- Smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`

Resources

- `cpu: 24`, `memory: 98304`, `batch_size: 64`, `mixed_precision: true`

Cache and volumes

- Data (read-only) mounted at `/data` (S3); persistent SSD volume at `/results`.
- Ensure `data.cache_dir: /results/cache/tusz` in configs; app logs verify NPZs/manifest.

Resuming

- Use `--resume` flag or set `training.resume: true`.
- Training prioritizes `mid_epoch_*.pt` when resuming; falls back to `last.pt`.

Troubleshooting

- PyG/Mamba import issues: the image pins CUDA 12.1 + torch 2.2.2+cu121; rebuild if diverged.
- Slow/stuck at epoch boundaries: allocate 24 CPU and 96GB RAM (see function in `deploy/modal/app.py`).
- Logs: `modal app logs <app-id>`; stop: `modal app stop <app-id>`.
