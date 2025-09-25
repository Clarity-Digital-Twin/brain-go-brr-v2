# Modal Training (A100-80GB)

Commands

- Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
- Smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
- Clean old cache (run once after split fix): `modal run deploy/modal/app.py --action clean-cache`

Resources

- `cpu: 24`, `memory: 98304`, `batch_size: 64`, `mixed_precision: true`
- Prefer FP16 on A100 (tensor cores); keep dynamic PE enabled (default) — safeguards handle ill‑conditioned graphs.

Cache and volumes

- Data (read-only) mounted at `/data` (S3); persistent SSD volume at `/results`.
- Ensure `data.data_dir: /data/edf` and `data.split_policy: official_tusz`.
- Ensure `data.cache_dir: /results/cache/tusz` in configs; app logs verify NPZs/manifest.
- Cache is built directly on `/results` and reused across runs; do not copy to/from S3.

Patient disjointness

- On startup, the app verifies that patient sets in `/data/edf/train` and `/data/edf/dev` are disjoint and aborts if not.

S3 cache (not recommended)

- Do not rely on S3 for cache during training. Prefer building once on `/results` (persistent volume) and reusing.
- If you must bootstrap cache from local, use Modal volumes instead of S3:
  - `modal volume put brain-go-brr-results cache/tusz /results/cache/tusz`

Resuming

- Use `--resume` flag or set `training.resume: true`.
- Training prioritizes `mid_epoch_*.pt` when resuming; falls back to `last.pt`.

Troubleshooting

- PyG/Mamba import issues: the image pins CUDA 12.1 + torch 2.2.2+cu121; rebuild if diverged.
- Slow/stuck at epoch boundaries: allocate 24 CPU and 96GB RAM (see function in `deploy/modal/app.py`).
- Logs: `modal app logs <app-id>`; stop: `modal app stop <app-id>`.
