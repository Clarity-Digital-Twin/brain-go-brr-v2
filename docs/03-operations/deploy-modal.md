Modal Deployment — Single Source of Truth

Principles
- Gate all runs on manifest: do not start training unless `partial>0` or `full>0`.
- Mount data (read-only) and results/cache (persistent) explicitly; avoid ephemeral paths.
- Kill long cache builds immediately if preflight fails; fix CSV/paths first.

Preflight (Modal)
1) Build image and ensure `make q` passes locally (same code goes into the image).
2) Mount EDF+CSV root to `/data/edf/train` (read-only) and `/results` as persistent volume.
3) Build or rescan manifest on `/results/cache/...`.
4) Require `partial>0` or `full>0`; otherwise STOP and fix.

Core commands (Modal CLI)
- Smoke test (safe default via local_entrypoint):
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_a100.yaml`
- Full training on A100:
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/train_a100.yaml`
- Resume training:
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/train_a100.yaml --resume true`
- Evaluate checkpoint:
  - `modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.pt`

Build/scan cache inside the container
- Build cache:
  - `python -m src build-cache --data-dir /data/edf/train --cache-dir /results/cache/tusz/train`
- Scan manifest:
  - `python -m src scan-cache --cache-dir /results/cache/tusz/train`

Cost control
- Build cache once; reuse across runs via `/results` volume.
- Always validate manifest before launching large GPU jobs.
- If batches show 0% seizures early: stop run and fix CSV/paths; re-scan manifest.

Observability & logging
- Real-time logs are enabled (PYTHONUNBUFFERED=1); add `flush=True` to prints for critical steps.
- Prefer W&B for cloud metrics; or write TensorBoard logs to `/results/runs`.
- Modal app control:
  - List: `modal app list`
  - Logs: `modal app logs <app-id>`
  - Stop: `modal app stop <app-id>`

CUDA/Mamba notes
- CUDA kernels coerce unsupported `d_conv` to 4 automatically.
- Force Conv1d fallback if needed: `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- The image compiles mamba-ssm from source against PyTorch 2.2.2+cu121.

Code anchors
- Modal entrypoint and functions: `deploy/modal/app.py` (uses `--action` local_entrypoint).
- Data pipeline docs: `../01-data-pipeline/*` (CSV_BI parsing, channels, cache+sampling).

Troubleshooting
- 0 windows in BalancedSeizureDataset: re-scan manifest; fix CSV_BI parser; rebuild cache.
- Slow cache build: ensure `/data` (read-only) and `/results` are on fast storage; avoid network latency.
- Memory errors: reduce batch size; verify A100 profile in config; prefer mixed precision.

