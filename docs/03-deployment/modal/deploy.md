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
5) Manifest sanity: training auto-validates the existing manifest and deletes/rebuilds if it is empty or stale (see troubleshooting). You can force a rebuild via `BGB_FORCE_MANIFEST_REBUILD=1`.

Core commands (Modal CLI)
- Smoke test (detached):
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_a100.yaml`
- Full training on A100 (detached):
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/train_a100.yaml`
- Resume training:
  - `modal run --detach deploy/modal/app.py --action train --config configs/modal/train_a100.yaml --resume true`
- Evaluate checkpoint:
  - `modal run deploy/modal/app.py::evaluate --checkpoint-path /results/tusz_a100_100ep/checkpoints/best.pt`

Build/scan cache inside the container
- Build cache:
  - `python -m src build-cache --data-dir /data/edf/train --cache-dir /results/cache/tusz/train`
- Scan manifest:
  - `python -m src scan-cache --cache-dir /results/cache/tusz/train`
 - Force rebuild on next train run:
   - Set env `BGB_FORCE_MANIFEST_REBUILD=1` when launching training

Cost control
- Build cache once; reuse across runs via `/results` volume.
- **CRITICAL**: Optimize cache to local volume (saves 90% compute cost!)
- Always validate manifest before launching large GPU jobs.
- If batches show 0% seizures early: stop run and fix CSV/paths; re-scan manifest.
- **With optimizations**: ~100 hours ($319) instead of ~1000 hours ($3,190)

Observability & logging
- Real-time logs are enabled (PYTHONUNBUFFERED=1); add `flush=True` to prints for critical steps.
- Prefer W&B for cloud metrics; or write TensorBoard logs to `/results/runs`.
- Modal app control:
  - List: `modal app list`
  - Logs: `modal app logs <app-id>`
  - Stop: `modal app stop <app-id>`
 - Training emits manifest checks: look for `[CACHE]`, `[DATA] Built manifest ...`, and `[DATASET] BalancedSeizureDataset ...` lines

CUDA/Mamba notes (CRITICAL - Must Match Local Setup)
- CUDA kernels coerce unsupported `d_conv` to 4 automatically.
- Force Conv1d fallback if needed: `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- The image MUST compile mamba-ssm from source against PyTorch 2.2.2+cu121.
- CRITICAL VERSION REQUIREMENTS:
  - torch==2.2.2+cu121 (NOT 2.8.0 from Modal's mirror!)
  - causal-conv1d==1.4.0 (provides CUDA kernels for Mamba)
  - mamba-ssm==2.2.2 (must compile against exact torch version)
- Test Mamba CUDA before training: `modal run deploy/modal/app.py --action test-mamba`
- If you see "'NoneType' object is not callable" errors, the CUDA kernels failed to compile
- Required exact versions (from setup-guide.md):
  - PyTorch 2.2.2+cu121 (NOT 2.8.0 from Modal mirror!)
  - mamba-ssm==2.2.2 (NOT 2.2.4/2.2.5 which have bugs)
  - causal-conv1d==1.4.0 (1.5+ requires PyTorch 2.4+)
  - numpy<2.0 (2.x breaks mamba-ssm)
- Test before training: `modal run deploy/modal/app.py --action test-mamba`
- If Mamba fails with `'NoneType' object is not callable`, see troubleshooting.md

Code anchors
- Modal entrypoint and functions: `deploy/modal/app.py` (uses `--action` local_entrypoint).
- Data pipeline docs: `../01-data-pipeline/*` (CSV_BI parsing, channels, cache+sampling).

Storage Architecture
- **S3 Bucket**: `brain-go-brr-eeg-data-20250919` - Raw EDF files only
- **S3 Mount**: `/data/edf/train/` - CloudBucketMount for reading raw data
- **Modal Volume**: `brain-go-brr-results` (310GB) at `/results/`
  - `/results/cache/tusz/train/` - 3734 NPZ files on SSD (built on first run)
  - `/results/checkpoints/` - Model checkpoints
  - `/results/wandb/` - W&B logs
- **Key fact**: Cache is built directly to Modal SSD, never touches S3

Performance Settings (A100 Optimized)
- **Batch size**: 128 (uses full 80GB VRAM)
- **Mixed precision**: true (A100 is 3.8x faster at FP16 than FP32)
- **W&B entity**: Team name if using team API key (e.g., `jj-vcmcswaggins-novamindnyc`)
- **Expected performance**: ~5s/batch, ~1hr/epoch, ~100hrs total (~$319)

Troubleshooting
- 0 windows in BalancedSeizureDataset: re-scan manifest; fix CSV_BI parser; rebuild cache.
- **Slow training (48s/batch)**: Cache not optimized! Check `/results/cache/tusz/train` exists
- Memory errors: reduce batch size; verify A100 profile in config; prefer mixed precision.
- Empty/stale manifest on `/results`: training now validates/deletes bad manifests and rebuilds from existing cache automatically. To force a rebuild regardless, set `BGB_FORCE_MANIFEST_REBUILD=1`.
- **W&B not logging**: Check entity name in config and WANDB_API_KEY in secrets
 - See also: `./troubleshooting.md` → "Modal vs Local divergence".
