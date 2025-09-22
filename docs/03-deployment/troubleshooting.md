Deployment Troubleshooting

Modal vs Local divergence
- Symptom: 15–25s/batch locally; GPU util ~10–20%.
  - Cause: DataLoader starvation (`num_workers=0`, `pin_memory=false`).
  - Fix: Prefer `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=2` on WSL2 ext4. Fall back to 0/false only if hangs.

- Symptom: On Modal, `BalancedSeizureDataset failed: No partial seizure windows found ...` and training falls back to `EEGWindowDataset`.
  - Cause: Stale/empty manifest in `/results/cache/...` or path mismatch.
  - Fix: Training now auto‑validates and deletes bad manifests, then rebuilds from existing cache. To force a rebuild, set `BGB_FORCE_MANIFEST_REBUILD=1` at launch.

- Symptom: Manifest shows 0 windows despite `.npz` files present.
  - Cause: Early empty manifest persisted; tqdm issues in subprocess.
  - Fix: On Modal we disable tqdm (`BGB_DISABLE_TQDM=1`). Rebuild manifest with `python -m src scan-cache --cache-dir <cache_dir>` or force via env variable above.

Zero seizures detected (manifest)
- Symptom: `partial=0, full=0, none=N` after scan-cache; training collapses to all-negative
- Fixes:
  - Verify CSV_BI parsing (channel,start,stop,label,confidence)
  - Ensure seizure type set includes {gnsz,fnsz,cpsz,absz,spsz,tcsz,tnsz,mysz}
  - Rebuild cache; re-scan; only train if partial>0 or full>0

Hangs or deadlocks (WSL2)
- Cause: multiprocessing DataLoader
- Fix: `num_workers=0`; avoid pin_memory; keep cache/data on WSL ext4

CUDA/Mamba kernel issues
- Symptom: errors about d_conv sizes
- Fix: kernels coerce unsupported d_conv to 4; or set `SEIZURE_MAMBA_FORCE_FALLBACK=1`

Mamba CUDA kernels failing on Modal (CRITICAL)
- Symptom: `'NoneType' object is not callable` when calling Mamba2 layers
- Cause: causal-conv1d CUDA kernels not compiled/installed properly
- Root issues:
  1. Modal's PyPI mirror serves wrong PyTorch version (2.8.0 vs required 2.2.2)
  2. mamba-ssm requires causal-conv1d package with compiled CUDA kernels
  3. Build isolation prevents access to installed PyTorch during compilation
- Fix in deploy/modal/app.py:
  1. Force exact PyTorch version: `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
  2. Set CUDA env vars BEFORE installs: CUDA_HOME, PATH, LD_LIBRARY_PATH
  3. Install with --no-build-isolation: `pip install --no-build-isolation causal-conv1d==1.4.0`
  4. Same for mamba-ssm: `pip install --no-build-isolation mamba-ssm==2.2.2`
  5. Add verification test in image build to ensure kernels work
- Verification: Run `modal run deploy/modal/app.py --action test-mamba` before training

EDF read failure
- Symptom: MNE ValueError/OSError on EDF header
- Fix: header repair path; see ../01-data-pipeline/tusz-edf-repair.md

Slow training on Modal (48s per batch)
- Cause: S3 CloudBucketMount has terrible random access performance (100-700ms per file)
- Fix: Cache optimizer automatically copies NPZ files from S3 to local volume (30-60 min one-time)
- Verify: Look for "[CACHE] Local cache already exists with X NPZ files"
- Force re-optimize: Set `BGB_FORCE_CACHE_COPY=1`
- Impact: 10x speedup (48s → 5s per batch)

Slow IO on WSL
- Fix: keep local caches on ext4; avoid network-mounted paths

W&B not logging
- Symptom: 404 error on W&B dashboard, no runs showing
- Cause: WandBLogger not instantiated or wrong entity name
- Fix: Entity must be username (e.g., `jj-vcmcswaggins`), not team name
- Verify: WANDB_API_KEY in Modal secrets, `wandb.enabled: true` in config

Sampler used with balanced dataset
- Symptom: WeightedRandomSampler still applied
- Fix: training loop bypasses sampler for BalancedSeizureDataset; verify config's balanced flag and logs

Observability & logging (Modal)
- Set unbuffered output (already set): `PYTHONUNBUFFERED=1`.
- Add `flush=True` to key prints (epoch start/end, validation metrics, progress).
- Periodic progress: print every N batches (e.g., every 100) with loss and lr.
- Add a simple heartbeat (every ~5 minutes) to avoid long silent periods.
- Metrics:
  - TensorBoard (local and cloud): write to `/results/runs` and optionally serve.
  - Weights & Biases: initialize `wandb.init(...)` and log `train/` and `val/` scalars.
- Modal CLI tips:
  - List apps: `modal app list`
  - Stream logs: `modal app logs <app-id>`
  - Stop runaway job: `modal app stop <app-id>`

Manual controls (quick)
- Force manifest rebuild on next train run: `BGB_FORCE_MANIFEST_REBUILD=1` (local or Modal env).
- Rebuild manifest explicitly: `python -m src scan-cache --cache-dir <cache_dir>`.
- Pre-build cache: `python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>`.
