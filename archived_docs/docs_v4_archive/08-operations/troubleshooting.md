# Troubleshooting

Common issues

- Wrong cache dir: local `cache/tusz_mmap/`, Modal `/results/cache/tusz_mmap/` (persistent SSD)
- Dev manifest stale: `modal run deploy/modal/app.py --action check-cache`, delete bad manifest, relaunch (auto rebuild)
- No seizures in batches: enable `use_balanced_sampling`
- NaN losses on 4090: set `mixed_precision: false`
- Modal stuck: increase CPU (24) and RAM (96GB)
- Resume reruns the previous epoch: first resume after Oct 10 checkpoint fix will replay the last completed epoch (old checkpoints stored the current epoch index). Let it finish; new checkpoints contain `epoch+1` so future resumes jump ahead.
- PyG install fails: use prebuilt wheels
- CI/CD: If PyG isn’t installed in the workflow, tests that require it are skipped by markers; install cu121 wheels to exercise GNN tests.
- Evaluate CLI exits early:
  - "No config found" — pass `--config` or use a checkpoint with embedded config
  - "No EDF files found" — verify `*.edf` exist under the provided data path

Operational P0 checklist (must pass)

- GPU stack versions match exactly (Torch 2.5.0+cu124, mamba-ssm 2.2.5, PyG 2.6.1 from cu124 wheels). Install order: `make setup` then `make setup-gpu`.
- Cache present with seizures and manifest built: run `python -m src scan-cache --cache-dir <cache_split_dir>`; expect partial>0 or full>0.
- Dynamic PE deps present if `graph.enabled: true`: `pip show torch_geometric` should list installed wheels.
- After preprocessing changes (e.g., ±10σ clip), rebuild cache: `rm -rf cache/tusz_mmap && python -m src build-cache ...` (or rerun the conversion pipeline).

V3 NaN issues

- Primary contributing factor: dynamic PE eigendecomposition on poorly initialized adjacency.
- Default configs enable dynamic PE with safeguards; on consumer GPUs (RTX 4090), prefer increasing `graph.semi_dynamic_interval` first; as a last resort set `use_dynamic_pe: false`.
- Additional safeguards:
  - Edge similarity is clamped at the source with a configurable margin: `graph.edge_similarity_margin` (default 0.01)
  - Edge lift uses bounded activation + normalization (tanh + LayerNorm) by default (PR‑2). A conservative `[-3,3]` fallback only applies if PR‑2 is disabled.
  - Optimizer parameter groups (no weight decay on norms/bias)
  - Optional gradient sanitization (`BGB_SANITIZE_GRADS=1`)
- Details and timeline: `docs/08-operations/v3-nan-explosion-resolution.md`

Local training “gets stuck” checklist

- WSL2 dataloader: set `data.num_workers: 0` to avoid multiprocessing hangs.
- RTX 4090 NaNs: set `training.mixed_precision: false`; optionally reduce `learning_rate` or `batch_size`.
- Excessive CPU usage on Modal: ensure `resources.cpu: 24` and `resources.memory: 98304`.

Pre‑flight (before long runs)

- `make q` and `python -m src validate <config>` pass.
- `python -m src scan-cache --cache-dir <cache>` shows partial>0 or full>0.
- Startup logs show `BalancedSeizureDataset` and `Seizure ratio: ...`.

OOM root cause quick summary

- Full dynamic PE computes 960 eigendecompositions per window; the CUDA workspace across B×T can add several GB.
- Remedies, in order: increase `semi_dynamic_interval`, reduce `batch_size`, or (as a last resort) disable dynamic PE.

WSL2 OOM/driver artifact note

- If you see impossible memory in logs like `17179869184.00 GiB`, that is a reporting artifact after a hard OOM or driver fault.
- Fix: `wsl --shutdown` to reset the VM/GPU state, then relaunch. Also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` during smokes.

RTX 4090 SIGBUS / driver crash near batch ~3000

- Symptom: Training dies abruptly with `Signal 7 (SIGBUS)` or hard driver reset around batch 2900–3100 despite healthy metrics beforehand.
- Root cause: NVIDIA driver 572.xx (January 2025) has confirmed stability bugs on Ada GPUs; RTX 4090 requires the October 2025 branch.
- Fix: Upgrade Windows host driver to **581.42** (Game Ready) and perform a clean install. After reboot, `nvidia-smi` inside WSL2 should show `Driver Version: 581.42`.
- Verification: Re-run the affected workload; the crash disappears and long epochs complete normally.

CI note (PyG optional in unit tests)

- The test suite skips GNN‑dependent tests when `torch_geometric` is absent. To validate GNN paths in CI, install PyG using prebuilt wheels for torch 2.5.0+cu124.

NaN logits root cause quick summary

- Tiny batches (e.g., 1) and aggressive FP16 can amplify numerical noise in dynamic PE and post‑Mamba projections.
- Remedies, in order: increase `batch_size` (≥4), set `training.mixed_precision: false` on RTX 4090, reduce `learning_rate`.
- Env toggles: `export BGB_NAN_DEBUG=1` (extra logging), `export SEIZURE_MAMBA_FORCE_FALLBACK=1` (force Conv1d fallback).

Modal cache hygiene

- Always use the mmap cache at `/results/cache/tusz_mmap/` (populate from S3 once, then reuse).
- After deployment run `modal run deploy/modal/app.py --action check-cache` to validate train/dev counts and manifest health.
- If stray NPZ files are reported, clean them with `modal run deploy/modal/clean_stray_npz.py --confirm` (script verifies matching `_data.npy/_labels.npy` first).
- Results and checkpoints live under `/results/`; keep raw data mounted at `/data/edf`.
- Ensure `data.data_dir: /data/edf`; the loader enforces official patient-disjoint splits automatically (legacy `split_policy` field was removed in V4).

## Emergency Recovery Procedures

### Training Shows Zero Seizures
**Symptom**: "Seizure ratio: 0%" in logs
**Cause**: Missing or corrupted manifest file
```bash
# Validate Modal cache manifests first
modal run deploy/modal/app.py --action check-cache

# Force manifest rebuild locally
BGB_FORCE_MANIFEST_REBUILD=1 python -m src scan-cache --cache-dir cache/tusz_mmap/train

# Re-upload to S3 (including JSONs!)
aws s3 sync cache/tusz_mmap/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/ \
  --exclude "*.log" --exclude "__pycache__/*"

# Verify manifest uploaded
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/manifest.json

# Re-run Modal populate-cache
modal run --detach deploy/modal/app.py --action populate-cache
```

### Balanced Dataset Falls Back to `EEGWindowDataset`
**Symptom**: Startup log shows `[WARNING] BalancedSeizureDataset failed: ... falling back to EEGWindowDataset`
**Cause**: Train manifest missing/mismatched or a build older than v3.9.1 that passed EDF filenames into the balanced sampler
```bash
# 1. Stop the run immediately – training without the balanced sampler under-exposes seizures.

# 2. Inspect cache health on Modal
modal run deploy/modal/app.py --action check-cache

# 3. If manifest is missing/corrupt, rebuild locally and re-upload
python -m src scan-cache --cache-dir cache/tusz_mmap/train
aws s3 sync cache/tusz_mmap/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/ \
  --exclude "*.log" --exclude "__pycache__/*"
modal run --detach deploy/modal/app.py --action populate-cache

# 4. Ensure you are running v3.9.1+ (loop.py now maps EDF → *_data.npy before filtering)

# 5. Relaunch training and confirm the healthy log line:
#    [DATASET] BalancedSeizureDataset: <N> windows from manifest
```

### `Runner failed with exception: Worker disappeared`
**Symptom**: Modal restarts the container within seconds during the first 10–15 min of a run
**Cause**: Transient GPU reset/spot preemption while Triton kernels compile (Modal automatically respawns)
```bash
# 1. Allow Modal to restart; the new container will re-enter the loop automatically.
# 2. Expect the first healthy container to spend ~5–10 min building /results/cache/tusz_mmap/dev/_dataset_index.json
#    (one-time cost; the file is reused for future restarts).
# 3. Optionally lower the initial mid-epoch interval to 600 s so a checkpoint exists before the next restart.
# 4. If the error repeats multiple times, collect container IDs and open a Modal support ticket.
```

### Modal populate-cache Stops
**Symptom**: Function exits after ~8 minutes
**Cause**: Terminal disconnection without --detach
```bash
# Always use --detach for long-running commands
modal run --detach deploy/modal/app.py --action populate-cache

# Alternative: Use tmux for safety
tmux new -s populate
modal run deploy/modal/app.py --action populate-cache
# Ctrl+B then D to detach
tmux attach -t populate  # To reattach
```

### Modal training exits after ~23 h
**Symptom**: Logs show `[TIMEOUT] Wall-clock limit approaching …` followed by `[TIMEOUT] Saved timeout_exit.pt, resume with --resume flag`
**Cause**: Timeout guard (v3.9.0+) exiting cleanly before Modal’s 24 h limit
```bash
# Relaunch training with resume flag (loads timeout_exit.pt or latest mid_epoch_*.pt)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume
```

This is expected; no progress is lost beyond the last 10–30 minutes.

### Cache Corruption
**Symptom**: Unexpected errors during data loading
```bash
# Clean Modal cache completely
modal run deploy/modal/app.py --action clean-cache

# Delete local cache
rm -rf cache/tusz_mmap/

# Rebuild from scratch (takes 2-3 hours)
make train-local  # Will rebuild cache automatically

# Upload fixed cache to S3
aws s3 sync cache/tusz_mmap/ s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/ \
  --exclude "*.log" --exclude "__pycache__/*"

# Repopulate Modal
modal run --detach deploy/modal/app.py --action populate-cache
```

### Manifest Issues Quick Reference
| Issue | Solution |
|-------|----------|
| No train manifest | `python -m src scan-cache --cache-dir cache/tusz_mmap/train` |
| No dev manifest | `python -m src scan-cache --cache-dir cache/tusz_mmap/dev` (optional) |
| Stale manifest | `BGB_FORCE_MANIFEST_REBUILD=1` before scan-cache |
| Upload excluded JSONs | Remove `--exclude "*.json"` from S3 sync |
| Verify manifest size | Train: ~27MB, Dev: ~13MB |
