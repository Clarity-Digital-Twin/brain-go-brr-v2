# Cache Layout and Manifests

## Dataset Strategy (CRITICAL - This is CORRECT ML practice, not a bug!)

- **Training**: Uses `BalancedSeizureDataset` with manifest to oversample seizures (8% → ~30% in batches)
  - Why: Model needs enough seizures to learn patterns effectively
  - Requires: `train/manifest.json` (auto-created if missing)
- **Validation**: Uses `EEGWindowDataset` with natural distribution (~8% seizures)
  - Why: Measures real-world performance on true distribution, not inflated metrics
  - Manifest: `dev/manifest.json` is OPTIONAL (validation doesn't use it)
- **This is standard ML practice**: Train on balanced data to learn, validate on real distribution to measure

Locations

- Local: `cache/tusz/{train,dev}`  # CRITICAL: We use 'dev' to match TUSZ's official naming!
- Modal: `/results/cache/tusz/{train,dev}` (persistent SSD volume; no S3 mount)
- S3 (intermediate): `s3://brain-go-brr-eeg-data-20250919/cache/tusz/`

NPZ schema

- File name: `<edf_stem>_windows.npz`
- Arrays:
  - `windows`: `(n_windows, 19, 15360)` float32
  - `labels` (optional): `(n_windows, 15360)` float32 (binary mask)

Index cache

- `_dataset_index.json` stored alongside NPZs for fast dataset startup, containing file list and window counts.

Manifest (`manifest.json`)

- Built by scanning NPZs once; stored at split root.
- Keys: `partial_seizure`, `full_seizure`, `no_seizure`
- Each entry: `{ "cache_file": "<name>.npz", "window_idx": <int> }`
- Used by `BalancedSeizureDataset` to compose dataset and compute exact `seizure_ratio`.

Balanced recipe

- Keep ALL partial seizure windows.
- Add 0.3× as many full seizure windows.
- Add 2.5× as many background windows.

Class weighting

- Training derives `pos_weight = sqrt((1 - seizure_ratio) / seizure_ratio)` using the dataset’s exact `seizure_ratio`.

CLI commands

- Build cache: `python -m src build-cache --data-dir <edf_dir> --cache-dir <cache_split_dir> --split train`
- Scan and manifest: `python -m src scan-cache --cache-dir <cache_split_dir>`
  - The CLI also accepts `--split dev` (preferred) and `--split val` as a backward-compatible alias for `dev`.

Training behavior

- On startup, training validates or rebuilds manifest if empty/stale; switches to `BalancedSeizureDataset` when available.
- Fallback to `EEGWindowDataset` is automatic if balanced creation fails in smoke test.

Env flags

- `BGB_FORCE_MANIFEST_REBUILD=1` — delete and rebuild stale manifest on run
- `BGB_SMOKE_TEST=1` — limit to 3 files; disables expensive sampling paths
- `BGB_LIMIT_FILES=N` — cap file count for quick runs

Performance impact (Modal)

- Using the manifest’s exact `seizure_ratio` eliminates the historical “sample 1000 windows” step.
- Observed improvement: 2+ hours → < 1 second on Modal (≈7200× faster), while producing identical `pos_weight`.

Verification logs

Expect lines like:

```
[DATASET] BalancedSeizureDataset: XXXX windows from manifest
[DATASET] Using BalancedSeizureDataset known distribution
[DATASET] Seizure ratio: 34.2% (from manifest)
[DATASET] Using pos_weight: 1.39 (sqrt scaling)
```

When to rebuild the manifest

- Parser/label set changes (e.g., adding/removing seizure types)
- Windowing or preprocessing changes (size, stride, filters)
- Channel mapping changes
- Corrupted or stale manifest detected

Verification checklist

- Counts present per split:
  - `ls <cache_root>/train/*.npz | wc -l` (expect thousands for full)
  - `ls <cache_root>/dev/*.npz | wc -l` (expect hundreds for full)
- Dataset index exists: `<cache_root>/{train,dev}/_dataset_index.json`
- Train manifest exists (REQUIRED): `<cache_root>/train/manifest.json`
- Dev manifest exists (OPTIONAL but recommended): `<cache_root>/dev/manifest.json`
- Training logs show: "BalancedSeizureDataset" and non-zero seizure ratio
- Split policy in effect: logs show "OFFICIAL TUSZ SPLITS" and "✅ PATIENT DISJOINTNESS VERIFIED"

Modal paths (persistent volume)

- Train cache: `/results/cache/tusz/train`
- Dev cache: `/results/cache/tusz/dev`
- Results: `/results/` (metrics, checkpoints, logs)

## Complete Cache Workflow (Local → S3 → Modal)

### Phase 1: Local Cache Building
Cache builds automatically on first training run or manually via CLI:
```bash
# Automatic (during training)
make train-local  # Takes 2-3 hours on RTX 4090

# Manual (CLI)
python -m src build-cache --data-dir data/edf --cache-dir cache/tusz/train --split train
python -m src build-cache --data-dir data/edf --cache-dir cache/tusz/dev --split dev
```

### Phase 2: Manifest Generation (CRITICAL)
```bash
# Create train manifest (REQUIRED for balanced sampling!)
python -m src scan-cache --cache-dir cache/tusz/train

# Create dev manifest (optional but recommended)
python -m src scan-cache --cache-dir cache/tusz/dev
```

### Phase 3: S3 Upload (Fixed Script)
```bash
# IMPORTANT: The old script excluded JSON files. Use this fixed version:
aws s3 sync cache/tusz/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ \
  --exclude "*.log" --exclude "__pycache__/*" --exclude ".DS_Store"
aws s3 sync cache/tusz/dev/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ \
  --exclude "*.log" --exclude "__pycache__/*" --exclude ".DS_Store"

# Verify manifests uploaded
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/manifest.json
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/manifest.json
```

### Phase 4: Modal Cache Population
```bash
# CRITICAL: Always use --detach or it stops when terminal closes!
modal run --detach deploy/modal/app.py --action populate-cache

# Monitor progress
modal app list
modal app logs ap-XXXXXXXXXXXXXXXXXXXXX  # Replace with actual app-id
```

### Phase 5: Training
```bash
# Smoke test first
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

## Emergency Recovery Procedures

### If training shows zero seizures
```bash
# Force manifest rebuild locally
BGB_FORCE_MANIFEST_REBUILD=1 python -m src scan-cache --cache-dir cache/tusz/train

# Re-upload to S3 (including JSONs!)
# Then re-run populate-cache on Modal
```

### If Modal populate-cache stops
```bash
# Always use --detach
modal run --detach deploy/modal/app.py --action populate-cache

# Or run in tmux for safety
tmux new -s populate
modal run deploy/modal/app.py --action populate-cache
# Ctrl+B then D to detach
```

### If cache is corrupted
```bash
# Clean Modal cache
modal run deploy/modal/app.py --action clean-cache

# Delete local cache
rm -rf cache/tusz/

# Rebuild from scratch
make train-local  # Will rebuild cache
```

## Cost Implications

- **Cache building locally**: Free (uses your GPU)
- **S3 storage**: ~$10/month for 450GB
- **S3 egress**: ~$40 per full transfer to Modal
- **Modal populate-cache**: ~$0.50-1.00 (CPU compute)
- **Modal training without cache**: +$100-300 WASTED on A100 rebuilding

**ALWAYS populate cache before training to avoid waste!**

Notes

- Do not mount caches from S3 for training; keep NPZ caches on the Modal persistent SSD for performance and stability.
- Total data size: ~450GB preprocessed cache (306GB train + 143GB dev)
- Expected file counts: 4667 train NPZ files, 1832 dev NPZ files
