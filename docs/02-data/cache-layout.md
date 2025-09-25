# Cache Layout and Manifests

Locations

- Local: `cache/tusz/{train,dev}`
- Modal: `/cache/{train,dev}` (S3 mount from `s3://brain-go-brr-eeg-data-20250919/cache/tusz/`)

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
- Manifest exists and non-empty: `<cache_root>/{train,dev}/manifest.json`
- Training logs show: "BalancedSeizureDataset" and non-zero seizure ratio
- Split policy in effect: logs show "OFFICIAL TUSZ SPLITS" and "✅ PATIENT DISJOINTNESS VERIFIED"

Modal paths (S3 mount)

- Train: `/cache/train` (mounted from S3)
- Dev: `/cache/dev` (mounted from S3)
- Results: `/results/` (persistent volume for outputs only)
