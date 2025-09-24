# Cache Layout and Manifests

Locations

- Local: `cache/tusz/{train,val}`
- Modal: `/results/cache/tusz/{train, val}` (persistent SSD)

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
