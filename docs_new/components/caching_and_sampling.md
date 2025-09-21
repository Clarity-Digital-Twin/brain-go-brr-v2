Caching & Sampling (Manifest + Balanced)

Scope
- Cache NPZ layout, seizure categorization manifest, and BalancedSeizureDataset.

Manifest
- Built by scanning NPZs: categorizes each window by seizure ratio.
- Categories:
  - no_seizure: ratio == 0.0
  - full_seizure: ratio ≥ 0.99 (robust to edge effects)
  - partial_seizure: 0.0 < ratio < 0.99
- Storage: manifest.json next to NPZs; uses relative filenames for portability.

Cache payload (per NPZ)
- windows: float32, shape (n_windows, 19, 15360)
- labels: float32, shape (n_windows, 15360) — aggregated binary masks
- Optional: indices/metadata (file ids, window starts) for reconstruction

BalancedSeizureDataset (SeizureTransformer)
- Composition: ALL partial + 0.3× full + 2.5× no‑seizure.
- Deterministic sampling: numpy RNG with fixed seed; deterministic shuffle.
- Guards: fails fast if partial pool is empty; training aborts if dataset length is 0.
- Train uses balanced dataset; val/test use standard dataset (no balancing).

CLI
- Build cache (and manifest): `python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>`
- Scan existing cache: `python -m src scan-cache --cache-dir <cache_dir>`
- Training auto‑uses BalancedSeizureDataset if manifest exists and config enables it.

Code anchors
- src/brain_brr/data/cache_utils.py (scan_existing_cache)
- src/brain_brr/data/datasets.py (BalancedSeizureDataset)
- src/brain_brr/train/loop.py (dataset selection + guards)
- src/brain_brr/cli/cli.py (build-cache, scan-cache)

Docs
- TUSZ/CACHE_AND_SAMPLING.md
- TUSZ/PREFLIGHT_AND_TROUBLESHOOTING.md

Related Phase
- phases/PHASE1_DATA_PIPELINE.md (sampling section)

Notes
- Require partial>0 or full>0 before any training run.
- Manifest must remain colocated with NPZs under the chosen cache_dir.
