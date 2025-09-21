# TUSZ Sampling Strategy — SeizureTransformer Style (Canonical)

Last updated: 2025-09-21

We pre-categorize cached windows into {no_seizure, full_seizure, partial_seizure} and build a fixed, reproducible training pool.

Formula (from SeizureTransformer):
- D = Dps ∪ D*fs ∪ D*ns
- Dps = ALL partial-seizure windows (0% < seizure < ~100%)
- D*fs = 0.3 × |Dps| randomly selected full-seizure windows (≈100%)
- D*ns = 2.5 × |Dps| randomly selected no-seizure windows (0%)

Implementation
- Manifest builder: `src/brain_brr/data/cache_utils.py` (scan_existing_cache)
  - Scans `.npz` files and writes `manifest.json` with relative filenames
  - Categorizes windows by seizure ratio; corrupted/missing labels handled
- Dataset: `src/brain_brr/data/datasets.py` (BalancedSeizureDataset)
  - Uses ALL partial + 0.3× full + 2.5× background; stable RNG; shuffles
- Training: `src/brain_brr/train/loop.py`
  - Uses BalancedSeizureDataset when manifest exists; fails fast if empty

CLI
- Build manifest for an existing cache:
  - `python -m src scan-cache --cache-dir cache/tusz/train`

Notes
- Full-seizure threshold uses ≥0.99 ratio to avoid edge effects
- Validation uses standard dataset (no balancing)

