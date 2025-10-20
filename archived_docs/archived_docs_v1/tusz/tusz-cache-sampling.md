Cache, Manifest, Balanced Sampling (SeizureTransformer)

Objective

- Guarantee seizure representation in training with a reproducible, manifest-driven dataset.

Cache building

- Windowing: 60s windows, 10s stride (83% overlap), 256 Hz, per-channel z-score
- Cache format (v3.8.0+): `<basename>_data.npy` + `<basename>_labels.npy` (memory-mapped, zero-copy)
- Legacy format (deprecated): `<basename>_windows.npz` (compressed, high RAM usage)
- Labels: per-window time-step binary mask built from CSV_BI events

Manifest creation (categorization)

- `scan_existing_cache(cache_dir)` reads cache files (NPY pairs or legacy NPZ) and categorizes windows:
  - no_seizure: ratio == 0.0
  - full_seizure: ratio ≥ 0.99
  - partial_seizure: 0.0 < ratio < 0.99
- Output: `manifest.json` with relative filenames for portability
- Guard: warn/fail if zero partial/full (do not proceed to train)

BalancedSeizureDataset (exact formula)

- D = Dps ∪ D*fs ∪ D*ns
- Dps: ALL partial seizure windows
- D*fs: randomly select 0.3 × |Dps| full-seizure windows
- D*ns: randomly select 2.5 × |Dps| no-seizure windows
- Deterministic: numpy Generator with fixed seed for selection and shuffle

Training integration

- Train: BalancedSeizureDataset when manifest exists and non-empty (~30% seizures after balancing)
- Validation: ValidationDataset (natural ~8% seizure distribution, no balancing) for realistic metrics
- Fail-fast: exit if balanced dataset length is zero

Commands

See README.md for quick command reference.

Code anchors

- src/brain_brr/data/cache_utils.py: scan_existing_cache
- src/brain_brr/data/datasets.py: BalancedSeizureDataset
- src/brain_brr/train/loop.py: dataset selection and guards

