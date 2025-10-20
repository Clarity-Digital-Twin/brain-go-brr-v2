TUSZ: Canonical Overview

Audience: engineers implementing data I/O, caching, sampling, and training on TUSZ.

What matters most

- Use the CSV_BI annotation format (channel,start,stop,label,confidence) correctly.
- Build windows+labels cache and then a categorized manifest (partial/full/no-seizure) before training.
- Train with BalancedSeizureDataset (ALL partial + 0.3× full + 2.5× background) — SeizureTransformer-style.
- Enforce preflight checks: if the manifest has zero seizures, stop and fix inputs.

Pipeline (high level)

1) EDF+CSV_BI → parse → per-sample binary labels
2) Windowing: 60s windows, 10s stride (256 Hz), per-channel z-score
3) Cache: memory-mapped NPY pairs (`*_data.npy` + `*_labels.npy`) for zero-copy loading
4) Manifest: scan cache and categorize windows
5) Dataset: BalancedSeizureDataset from manifest for train; ValidationDataset for dev/test

Cache Format (v3.8.0+):
- Production: `filename_data.npy` + `filename_labels.npy` (memory-mapped, <1 GB RAM)
- Legacy: `filename_windows.npz` (compressed, deprecated, 387 GB RAM for full dataset)
- Manifest references NPY files directly for instant loading (99.6% faster startup)

For a step-by-step mapping to code, see DATA_FLOW.md

Key specs (implemented)

- Sampling: ALL partial seizure windows + 0.3× full + 2.5× no-seizure
- Full seizure threshold: ratio ≥ 0.99 in a window
- Partial seizure: 0 < ratio < 0.99
- Background: ratio = 0
- RNG: numpy Generator; deterministic via fixed seed (seed=42)
- Manifest portability: relative filenames (keep manifest.json next to cache directory)

Primary code references

- src/brain_brr/data/io.py: parse_tusz_csv, events_to_binary_mask
- src/brain_brr/data/cache_utils.py: scan_existing_cache
- src/brain_brr/data/datasets.py: BalancedSeizureDataset
- src/brain_brr/train/loop.py: training dataset selection and guards
