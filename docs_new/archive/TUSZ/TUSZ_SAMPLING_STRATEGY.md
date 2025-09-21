# TUSZ Sampling Strategy — SeizureTransformer Style (Canonical)

Last updated: 2025-09-21

We pre-categorize cached windows into {no_seizure, full_seizure, partial_seizure} and build a fixed, reproducible training pool. This matches the SeizureTransformer paper’s approach, with explicit thresholds and windowing defined below.

Formula (from SeizureTransformer):
- D = Dps ∪ D*fs ∪ D*ns
- Dps = ALL partial‑seizure windows (0% < seizure_ratio < ~100%)
- D*fs = 0.3 × |Dps| randomly selected full‑seizure windows (≈100%)
- D*ns = 2.5 × |Dps| randomly selected no‑seizure windows (0%)

Windowing and overlap
- fs = 256 Hz (constants.SAMPLING_RATE)
- Window = 60 s → 15,360 samples (constants.WINDOW_SAMPLES)
- Stride = 10 s (83% overlap, slightly higher than paper’s 75%)

Category thresholds (exact)
- Compute per‑window seizure_ratio = mean(label > 0) across the window’s 15,360 samples
- no_seizure: ratio == 0.0
- full_seizure: ratio ≥ 0.99 (robust to boundary rounding)
- partial_seizure: 0.0 < ratio < 0.99

Implementation
- Manifest builder: `src/brain_brr/data/cache_utils.py` (scan_existing_cache)
  - Writes `manifest.json` with relative filenames; skips corrupted NPZ with warning
- Dataset: `src/brain_brr/data/datasets.py` (BalancedSeizureDataset)
  - ALL partial + 0.3× full + 2.5× background; numpy RNG with seed; deterministic shuffle
- Training: `src/brain_brr/train/loop.py`
  - Uses BalancedSeizureDataset when manifest exists; validation uses standard dataset

Pitfalls and checks
- If CSV_BI annotations were not included in cache, all windows become no_seizure; verify manifest counts > 0 for partial/full before training
- Old caches built with the pre‑fix parser must be deleted and rebuilt

