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
  - Scans `*.npz` under the cache dir and writes `manifest.json`
  - Uses relative filenames for portability: `{ "cache_file": "<stem>_windows.npz", "window_idx": <int> }`
  - Treats NPZ without `labels` as all background (categorized into no_seizure)
  - Skips corrupted NPZ with a warning
- Dataset: `src/brain_brr/data/datasets.py` (BalancedSeizureDataset)
  - Composition: ALL partial + 0.3× full + 2.5× background
  - Selection without replacement; numpy RNG with seed (default 42); shuffled deterministically
  - Constructor: `BalancedSeizureDataset(cache_dir, full_ratio=0.3, background_ratio=2.5, seed=42)`
  - Fails fast if the manifest has zero partial windows
- Training integration: `src/brain_brr/train/loop.py`
  - If `config.data.use_balanced_sampling` is true and `manifest.json` exists (or is auto‑built), training uses BalancedSeizureDataset
  - Validation always uses the standard dataset (no class balancing)
  - Guardrails: exits if the balanced path yields 0 windows

CLI
- Build cache (per split) and emit manifest with counts:
  - `python -m src build-cache --data-dir <tusz/edf/root> --cache-dir <cache/train> --split train`
- Scan existing cache and (re)build manifest:
  - `python -m src scan-cache --cache-dir <cache/train>`

Reproducibility and expected composition
- Deterministic selection and shuffle via numpy RNG (seed=42 by default)
- Dataset‑level seizure window fraction ≈ (1.0 + 0.3) / (1.0 + 0.3 + 2.5) ≈ 34.2%
- Batch‑level prevalence will track dataset composition (no per‑batch balancing)

Differences vs paper (and rationale)
- Overlap: we use 83% (60s/10s) vs paper’s 75%; higher overlap increases training examples at modest cost and aligns with existing pipeline
- Full‑window threshold: ≥0.99 instead of exactly 1.0 to absorb labeling edge effects from window boundaries

Pitfalls and checks
- If CSV_BI annotations were not included in cache building, all windows will be categorized as no_seizure; use `scan-cache` to verify that partial and/or full counts are > 0 before training
- NPZ without labels is treated as background; ensure TUSZ CSV_BI files are present/paired
- Old caches built with the pre‑fix parser (wrong CSV columns) must be deleted and rebuilt

References
- Constants (fs, window, stride): `src/brain_brr/constants.py`
- Manifest builder: `src/brain_brr/data/cache_utils.py`
- Balanced dataset: `src/brain_brr/data/datasets.py`
- Training integration: `src/brain_brr/train/loop.py`
