# Data Overview

Corpus

- TUH EEG Seizure Corpus (TUSZ), 10–20 montage (19 channels), adults.
- Strict channel order is required throughout the pipeline.
- TUH‑specific guides: `docs/tusz/` (parsers, channels, EDF repair, preflight).

Shapes and units

- Window: `(19, 15360)` (60s at 256Hz) in microvolts (µV), dtype float32.
- Batch: `(B, 19, 15360)`.

Pipeline summary

- Load EDF with MNE; repair rare headers (colon→period) when necessary.
- Map channels to canonical order, apply synonyms (T7→T3, T8→T4, P7→T5, P8→T6).
- Resample to 256Hz, bandpass 0.5–120Hz, 60Hz notch; per‑channel z‑score.
- Extract 60s windows with 10s stride; save to NPZ (windows and optional labels).
- Build manifest categorizing windows as partial/full/no‑seizure; Balanced dataset uses manifest directly.

Channel order (must maintain)

- ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]

Dataset classes

- `EEGWindowDataset`: lazy NPZ cache builder/reader; always returns `(window, label)`; can operate on‑demand and also builds `_dataset_index.json` for fast restarts.
- `BalancedSeizureDataset`: reads manifest and builds a balanced sample: all partial, 0.3× full, 2.5× background; exposes exact `seizure_ratio` (no sampling).

Training integration

- Training auto‑creates/validates manifest and switches to `BalancedSeizureDataset` when available.
- Fallback sampler exists for `EEGWindowDataset` but is bypassed when using balanced manifest.

See also

- Preprocessing: `docs/02-data/preprocessing.md`
- Cache and manifest: `docs/02-data/cache-layout.md`
