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

- `EEGWindowDataset`: lazy NPZ cache builder/reader that can populate `_dataset_index.json` on demand. Returns dictionaries with keys `window`, `label`, `file_id`, and `window_start_s` so downstream code can stitch per-record timelines even when falling back from the balanced manifest.
- `BalancedSeizureDataset`: reads `train/manifest.json` and assembles a balanced slice (all partial, 0.3× full, 2.5× background). Returns the same dictionary structure and exposes `seizure_ratio` without sampling.
- `ValidationDataset`: loads `dev/manifest.json`, groups windows by cache file, and preserves natural ordering. Validation now depends on the manifest metadata instead of re-scanning every NPZ.

Dataset Strategy (CRITICAL - This is correct ML practice!)

- **Training**: Uses `BalancedSeizureDataset` to oversample seizures (8% → ~30% in batches)
  - Why: Model needs to see enough seizures to learn patterns effectively
  - Requires: `train/manifest.json` (auto-created if missing)
- **Validation**: Uses `ValidationDataset` with the natural distribution (~8% seizures)
  - Why: Accurate timeline stitching and FA math require manifest metadata
  - Requires: `dev/manifest.json` (regenerated alongside the balanced manifest)
  - Fallback: `EEGWindowDataset` only when a manifest is absent; metrics become slower and less precise

Training integration

- Training builds or validates the train manifest, then switches to `BalancedSeizureDataset`.
- Validation prefers `ValidationDataset`; if the manifest is missing it falls back to `EEGWindowDataset` and logs a warning because metrics degrade.
- Fallback sampler still exists for `EEGWindowDataset` but is rarely needed now that manifests are standard.

See also

- Preprocessing: `docs/02-data/preprocessing.md`
- Cache and manifest: `docs/02-data/cache-layout.md`
 - Official TUSZ splits and policy: `docs/tusz/tusz-splits.md`
