Channels, Montage, Synonyms, Interpolation

Canonical 10–20 order (must maintain)

- ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
  "Fz", "Cz", "Pz",
  "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]

Synonym handling

- T7→T3, T8→T4, P7→T5, P8→T6 (and similar historical aliases) normalized to canonical names

Missing channels

- Some EDFs have missing electrodes or alternative montages
- Our loader normalizes channel set and interpolates missing channels only when necessary

Interpolation policy

- Per-channel z-score normalization first (consistent scaling)
- Only midline channels Fz and Pz are interpolated (when missing) using MNE's
  montage-based interpolate_bads after inserting zero channels and applying a
  standard_1020 montage (best-effort). Other missing channels raise an error.
- Strict logging when interpolation occurs

Montage notes

- TUSZ annotations are often bipolar (CSV_BI), but we target a canonical set of 19 referential channels post-processing
- Ensure mapping from raw channels to canonical names occurs before windowing

Code anchors

- src/brain_brr/constants.py (channel order, synonyms)
- src/brain_brr/data/loader.py (EDF → canonical channel tensor)

