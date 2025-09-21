# TUSZ Channels — Canonical 10–20 Mapping

Last updated: 2025-09-21

Canonical order (19 ch):
`["Fp1","F3","C3","P3","F7","T3","T5","O1","Fz","Cz","Pz","Fp2","F4","C4","P4","F8","T4","T6","O2"]`

Synonyms
- `T7→T3`, `T8→T4`, `P7→T5`, `P8→T6`

Implementation
- Canonical names and synonyms: `src/brain_brr/constants.py`
- Cleaning/renaming: `src/brain_brr/data/io.py` (canonicalize raw.ch_names before picking)
- Picking/reordering: `src/brain_brr/utils/pick_utils.py`

Rules
- Strip EEG-specific prefixes/suffixes (e.g., `EEG CZ-REF`, `-LE`, `-AR`, `-AVG`)
- Map synonyms, then strictly pick the exact 19‑ch set in fixed order
- Error only if truly missing required channels

