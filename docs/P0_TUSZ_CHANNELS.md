# P0 — TUSZ Channel Name Mismatch (Resolved by Canonicalization)

Status: In Progress → Fixed by robust channel canonicalization in `load_edf_file`
Priority: P0
Owner: Data Pipeline
Date: 2025-09-18

## Problem

TUSZ EDF files use Temple naming that does not match our canonical 10–20 list:
- Examples: `EEG FP1-REF`, `EEG CZ-REF`, directories `01_tcp_ar/`, `02_tcp_le/`
- Our pipeline expects exact names in `constants.CHANNEL_NAMES_10_20` (19‑ch):
  `Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4, P4, F8, T4, T6, O2`

Previously this raised:
`ValueError: Missing required channels: [...]`

## Root Cause

- TUSZ labels carry prefixes/suffixes and casing: `EEG <NAME>-REF`.
- Some datasets use `T7/T8` and `P7/P8` instead of `T3/T4` and `T5/T6`.
- We validated channel presence strictly against canonical names before normalizing.

## Adopted Fix (clean, minimal, robust)

Implemented in `src/experiment/data.py:load_edf_file`:
- Read EDF via MNE (header repair fallback retained).
- Optional `standard_1020` montage (best effort).
- Canonicalize every `raw.ch_names` entry before validation:
  - Strip prefix `EEG `, strip suffixes `-REF`, `-LE`, `-AR`, `-AVG`.
  - Uppercase → map to canonical with:
    - Direct map for the 19 canonical names
    - Synonyms: `T7→T3`, `T8→T4`, `P7→T5`, `P8→T6` (from `constants.CHANNEL_SYNONYMS`)
- Rename channels in-memory to their canonical form, then pick/reorder to the
  19‑channel order.

This preserves our public API and avoids brittle per‑file heuristics.

## Why this is correct for TUSZ

- TUSZ referential montages label sensors as `EEG <SENSOR>-REF`; stripping the
  prefix/suffix yields the intended sensor name.
- Synonym handling unifies `T7/T8,P7/P8` into our canonical `T3/T4,T5/T6` set.
- We do not “take first 19 channels”; we always select the exact canonical set
  in a fixed order, or error if truly missing.

## Files/Lines

- `src/experiment/data.py`: channel canonicalization and rename
  - around `load_edf_file(...)` (normalization + rename before validation)
- `src/experiment/constants.py`: canonical list + synonyms (`T7/T8,P7/P8`)

## Quick Validation

```
python - << 'PY'
from pathlib import Path
from src.experiment.data import load_edf_file

edf = Path('data/tusz/edf/dev/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf')
x, fs = load_edf_file(edf)
print(x.shape, fs)
PY
```
Expected: `(19, n_samples)`, `fs≈256` (after later preprocessing we fix to 256 Hz).

## Acceptance Criteria

- `load_edf_file()` returns `(19, n_samples)` on TUSZ files without manual maps.
- Channel order exactly matches `CHANNEL_NAMES_10_20`.
- No missing‑channel errors for typical `01_tcp_ar` / `02_tcp_le` files.
- Training smoke runs on TUSZ with `configs/tusz_train.yaml`.

## Notes

- We keep montage application best‑effort to help MNE label inference; it is not
  relied upon for renaming.
- If an EDF truly lacks required sensors, we still raise a clear `ValueError`.
- If both synonym and canonical variants co‑exist (rare), MNE will refuse
  duplicate names on rename; this signals a genuinely ambiguous header.

