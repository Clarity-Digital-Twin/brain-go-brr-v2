# TUSZ EDF Header Repair — Canonical Reference

Last updated: 2025-09-21

Issue
- Some TUSZ EDF files contain malformed start date (colons instead of periods) in EDF header bytes 168–175, e.g., `01:01:85` vs `01.01.85`.

Implementation (current codebase)
- `_repair_edf_header_inplace`: `src/brain_brr/data/io.py`
- Integration in EDF load path with retry on header failure: `src/brain_brr/data/io.py`

Strategy
1) Try standard load
2) If header error: repair date separators and retry
3) Optional fallback to permissive reader if needed

Notes
- Non-destructive: operates on a copy/temp path if needed
- Ensures 100% coverage of known problematic TUSZ files

