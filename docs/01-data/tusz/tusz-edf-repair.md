EDF Header Repair (TUSZ)

Problem

- A small number of TUSZ EDFs have malformed header fields (e.g., time components using `:` instead of `.`, invalid date strings), which causes MNE to raise parsing errors and prevents cache building.

Symptoms

- `mne.io.read_raw_edf` raises `ValueError`/`OSError` related to start date/time parsing.
- Cache build skips whole recordings due to unreadable EDF.

Strategy (implemented behavior)

- Attempt MNE load normally.
- On known header parse errors, apply a minimal repair of the EDF header fields to normalize date/time formatting and retry.
- If repair still fails, skip the file with a warning so the pipeline can proceed (and surface the count at the end).

Repairs (typical cases)

- Normalize date/time tokens to the EDF specification (e.g., replace `:` with `.` where required).
- Ensure fields are the correct width and zero-padded as needed.
- Leave all signal samples and annotations untouched.

Operational guidance

- Repairs are only applied when the initial read fails with a recognized parse error.
- The fix is intentionally minimal and conservative; if an EDF remains unreadable after repair, the file is excluded and reported.

How to verify

1) Run a small cache build over a subset that includes the problematic file(s).
2) Confirm the loader logs either a successful repair+read or a clear skip message.
3) After build, run `python -m src scan-cache --cache-dir <cache_dir>`; the manifest counts should not drop to zero due to one bad EDF.

Related docs

- CSV_BI parsing: CSV_BI_PARSER.md
- Preflight & troubleshooting: PREFLIGHT_AND_TROUBLESHOOTING.md

Code anchors

- Loader entry: channel normalization and EDF open live in the data loader.
- Repair helper: header normalization is applied on read failure before retry.

