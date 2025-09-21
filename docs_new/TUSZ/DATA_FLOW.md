TUSZ Data Flow (Canonical)

Purpose
- Show exactly where each TUSZ concern is handled in code: CSV_BI parsing, channel normalization, windowing, cache creation, manifest categorization, balanced sampling, and training integration.

End-to-end flow

1) EDF+CSV_BI → events
   - Parser: `src/brain_brr/data/io.py:parse_tusz_csv`
     - Format: `channel,start_time,stop_time,label,confidence`
     - Duration parsed from `# duration = ... secs`
     - Seizure label set: {seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz}
     - Output: list of (start_sec, end_sec, label)

2) Events → per-sample mask
   - Mask: `src/brain_brr/data/io.py:events_to_binary_mask`
     - fs=256 Hz; mask[start:end] = 1.0 for seizure events
     - Deterministic rounding (int(start*fs), int(end*fs))

3) EDF → canonical channels
   - Loader: `src/brain_brr/data/loader.py`
     - Canonical 19-ch order (see constants)
     - Synonyms (T7→T3, T8→T4, P7→T5, P8→T6)
     - Interpolation when rare channels missing; logs occurrences

4) Preprocess + windowing
   - Bandpass 0.5–120 Hz; 60 Hz notch
   - Resample to 256 Hz
   - Per-channel z-score
   - Window=60s, stride=10s → per-window tensors and label masks (shape: [n_windows, 19, 15360])

5) Cache creation
   - Cache writer: stored as NPZ with `windows` and `labels`
   - File pattern: `<basename>_windows.npz`
   - Code path: `src/brain_brr/data/dataset.py` (EEGWindowDataset caching path)

6) Manifest categorization
   - Scanner: `src/brain_brr/data/cache_utils.py:scan_existing_cache`
     - Categories per window:
       - no_seizure: ratio == 0
       - full_seizure: ratio ≥ 0.99
       - partial_seizure: 0 < ratio < 0.99
     - Output: `manifest.json` (relative filenames)
     - Guard: warn/fail on zero partial/full to prevent wasted runs

7) Balanced training dataset
   - Dataset: `src/brain_brr/data/datasets.py:BalancedSeizureDataset`
     - Composition: ALL partial + 0.3× full + 2.5× background
     - RNG: numpy Generator with seed; deterministic shuffle
     - Fails fast if no partial windows

8) Training integration
   - Selection: `src/brain_brr/train/loop.py`
     - Uses BalancedSeizureDataset when `manifest.json` exists and non-empty
     - Validation uses standard dataset (no balancing)
     - Exits if balanced dataset length is zero

Known failure points (and fixes)
- CSV_BI misparsed as simple CSV → all-zero masks → FIX: parse_tusz_csv handles CSV_BI
- Missing seizure types (e.g., `cpsz`) → false background → FIX: complete seizure label set
- Broken EDF header → read failure → FIX: minimal header repair then retry
- No manifest/guards → training proceeds with zero seizures → FIX: scan-cache + fail-fast

Operational commands
- Build cache: `python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>`
- Scan manifest: `python -m src scan-cache --cache-dir <cache_dir>`
- Train (balanced auto-detected): `python -m src train <config.yaml>`

See also
- CSV_BI_PARSER.md, CHANNELS_AND_MONTAGE.md, CACHE_AND_SAMPLING.md, PREFLIGHT_AND_TROUBLESHOOTING.md, EDF_HEADER_REPAIR.md

