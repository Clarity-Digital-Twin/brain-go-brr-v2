TUSZ Data Flow (Canonical)

Purpose
- Show exactly where each TUSZ concern is handled in code: CSV_BI parsing, channel normalization, windowing, cache creation, manifest categorization, balanced sampling, and training integration.

End-to-end flow

1) EDF+CSV_BI → events
   - Parser: `src/brain_brr/data/io.py:parse_tusz_csv`
     - Format: `channel,start_time,stop_time,label,confidence`
     - Duration parsed from `# duration = ... secs`
     - **Seizure label set (v2.0.3)**: {seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz}
       - **CRITICAL**: mysz (myoclonic) was missing until Sept 2025 - affected 44 annotations
       - **NOTE**: spkz does NOT exist in TUSZ v2.0.3 despite being in older docs
     - Output: list of (start_sec, end_sec, label)

2) Events → per-sample mask
   - Mask: `src/brain_brr/data/io.py:events_to_binary_mask`
     - fs=256 Hz; mask[start:end] = 1.0 for seizure events
     - Deterministic rounding (int(start*fs), int(end*fs))

3) EDF → canonical channels
   - Loader: `src/brain_brr/data/io.py:load_edf_file`
     - Canonical 19-ch order (see constants.CHANNEL_NAMES_10_20)
     - Synonyms (T7→T3, T8→T4, P7→T5, P8→T6) via constants.CHANNEL_SYNONYMS
     - Interpolation: only Fz/Pz when missing (via MNE montage)
     - Channel ordering: `src/brain_brr/utils/pick_utils.py:pick_and_order` (robust across MNE versions)

4) Preprocess + windowing
   - Bandpass 0.5–120 Hz; 60 Hz notch
   - Resample to 256 Hz
   - Per-channel z-score
   - Window=60s, stride=10s → per-window tensors and label masks (shape: [n_windows, 19, 15360])

5) Cache creation (OFFLINE - NOT done by datasets.py!)
   - **CRITICAL**: Cache files must be pre-generated offline before training
   - File format: `<basename>_data.npy` + `<basename>_labels.npy` (memory-mapped, v3.8.0+)
   - **Cache generation methods**:
     - **Modal (recommended)**: `modal run deploy/modal/app.py --action populate-cache`
     - **NPZ conversion**: `scripts/convert_cache_to_mmap.py --source <npz_dir> --dest <npy_dir>`
     - **Legacy CLI** (`python -m src build-cache`): EXISTS but deprecated, doesn't write NPY caches!
   - **Cache loading** (read-only):
     - Datasets: `src/brain_brr/data/datasets.py` (EEGWindowDataset, BalancedSeizureDataset, ValidationDataset)
     - Mmap loader: `src/brain_brr/data/cache_utils.py:load_cache_mmap` (zero-copy, OS-managed)
   - Legacy NPZ format (`*_windows.npz`) deprecated but still readable via scan_existing_cache

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
     - Train: BalancedSeizureDataset when `manifest.json` exists and non-empty
     - Validation: ValidationDataset (natural ~8% seizure distribution, no balancing)
     - Exits if balanced dataset length is zero (fail-fast on missing seizures)

Known failure points (and fixes)
- CSV_BI misparsed as simple CSV → all-zero masks → FIX: parse_tusz_csv handles CSV_BI
- Missing seizure types (e.g., `cpsz`) → false background → FIX: complete seizure label set
- **mysz missing (Sept 2025 discovery)** → 44 seizures marked as background → FIX: added mysz to set
- **spkz in old docs but not in data** → potential confusion → FIX: removed spkz from all docs
- **Invalid caches from before mysz fix** → wrong training data → FIX: Delete cache/*.npy + manifest.json, rebuild
- **Old NPZ caches (pre-v3.8.0)** → massive RAM usage → FIX: Convert to NPY mmap via scripts/convert_cache_to_mmap.py
- Broken EDF header → read failure → FIX: minimal header repair then retry
- No manifest/guards → training proceeds with zero seizures → FIX: scan-cache + fail-fast

Operational commands
See README.md for quick command reference.

See also
- tusz-csv-parser.md, tusz-channels.md, tusz-cache-sampling.md, tusz-preflight.md, tusz-edf-repair.md

