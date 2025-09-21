# PHASE1_DATA_PIPELINE.md - EEG Data Loading & Preprocessing

Note (2025-09-21): Canonical implementation references
- CSV_BI parsing: `docs/references/TUSZ_CSV_BI_PARSER.md`
- Channels (19‑ch + synonyms): `docs/references/TUSZ_CHANNELS.md`
- EDF header repair: `docs/references/TUSZ_EDF_HEADER_FIX.md`
- Balanced dataset (manifest): `docs/references/TUSZ_SAMPLING_STRATEGY.md`
- Current modules under `src/brain_brr/*` (older `src/experiment/*` paths are historical)

## 🎯 Phase 1 Goal
Build a bulletproof data pipeline that loads TUH/CHB-MIT/custom EEG and outputs standardized windows ready for model training.

## 📋 Phase 1 Checklist
- [ ] EDF file reader with MNE (robust I/O)
- [ ] Channel validation, ordering, and montage alignment
- [ ] Signal preprocessing (resample, bandpass, notch, normalize)
- [ ] Window extraction with metadata
- [ ] PyTorch Dataset for windows
- [ ] Validation script on sample files
- [ ] TDD: unit + integration tests in `tests/`
- [ ] Quality gates: `make q` green (ruff + mypy)

## 🔧 Implementation Files
```
src/experiment/data.py         # Core data functions (I/O + preprocessing + windows + Dataset)
src/experiment/constants.py    # Channel names, synonyms, sampling rate, window params
tests/test_data.py             # Unit + integration tests (pytest markers)
scripts/validate_data.py       # Manual validation script (optional)
```

## 📊 Data Specifications

### Input Requirements
| Specification | Value | Notes |
|--------------|-------|-------|
| File Format | EDF/EDF+ | European Data Format |
| Channels | 19 (10-20) | Fixed canonical order required |
| Original Fs | Variable | Common: 250–500 Hz |
| Duration | Any | 30s to 24+ hours |

### Output Requirements
| Specification | Value | Notes |
|--------------|-------|-------|
| Target Fs | 256 Hz | Fixed for all data |
| Window Size | 60 seconds | 15,360 samples |
| Window Stride | 10 seconds | 2,560 samples (50s overlap) |
| Data Type | float32 | Memory efficient |
| Units | Microvolts (µV) | Always convert from Volts |
| Shape | (B, 19, 15360) | B = batch size |

## 🏗️ Core Components (Design + Signatures)

### 1) Channel & Sampling Constants
```
# src/experiment/constants.py

CHANNEL_NAMES_10_20 = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",  # Left
    "Fz", "Cz", "Pz",                                   # Midline
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"   # Right
]

# Optional mapping for dataset synonyms (e.g., T3→T7, T4→T8)
CHANNEL_SYNONYMS = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
}

SAMPLING_RATE = 256
WINDOW_SIZE_SEC = 60
STRIDE_SIZE_SEC = 10
WINDOW_SAMPLES = WINDOW_SIZE_SEC * SAMPLING_RATE   # 15360
STRIDE_SAMPLES = STRIDE_SIZE_SEC * SAMPLING_RATE   # 2560
```

### 2) EDF Reader with Header Repair Fallback
```
# src/experiment/data.py

from pathlib import Path
from typing import Tuple
import numpy as np
import mne

def load_edf_file(
    file_path: Path,
    target_channels: list[str],  # constants.CHANNEL_NAMES_10_20
    apply_montage: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Load EDF and return data (n_channels, n_samples) in canonical order and original fs.

    - Uses MNE for robust EDF reading (already permissive with malformed headers).
    - If MNE fails on header issues (e.g., TUSZ date separators), repairs on temp copy.
    - Enforces 19-channel 10–20 order; errors if required channels are missing.
    - Always converts units to microvolts (µV) after `raw.get_data()`.
    - Optionally sets montage to 'standard_1020' for alignment.
    """
```

Key rules
- Use `mne.io.read_raw_edf(file_path, preload=True, verbose="WARNING")`.
- **NEW**: If MNE fails with header/startdate error, create temp copy, fix date separators (colons→periods at byte 168-175), retry.
- If `apply_montage`, call `raw.set_montage("standard_1020", on_missing="ignore")`.
- Handle synonyms: map channel names via `CHANNEL_SYNONYMS` before validation.
- Validate presence and order; `raw.pick_channels(target_channels, ordered=True)`.
- `raw.get_data()` returns Volts; standardize by multiplying by `1e6` → µV; cast to `np.float32`.

**Header Repair**: Based on TUSZ v2.0.3 experience, 1/865 files has malformed date separators (colons instead of periods). MNE is typically permissive, but we add fallback for edge cases.

### 3) Preprocessing Pipeline
```
# src/experiment/data.py

from scipy.signal import resample, butter, lfilter, iirnotch

def preprocess_recording(
    data: np.ndarray,
    fs_original: float,
    target_fs: int = 256,
    bandpass: tuple[float, float] = (0.5, 120.0),
    notch_freq: int = 60,  # Set 50 for EU
) -> np.ndarray:
    """
    Steps:
      1) Resample to target_fs using scipy.signal.resample (Phase 1 baseline)
      2) Bandpass (Butterworth order=3) with lfilter for parity with our strategy doc
      3) Notch at powerline frequency using iirnotch
      4) Per-channel z-score normalization

    Returns float32 array of shape (n_channels, n_samples_new), NaN/Inf replaced with 0.
    """
```

Notes
- We use `lfilter` (not `filtfilt`) to stay consistent with PREPROCESSING_STRATEGY.md for reproducibility.
- Keep processing after resample to the new fs.
- Compute z-score per channel with epsilon for stability; then `np.nan_to_num`.

### 4) Window Extraction
```
# src/experiment/data.py

def extract_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Returns:
      windows: (n_windows, n_channels, window_size)
      window_labels: (n_windows, window_size) or None
      metadata: {"start_samples": List[int]}

    Only full windows are returned (no padding). End-truncation is acceptable in Phase 1.
    """
```

### 5) PyTorch Dataset
```
# src/experiment/data.py

class EEGWindowDataset(torch.utils.data.Dataset):
    """Materializes windows for training. Phase 1: simple in-memory baseline.
    Phase 2+: consider memmap backed storage and per-file indexing to reduce RAM.
    """
```

Rules
- Use `Path` for file paths. Ensure `cache_dir` exists before writing `.npz`.
- Cache payload includes `windows` and optional `labels`. Keep dtype `float32`.
- Carry `file_ids` and `start_samples` to enable timeline reconstruction later.
- If labels are intervals (e.g., CSV_BI), convert to a binary per-sample mask at 256 Hz.

## ✅ TDD Plan (Tests First)

Place tests in `tests/test_data.py` using pytest markers.

Unit tests (fast)
- `test_load_edf_missing_channels_raises()` — simulate missing channel → ValueError
- `test_load_edf_orders_channels_correctly()` — mock MNE Raw ch_names → exact order
- `test_preprocess_shapes_and_dtype()` — shape changes with resample, dtype float32, finite values
- `test_extract_windows_counts_and_metadata()` — correct n_windows and start_samples
- `test_extract_windows_with_labels_alignment()` — y-window slices match input mask

Integration tests (medium)
- `test_end_to_end_synthetic_roundtrip()` — synthetic sine + impulse → load → preprocess → windows → expected shapes
- `test_dataset_len_and_item_shapes()` — dataset returns tensors of shape (19, 15360) [+ label]

Markers and commands
- Use `@pytest.mark.unit` and `@pytest.mark.integration`
- Run fast: `make t` (alias: `make test-fast`)
- Full: `make test`
- Quality: `make q` (ruff + mypy)

Fixtures and utilities
- Synthetic generator producing (19, n_samples) at 256 Hz
- Label mask generator for short seizure bursts

## 🧪 Validation Script (Optional Manual Check)
```
# scripts/validate_data.py

# Load → preprocess → window → assert shapes, dtypes, and finiteness
```

## 🧹 Clean Code Rules (Phase 1)
- Full type hints on public functions and methods
- Docstrings describing inputs/outputs, shapes, units
- No mutable default arguments; use `None` and create inside
- Prefer small, pure functions; deterministic behavior
- Import order: stdlib → third-party → first-party (experiment)
- Consistent errors: `ValueError` for validation; surface clear messages

## 📈 Quality & Performance

Data quality checks
- No NaN/Inf values after preprocessing
- Channels in canonical 10–20 order
- Fs exactly 256 Hz post-resample
- Windows overlap = 50 seconds (60s window, 10s stride)
- Z-score per channel (mean≈0, std≈1) on non-flat segments

Baseline performance targets (CPU, single file)
- 1-hour EDF: load ≤ 30–60s (target) — measure and record
- Preprocess ≤ 30s; window extraction ≤ 5s
- Support parallel loading later via DataLoader `num_workers`

## 🚨 Common Issues & Solutions
| Issue | Solution |
|---|---|
| Missing channels | Use `CHANNEL_SYNONYMS`; error if still missing |
| Non-standard montage | `raw.set_montage('standard_1020')` with `on_missing='ignore'` |
| Malformed EDF header | Auto-repair date separators on temp copy (TUSZ fix) |
| Corrupted EDF | Catch MNE errors; skip file with log; continue pipeline |
| Memory pressure | Switch to memmap cache; per-file index in Dataset |
| Slow resample | Consider `resample_poly` in Phase 2 if needed |

## ✅ Definition of Done (Phase 1)
1) Code complete with type hints and docstrings
2) Tests passing: `pytest -m unit -v` and `pytest -m integration -v`
3) Quality gates: `make q` passes (ruff + mypy)
4) Validation: 10 sample files (TUH/CHB) processed without errors
5) Docs: this spec implemented faithfully; README remains consistent

## 🔗 Dependencies
- Base deps cover Phase 1: `mne`, `scipy`, `numpy`, `torch`
- No extras required (gpu/post/eval optional later)

---
Status: Ready for implementation (TDD-first) ✅
Estimated Time: 2–3 days
Owners: Data pipeline duo (eng + reviewer) 🧪
