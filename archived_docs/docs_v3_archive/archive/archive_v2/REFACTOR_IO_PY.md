# Refactor Plan — src/brain_brr/data/io.py

**Status**: ✅ VERIFIED - Based on actual code (2025-10-02)
**Priority**: LOW (actually quite clean already)
**Owner**: Senior ML auditor (Codex)
**Scope**: EDF loading and channel handling (`load_edf_file` ≈153 lines)

---

## 1. Context & Problem Statement

**IMPORTANT**: Previous version of this plan was CRITICALLY FLAWED - it hallucinated operations that don't exist.

**ACTUAL DATA PIPELINE** (verified against code):
```
1. load_edf_file(edf_path) → (data_µV, fs)        [io.py]
   ↓
2. preprocess_recording(data_µV, fs) → data_proc   [preprocess.py - SEPARATE MODULE]
   ↓
3. parse_tusz_csv(csv_path) → events               [io.py - SEPARATE FUNCTION]
   ↓
4. events_to_binary_mask(events, ...) → labels     [io.py - SEPARATE FUNCTION]
   ↓
5. extract_windows(data_proc, labels) → windows    [windows.py - SEPARATE MODULE]
```

**What `load_edf_file()` ACTUALLY does** (lines 54-206):
1. ✅ Read EDF with MNE (with header repair fallback for TUSZ files)
2. ✅ Normalize channel names (TUSZ-specific cleaning: remove "EEG" prefix, "-LE" suffix)
3. ✅ Apply standard casing (case-insensitive match to target channels)
4. ✅ Apply channel synonyms (e.g., T7→T3 for 10-20 compatibility)
5. ✅ Filter to target channels
6. ✅ Interpolate missing midline channels (Fz, Pz) using montage
7. ✅ Apply montage (best-effort for spatial positioning)
8. ✅ Return (data_microvolts, sampling_rate) - RAW data, NO processing

**What it does NOT do** (contrary to previous hallucinated plan):
- ❌ Does NOT resample (handled by `preprocess.py:preprocess_recording()`)
- ❌ Does NOT filter (handled by `preprocess.py:preprocess_recording()`)
- ❌ Does NOT parse labels (handled by separate `parse_tusz_csv()` function)
- ❌ Does NOT create binary masks (handled by separate `events_to_binary_mask()`)

**Current Pain Points** (real ones):
1. **Channel handling complexity** (lines 104-206):
   - TUSZ-specific name cleaning
   - Case-insensitive matching
   - Synonym application
   - Interpolation logic
   - All mixed in one 100-line block
2. **Header repair logic** (lines 86-102):
   - Temp file creation for TUSZ date format fix
   - Could be extracted for clarity
3. **Montage application** (lines 149-174):
   - Interpolation requires montage
   - Montage application is best-effort
   - Logic is interleaved with channel selection

---

## 2. Goals

1. **Clarify responsibilities** - `load_edf_file` should focus on EDF I/O, not channel gymnastics
2. **Extract channel handling** - Clean separation of concerns
3. **Improve testability** - Each step should be unit-testable in isolation
4. **Maintain compatibility** - NO changes to output format or behavior
5. **Document pipeline** - Make it crystal clear what happens where

---

## 3. Proposed Architecture

**Option A: Minimal Refactor** (Recommended - io.py is actually quite clean)
```python
# Keep load_edf_file as orchestrator, extract complex sub-steps:

def load_edf_file(...) -> tuple[npt.NDArray[np.float32], float]:
    # Step 1: Read EDF (potentially with repair)
    raw = _read_edf_with_repair(file_path)

    # Step 2: Channel normalization and selection
    raw = _normalize_and_select_channels(raw, target_channels, channel_synonyms)

    # Step 3: Handle missing channels via interpolation
    raw = _handle_missing_channels(raw, target_channels, apply_montage)

    # Step 4: Extract data
    data_microvolts = raw.get_data() * 1e6
    fs = raw.info["sfreq"]

    return data_microvolts.astype(np.float32), fs

# New helpers (all private, in same file):
def _read_edf_with_repair(file_path: Path) -> mne.io.Raw: ...
def _normalize_channel_names(raw: mne.io.Raw) -> mne.io.Raw: ...
def _normalize_and_select_channels(raw, target, synonyms) -> mne.io.Raw: ...
def _handle_missing_channels(raw, target, apply_montage) -> mne.io.Raw: ...
```

**Option B: Aggressive Refactor** (If we want maximum modularity)
```python
# Move to data/edf/ subpackage:
data/edf/
├── __init__.py
├── reader.py           # _read_edf_with_repair
├── channels.py         # Channel normalization, synonym handling
├── interpolation.py    # Missing channel interpolation
└── montage.py          # Montage application
```

**Recommendation**: **Option A** - Keep everything in io.py with extracted helpers. The file is only 313 lines total and already well-organized. Over-splitting would hurt readability.

---

## 4. Step-by-Step Plan

### Phase 0 – Baseline Snapshot
- Run `make test` to capture green state
- Document current test coverage for io.py (currently 94%)
- Save sample outputs from `load_edf_file` for regression testing

### Phase 1 – Extract EDF Reading Logic
1. Create `_read_edf_with_repair(file_path: Path) -> mne.io.Raw`
   - Encapsulates lines 86-102 (MNE read + temp file repair logic)
   - Returns raw MNE object
2. Replace inline logic in `load_edf_file` with helper call
3. Add unit test with mocked MNE to verify repair logic
4. Run regression test to verify outputs unchanged

### Phase 2 – Extract Channel Normalization
1. Create `_normalize_channel_names(raw: mne.io.Raw) -> mne.io.Raw`
   - TUSZ cleaning (lines 106-123): remove "EEG", "-LE", etc.
   - Case normalization (lines 127-136)
   - Synonym application (lines 139-142)
2. Add unit test with synthetic MNE object
3. Verify behavior with TUSZ sample files

### Phase 3 – Extract Interpolation Logic
1. Create `_interpolate_missing_channels(raw, missing, montage) -> mne.io.Raw`
   - Handles midline interpolation (lines 149-174)
   - Encapsulates montage-based interpolation
2. Add unit test with synthetic signals
3. Verify Fz/Pz interpolation works as expected

### Phase 4 – Montage Helper
1. Create `_apply_montage_best_effort(raw: mne.io.Raw) -> None`
   - Extracts lines 209-218 (currently in separate function, just document)
   - Best-effort montage application (already separate at line 209)
2. No changes needed - already a clean helper!

### Phase 5 – Documentation & Cleanup
- Add comprehensive docstrings to each helper
- Document TUSZ-specific quirks (date format, channel names)
- Update pipeline diagram in docs to show correct flow
- Add comments explaining why certain operations are needed

### Phase 6 – Regression Validation
- Re-run full test suite (`make test`)
- Compare outputs with baseline using `np.allclose`
- Verify EDF files with various channel configurations still work
- Run smoke test to ensure end-to-end pipeline works

---

## 5. Testing Strategy

### New Unit Tests (`tests/unit/data/test_io_helpers.py`)

```python
def test_normalize_tusz_channel_names():
    """Test TUSZ-specific channel name cleaning."""
    # Test: "EEG FP1-LE" → "Fp1"
    # Test: "EEG T7-REF" → "T7"
    # Test: Already clean names unchanged

def test_channel_synonym_application():
    """Test synonym mapping (T7→T3, etc.)."""
    # Verify old 10-20 names map to standard

def test_case_insensitive_matching():
    """Test case-insensitive channel matching."""
    # "fp1", "FP1", "Fp1" all match "Fp1"

def test_midline_interpolation():
    """Test Fz/Pz interpolation from neighbors."""
    # Synthetic signal with missing Fz
    # Verify interpolation produces reasonable values

def test_header_repair():
    """Test TUSZ date format repair."""
    # Mock EDF with colon date separator
    # Verify repair succeeds
```

### Existing Tests to Verify
- `test_load_edf_file()` - Verify outputs unchanged
- `test_tusz_channel_variations()` - Verify TUSZ compatibility
- Integration tests with real TUSZ files

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Channel order changes after refactor | Use robust `pick_and_order` utility (already in place) |
| Interpolation behavior differs | Copy exact logic; add regression test with known outputs |
| TUSZ quirks break | Keep TUSZ-specific handling together; comprehensive test coverage |
| MNE API changes | Pin MNE version; use defensive programming |

---

## 7. Rollback Plan

- Keep each phase in separate commit
- If regression detected, revert to baseline
- Retain new unit tests for future attempts
- Restore baseline outputs to verify rollback success

---

## 8. Open Questions

1. **Should we extract to separate module?**
   - Recommendation: NO - keep in io.py for simplicity
   - File is only 313 lines, well within acceptable range

2. **Should interpolation be configurable?**
   - Currently hardcoded to Fz/Pz midline channels
   - Could make extensible for other missing channels
   - Defer to v4 unless needed

3. **Should we support other EDF variants?**
   - Currently optimized for TUSZ
   - Could add support for EDF+ or other formats
   - Out of scope for this refactor

---

## 9. Why This Refactor is LOW Priority

**Current Reality:**
- `load_edf_file` is already well-structured (153 lines is reasonable)
- Clear separation from preprocessing (that's in preprocess.py)
- Clear separation from label handling (separate functions)
- 94% test coverage (excellent)
- Working reliably in production

**What We'd Gain:**
- Slightly better testability (helpers could be tested in isolation)
- Marginally clearer code organization
- ~20-30 line reduction via extraction

**What We'd Risk:**
- Breaking working code during extraction
- Over-engineering a function that's already clean
- Time investment with minimal payoff

**Recommendation**:
- **Defer this refactor** until there's a concrete need (e.g., supporting new EDF formats)
- Focus on detector.py and metrics.py refactors first (those have real complexity issues)
- If implemented, use **Option A (minimal)** - keep helpers in same file

---

## 10. Correct Documentation of Actual Pipeline

**For reference, the REAL data flow is:**

```python
# 1. EDF Loading (io.py:load_edf_file)
data_uv, fs = load_edf_file(edf_path)
# Returns: RAW microvolts, NO filtering/resampling

# 2. Signal Processing (preprocess.py:preprocess_recording)
data_proc = preprocess_recording(data_uv, fs_original=fs)
# Does: resample, bandpass, notch, z-score, clip outliers

# 3. Label Loading (io.py:parse_tusz_csv + events_to_binary_mask)
duration_s, events = parse_tusz_csv(csv_path)
labels = events_to_binary_mask(events, duration_s, fs=256)

# 4. Windowing (windows.py:extract_windows)
windows, window_labels, starts = extract_windows(
    data_proc,
    window_size=15360,  # 60s @ 256Hz
    stride=2560,        # 10s @ 256Hz
    labels=labels
)

# 5. Caching (datasets.py)
np.savez_compressed(cache_path, windows=windows, labels=window_labels)
```

This pipeline is **ALREADY CLEAN** - each step is in the right module:
- ✅ io.py: EDF I/O and TUSZ CSV parsing
- ✅ preprocess.py: Signal processing
- ✅ windows.py: Windowing logic
- ✅ datasets.py: Dataset orchestration

**Conclusion**: Minimal refactoring needed, if any.

---

**Audit Status**: ✅ VERIFIED against actual code (2025-10-02)
**Priority**: LOW (currently well-organized)
**Recommendation**: Defer unless specific need arises
