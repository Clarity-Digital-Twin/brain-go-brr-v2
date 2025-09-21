# [ARCHIVED] INTERPOLATION IMPLEMENTATION PLAN

Status: Historical investigation. Channel handling/canonicalization is documented in:
- docs/references/TUSZ_CHANNELS.md
- docs/implementation/PREPROCESSING_STRATEGY.md

**Goal**: Add MNE interpolation to handle 249 files missing Fz/Pz WITHOUT breaking ANYTHING
**Constraint**: MINIMAL changes, MAXIMUM compatibility, NO insane universe creation

## 📍 CURRENT STATE ANALYSIS

### What We Have Working
1. **load_edf_file()** in `src/experiment/data.py`:
   - Loads EDF with MNE ✅
   - Canonicalizes channel names ✅
   - Handles -REF, -LE suffixes ✅
   - Validates 19 channels ✅
   - **CRASHES on missing channels** ❌

2. **Test Suite**:
   - `tests/test_data.py` - Tests data loading
   - `tests/test_smoke.py` - Quick integration tests
   - All currently PASSING ✅

3. **Pipeline**:
   - `src/experiment/pipeline.py` calls load_edf_file
   - Creates EEGWindowDataset which processes files
   - Training crashes on file #47 (missing Fz/Pz)

## 🔧 THE SURGICAL CHANGES NEEDED

### Change #1: Update `load_edf_file()` ONLY
**File**: `src/experiment/data.py`
**Lines**: ~150-170 (the validation section)

**CURRENT CODE** (crashes):
```python
# Line 162-165
available = [ch for ch in target_channels if ch in raw.ch_names]
if len(available) != len(target_channels):
    missing = set(target_channels) - set(available)
    raise ValueError(f"Missing required channels: {sorted(missing)}")
```

**NEW CODE** (interpolates):
```python
# Line 162-165 REPLACEMENT
available = [ch for ch in target_channels if ch in raw.ch_names]
if len(available) != len(target_channels):
    missing = set(target_channels) - set(available)

    # NEW: Interpolate missing channels instead of crashing
    if missing:
        # Only allow Fz/Pz interpolation (known issue)
        if missing.issubset({'Fz', 'Pz'}):
            # Mark as bad channels
            raw.info['bads'].extend(list(missing))

            # Set montage for spatial info (required for interpolation)
            try:
                raw.set_montage('standard_1020', on_missing='ignore', match_case=False)
                # Interpolate the missing channels
                raw.interpolate_bads(reset_bads=True, mode='accurate')

                # Log but don't crash
                import logging
                logging.warning(f"Interpolated {missing} for {edf_path.name}")
            except Exception as e:
                # If interpolation fails, then raise original error
                raise ValueError(f"Could not interpolate missing channels {sorted(missing)}: {e}")
        else:
            # Unknown missing channels - still crash
            raise ValueError(f"Missing required channels: {sorted(missing)}")
```

### Change #2: Add Logging Configuration
**File**: `src/experiment/data.py`
**Location**: Top of file after imports

**ADD**:
```python
import logging

# Configure logging for interpolation tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
```

### Change #3: Track Interpolated Files
**File**: `src/experiment/data.py`
**Location**: In `load_edf_file()` function

**ADD** (optional but useful):
```python
# At function start, add parameter
def load_edf_file(
    edf_path: Path,
    track_interpolated: bool = True  # NEW
) -> tuple[np.ndarray, float]:

    # ... existing code ...

    # After interpolation
    if track_interpolated and missing:
        # Write to tracking file
        with open('interpolated_files.log', 'a') as f:
            f.write(f"{edf_path}\n")
```

## 🧪 TESTING STRATEGY

### Test #1: Verify Interpolation Works
**Create**: `tests/test_interpolation.py`

```python
"""Test interpolation of missing Fz/Pz channels."""
import pytest
from pathlib import Path
from src.experiment.data import load_edf_file

def test_interpolation_on_problem_file():
    """Test that we can load a file missing Fz/Pz."""
    # Use actual problem file we identified
    problem_file = Path('data/tusz/edf/dev/aaaaahie/s030_2016/01_tcp_ar/aaaaahie_s030_t001.edf')

    if problem_file.exists():
        # Should NOT crash anymore
        data, fs = load_edf_file(problem_file)

        # Should have 19 channels after interpolation
        assert data.shape[0] == 19

        # Check Fz (index 16) and Pz (index 18) are not all zeros
        assert not np.allclose(data[16], 0)  # Fz interpolated
        assert not np.allclose(data[18], 0)  # Pz interpolated

def test_normal_file_unchanged():
    """Test that files with all channels are unchanged."""
    # Use a file we know has all channels
    good_file = Path('data/tusz/edf/dev/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf')

    if good_file.exists():
        data, fs = load_edf_file(good_file)
        assert data.shape[0] == 19
        # No interpolation warning should be logged
```

### Test #2: Integration Test
**Run existing tests** - they should still pass!

```bash
make test  # Should still work
```

### Test #3: Pipeline Test
**Manual test** of training:

```bash
# Should no longer crash on file #47
make train-local
```

## 📊 VALIDATION PLAN

### Step 1: Identify Test Files
```python
# Get list of problem files from our CSV
import pandas as pd
df = pd.read_csv('tusz_channel_analysis.csv')
problem_files = df[df['missing_required'].str.contains('Fz', na=False)]['file'].tolist()

# Test first 5
for f in problem_files[:5]:
    load_edf_file(Path(f))  # Should work now!
```

### Step 2: Compare Interpolated vs Original
For files that HAVE Fz/Pz, we can:
1. Artificially remove them
2. Interpolate
3. Compare to original
4. Measure error

### Step 3: Monitor Training Metrics
Track separately:
- Loss on interpolated files
- Loss on normal files
- See if there's a significant difference

## 🚫 WHAT WE'RE NOT CHANGING

1. **Model architecture** - Stays exactly the same
2. **Pipeline.py** - No changes needed
3. **Configs** - No changes needed
4. **Window extraction** - Same
5. **Preprocessing** - Same (happens after interpolation)

## 📝 IMPLEMENTATION CHECKLIST

- [ ] 1. Back up current `src/experiment/data.py`
- [ ] 2. Add logging import and configuration
- [ ] 3. Modify validation section to interpolate instead of crash
- [ ] 4. Test on one problem file manually
- [ ] 5. Run full test suite (`make test`)
- [ ] 6. Create `test_interpolation.py`
- [ ] 7. Test training doesn't crash (`make train-local`)
- [ ] 8. Commit with clear message about interpolation
- [ ] 9. Document in README about 3.4% interpolation

## ⚠️ CRITICAL SAFETY CHECKS

### Check #1: Only Fz/Pz
```python
# We ONLY interpolate known missing channels
if missing.issubset({'Fz', 'Pz'}):  # STRICT!
    # interpolate
else:
    # still crash - unknown issue
```

### Check #2: Montage Setting
```python
# MUST use on_missing='ignore' or it crashes on missing channels
raw.set_montage('standard_1020', on_missing='ignore')
```

### Check #3: Fallback
```python
try:
    raw.interpolate_bads()
except Exception as e:
    # If MNE can't interpolate, raise original error
    raise ValueError(f"Missing required channels: {sorted(missing)}")
```

## 🎯 SUCCESS CRITERIA

1. ✅ Training runs without crashing
2. ✅ All 7,364 files can be loaded
3. ✅ Test suite still passes
4. ✅ Interpolation logged for tracking
5. ✅ Model performance maintained

## 🏃‍♂️ EXECUTION ORDER

1. **FIRST**: Make the code change (5 minutes)
2. **SECOND**: Test on one problem file (2 minutes)
3. **THIRD**: Run test suite (5 minutes)
4. **FOURTH**: Start training (let it run)
5. **FIFTH**: Create formal test file (10 minutes)

## 💡 WHY THIS PLAN WORKS

1. **Minimal Change**: Only touching the exact failure point
2. **Backward Compatible**: Files with all channels unchanged
3. **Safe**: Only interpolates known missing pattern
4. **Trackable**: Logs which files were interpolated
5. **Testable**: Easy to verify it works

## 🚀 LET'S DO THIS!

Total implementation time: **30 minutes**
Risk level: **LOW**
Confidence: **HIGH**

This is a surgical strike - we fix ONLY the interpolation issue without touching ANYTHING else!
