# TUSZ EDF Header Repair - Critical Information for EDF Processing

Status: RESOLVED (2025-09-21)

Resolution summary:
- Implemented header repair (colon→period) fallback and retry inside loader
- Files: src/brain_brr/data/io.py:34 (_repair_edf_header_inplace), integration at src/brain_brr/data/io.py:86
- See also: docs/archive/bugs/RESOLUTION_STATUS.md

## Quick Summary

When processing TUSZ v2.0.3 eval dataset, one file (`aaaaaaaq_s007_t000.edf`) has a malformed EDF header that will crash standard pyedflib loaders. The issue is incorrect date separators (colons instead of periods) at byte offset 168.

## The Specific Problem

**File Path:** `data/tusz/edf/eval/aaaaaaaq/s007_2014/01_tcp_ar/aaaaaaaq_s007_t000.edf`

**Error Message:**
```
the file is not EDF(+) or BDF(+) compliant, the startdate is incorrect,
it might contain incorrect characters, such as ':' instead of '.'
```

**Root Cause:**
- EDF standard requires dates in format: `DD.MM.YY` (periods as separators)
- This file has: `01:01:85` at byte offset 168 (colons instead of periods)
- pyedflib strictly enforces EDF compliance and will reject the file

**Actual Header Bytes:**
- Byte 168-176 (startdate): `b'01:01:85'` ❌ (should be `01.01.85`)
- Byte 176-184 (starttime): `b'00.00.00'` ✅ (correct format)

## How We Fixed It

We implemented a three-layer fallback strategy in `src/seizure_evaluation/utils/edf_repair.py`:

### 1. Try Standard pyedflib Load
First attempt to load normally with `epilepsy2bids.eeg.Eeg.loadEdf()`

### 2. Header Repair + Retry
If pyedflib fails with a startdate/header error:
- Create a temporary copy of the file
- Fix the header by replacing colons with periods at byte offset 168-176
- Retry loading with pyedflib
- Delete the temporary file

### 3. MNE Fallback (Optional)
If repair still fails and MNE is installed, use MNE's more permissive EDF reader

## Implementation Code

```python
from seizure_evaluation.utils.edf_repair import load_with_fallback

# This function handles all three strategies automatically
eeg, load_method = load_with_fallback(edf_path)

# load_method will be one of:
# - "pyedflib" (standard load worked)
# - "pyedflib+repaired" (header repair was needed)
# - "mne" (MNE fallback was used)
```

## Key Functions

**`validate_edf_header(edf_path)`** - Check if an EDF file has valid header format

**`repair_edf_header_copy(edf_path, output_path=None)`** - Create a repaired copy with fixed date/time separators

**`load_with_fallback(edf_path)`** - Main loader that tries all strategies

## Important Notes

1. **This only affects 1 out of 865 files** in TUSZ v2.0.3 eval set
2. **The repair is non-destructive** - works on a temporary copy
3. **Common in clinical data** - Many hospital EDF files have similar formatting issues
4. **pyedflib is correct to be strict** - The file technically violates EDF specification
5. **Results are identical** - The repaired file loads with the same data, just fixed header

## When You Might Encounter This

- Processing TUSZ v2.0.3 dataset (eval split specifically)
- Working with older clinical EDF files
- Using pyedflib or other strict EDF parsers
- Files from certain EEG acquisition systems that use non-standard separators

## Verification

After implementing the fix, we achieved:
- **865/865 files processed** (100% coverage)
- The problematic file loads successfully via `pyedflib+repaired`
- All 19 channels at 256 Hz correctly loaded
- Model inference runs normally on the repaired data

## For Other Agents/Developers

If you're building a TUSZ evaluation pipeline:

1. **Don't skip this file** - It contains valid seizure data
2. **Use our repair utility** - Copy the `edf_repair.py` module
3. **Track load methods** - Know which files needed repair for reproducibility
4. **Consider MNE as backup** - More permissive but slower than pyedflib

## Technical Details

**Header Structure (EDF Standard):**
- Bytes 0-7: Version (should be "0       ")
- Bytes 8-87: Patient ID
- Bytes 88-167: Recording ID
- **Bytes 168-175: Start date (DD.MM.YY)**
- **Bytes 176-183: Start time (HH.MM.SS)**
- Bytes 184-191: Header size
- ... continues with signal specifications

The issue is specifically in bytes 168-175 where the date separator character is wrong.

## References

- EDF Specification: https://www.edfplus.info/specs/edf.html
- pyedflib Documentation: https://github.com/holgern/pyedflib
- TUSZ Dataset: https://isip.piconepress.com/projects/tuh_eeg/
- Our Implementation: `src/seizure_evaluation/utils/edf_repair.py`

---

*This document created from actual debugging experience processing TUSZ v2.0.3 eval set with SeizureTransformer. The fix has been validated and allows 100% file coverage.*
