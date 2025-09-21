# CRITICAL: TUSZ Seizure Type Documentation

## ⚠️ VERSION MATTERS - USE v2.0.3 DATA ONLY

**THIS DOCUMENT IS CRITICAL FOR CORRECT SEIZURE DETECTION TRAINING**

---

## Executive Summary

The TUH EEG Seizure Corpus (TUSZ) v2.0.3 contains **8 distinct seizure types** that MUST ALL be detected for proper training. Missing even one type (like `mysz`) means losing training data and degrading model performance.

## Seizure Types in TUSZ v2.0.3

Based on empirical analysis of the actual CSV annotations in our data (2025-09-21):

| Seizure Type | Full Name | Occurrences | Frequency |
|--------------|-----------|-------------|-----------|
| `gnsz` | Generalized Non-Specific Seizure | 23,804 | 46.5% |
| `fnsz` | Focal Non-Specific Seizure | 19,000 | 37.1% |
| `cpsz` | Complex Partial Seizure | 3,597 | 7.0% |
| `absz` | Absence Seizure | 2,507 | 4.9% |
| `spsz` | Simple Partial Seizure | 942 | 1.8% |
| `tcsz` | Tonic-Clonic Seizure | 857 | 1.7% |
| `tnsz` | Tonic Seizure | 410 | 0.8% |
| `mysz` | Myoclonic Seizure | 44 | 0.1% |
| **TOTAL** | | **51,161** | **100%** |

## ⚠️ Critical Implementation Requirements

### CORRECT Implementation (as of v0.2.0+)
```python
seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```

### INCORRECT Implementations (DO NOT USE)
```python
# WRONG - Missing mysz (loses 44 seizure events)
seizure_labels = {"seiz", "gnsz", "fnsz", "spsz", "cpsz", "absz", "tnsz", "tcsz", "spkz"}

# WRONG - Only looking for generic "seiz" (loses ALL seizures - TUSZ doesn't use this label!)
seizure_labels = {"seiz"}

# WRONG - Missing multiple types
seizure_labels = {"gnsz", "fnsz", "cpsz", "tcsz"}  # Missing 40% of seizures!
```

## Version-Specific Differences

### TUSZ v2.0.3 (CURRENT - USE THIS)
- Released: 2024-06-18
- Has all 8 seizure types listed above
- Fixed annotation issues from earlier versions
- This is what our codebase expects

### TUSZ v1.x (OUTDATED - DO NOT USE)
- May have different seizure type labels
- Shah et al. 2018 paper lists additional types (CNSZ, ATSZ) not found in v2.0.3
- Different CSV format structure

## Data Validation

### Quick Validation Check
Run this command to verify your TUSZ data has all expected seizure types:
```bash
find /path/to/tusz/edf -name "*.csv" -exec grep -h "sz," {} \; 2>/dev/null | \
    cut -d',' -f4 | sort | uniq -c | sort -rn
```

Expected output should match the table above.

### In Code Validation
The parser in `src/brain_brr/data/io.py` will detect these seizure types:
```python
# Line 297-301 in io.py
if seizure_labels is None:
    # TUSZ seizure types found in v2.0.3 data (ordered by frequency in corpus):
    # gnsz=generalized non-specific, fnsz=focal non-specific, cpsz=complex partial,
    # absz=absence, spsz=simple partial, tcsz=tonic-clonic, tnsz=tonic, mysz=myoclonic
    seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```

## Why This Matters

1. **Training Coverage**: Missing `mysz` means losing 44 seizure events. While small (0.1%), these are RARE seizures that the model needs to learn.

2. **Class Imbalance**: TUSZ already has extreme class imbalance (>99% background). Every seizure matters for the BalancedSeizureDataset sampling.

3. **Clinical Relevance**: Different seizure types have different clinical presentations. Training on all types improves generalization.

4. **Benchmarking**: When comparing to papers using TUSZ, ensure they're using the same version and detecting all types.

## References

1. **TUSZ v2.0.3 AAREADME.txt**: Official documentation (in `/data_ext4/tusz/`)
2. **Shah et al. 2018**: "The Temple University Hospital Seizure Detection Corpus" - describes v1.x
3. **SeizureTransformer (2025)**: Uses TUSZ v2.0.3 for training
4. **Our empirical analysis**: Direct count from 51,161 seizure annotations in the actual CSV files

## Changelog

- **2025-09-21**: Discovered missing `mysz` type, added to v0.2.0
- **2025-09-21**: Removed non-existent `spkz` type
- **2025-09-20**: Initial implementation with incomplete type list

---

**REMEMBER**: Always validate seizure type detection against the actual data, not just documentation. Papers may reference older versions with different labels.