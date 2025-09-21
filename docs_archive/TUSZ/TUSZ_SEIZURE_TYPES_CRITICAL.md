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

```python
seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```

## Validation
- Shell: grep CSVs to confirm label presence/frequency
- Code: parser’s default seizure label set must include the full list above

