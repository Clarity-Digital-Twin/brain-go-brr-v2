TUSZ mysz Seizure Type Crisis - Postmortem

Date: September 21, 2025
Severity: CRITICAL
Impact: 44 seizure events misclassified as background

Discovery Timeline

1. Initial concern raised about verifying all TUSZ seizure labels were being detected
2. Empirical analysis performed on actual TUSZ v2.0.3 CSV files
3. Discovery: `mysz` (myoclonic seizures) exists in data but missing from our code
4. Discovery: `spkz` exists in our code but NOT in any TUSZ v2.0.3 data

The Numbers (TUSZ v2.0.3)

From empirical analysis of 3734 annotation files:
- gnsz: 19,925 occurrences (46.5%) - generalized non-specific
- fnsz: 15,878 occurrences (37.1%) - focal non-specific
- cpsz: 3,498 occurrences (8.2%) - complex partial
- absz: 1,351 occurrences (3.2%) - absence
- spsz: 1,057 occurrences (2.5%) - simple partial
- tcsz: 725 occurrences (1.7%) - tonic-clonic
- tnsz: 349 occurrences (0.8%) - tonic
- **mysz: 44 occurrences (0.1%) - myoclonic** ← MISSING FROM OUR CODE
- spkz: 0 occurrences ← IN OUR CODE BUT DOESN'T EXIST

Affected Files

Two training files contain mysz seizures that were being missed:
1. `dev/01_tcp_ar/081/00008184/s002_2012_02_21/00008184_s002_t000.csv` - 29 mysz events
2. `train/01_tcp_ar/065/00006514/s005_2011_09_21/00006514_s005_t005.csv` - 15 mysz events

Root Cause

Our seizure detection set was based on outdated documentation/assumptions:
```python
# BEFORE (missing mysz, had non-existent spkz):
seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "spkz"}

# AFTER (correct for v2.0.3):
seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```

Impact Assessment

1. **Training Data Corruption**: All caches built before this fix have 44 mysz seizures incorrectly labeled as background
2. **Model Performance**: Models trained on corrupted caches would learn to ignore myoclonic seizures
3. **Clinical Impact**: Myoclonic seizures are rare (0.1%) but clinically important - complete miss rate

Fix Applied

1. Updated `src/brain_brr/data/io.py` lines 297-301 with correct seizure set
2. Deleted ALL existing caches (local and Modal)
3. Rebuilt caches with correct seizure detection
4. Restarted all training runs with fixed data

Verification

After fix, manifest scan shows:
- Partial seizure windows detected
- Full seizure windows detected
- mysz seizures now correctly identified in binary masks

Lessons Learned

1. **Never trust documentation** - always verify against actual data
2. **Empirical validation required** - count actual occurrences in dataset
3. **Version-specific knowledge** - TUSZ v1.x had different seizure types than v2.0.3
4. **Cache invalidation critical** - when detection logic changes, ALL caches must be rebuilt
5. **Rare events matter** - even 0.1% of data can be clinically significant

Prevention Measures

1. Created comprehensive test: `tests/unit/data/test_all_seizure_types_v203.py`
2. Added empirical seizure type counts to documentation
3. Made seizure set explicit in code with detailed comments
4. Added cache rebuild warnings in documentation

Code Reference

The fix in `src/brain_brr/data/io.py:297-301`:
```python
if seizure_labels is None:
    # TUSZ seizure types found in v2.0.3 data (ordered by frequency in corpus):
    # gnsz=generalized non-specific, fnsz=focal non-specific, cpsz=complex partial,
    # absz=absence, spsz=simple partial, tcsz=tonic-clonic, tnsz=tonic, mysz=myoclonic
    seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```

Status: RESOLVED

All caches rebuilt, all training restarted with correct seizure detection.