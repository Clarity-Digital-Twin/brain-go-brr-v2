# CRITICAL FIX: Patient Leakage in Train/Val Splits

## Date: 2025-09-24

## **CRITICAL BUG DISCOVERED**

### The Problem
Our previous implementation had **PATIENT-LEVEL DATA LEAKAGE** between training and validation sets:
- Used naive alphabetical file-based splitting
- Patient `aaaaagxr` appeared in BOTH train (sessions s018) and val (sessions s001, s002, s004)
- This invalidated ALL validation metrics and model selection

### Impact
- **ALL previous training runs are scientifically invalid**
- **ALL validation metrics were artificially inflated**
- **ALL GPU hours spent (local RTX 4090 + Modal A100) produced unusable models**

## **THE FIX**

### What We Changed
1. **Created `src/brain_brr/data/tusz_splits.py`** - Proper TUSZ split handling
2. **Updated training loop** - Uses official TUSZ train/dev/eval splits
3. **Added patient disjointness validation** - Fails fast if any patient appears in multiple splits
4. **Updated configs** - Now use `split_policy: official_tusz`

### The Correct Protocol
```
TUSZ Official Splits (PATIENT-DISJOINT):
├── train/  (579 patients, 4667 files) → Training
├── dev/    (53 patients, 1832 files)  → Validation/Hyperparameter tuning
└── eval/   (43 patients, 865 files)   → Final testing (NEVER TOUCH until end)
```

## **Action Items**

### Immediate
- [x] Stop all training (local and Modal)
- [x] Delete contaminated cache directories
- [x] Implement proper split handling
- [x] Update all configs

### Next Steps
- [ ] Rebuild cache with correct splits: `cache/tusz/train/` and `cache/tusz/dev/`
- [ ] Restart training with proper splits
- [ ] After training: Evaluate ONCE on `eval/` for final metrics

## **Configuration**

### configs/local/train.yaml
```yaml
data:
  data_dir: data_ext4/tusz/edf        # Parent dir containing train/dev/eval
  cache_dir: cache/tusz                # Will create train/ and dev/ subdirs
  split_policy: official_tusz          # Use TUSZ official patient-disjoint splits!
```

## **Verification**

Run this to verify no patient leakage:
```python
from src.brain_brr.data.tusz_splits import load_tusz_for_training
splits = load_tusz_for_training(data_root, use_eval=False, verbose=True)
# Will raise ValueError if any patient appears in multiple splits
```

## **Key Learnings**

1. **ALWAYS use patient-level splits** for medical data
2. **NEVER split by files** when files belong to patients
3. **ALWAYS validate disjointness** before training
4. **Official splits exist for a reason** - use them!

## **Credit**

Bug discovered through careful analysis of BUGS.md report.
Fix implemented immediately upon discovery.

---

**ALL PREVIOUS RESULTS ARE INVALID. Start fresh with proper splits.**