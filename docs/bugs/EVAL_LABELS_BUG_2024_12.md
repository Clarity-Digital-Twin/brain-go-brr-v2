# CRITICAL BUG: Evaluation Service Does Not Load Labels

**Date**: 2024-12-18
**Severity**: CRITICAL
**Status**: FIXED (2 bugs found and fixed)

## Summary

The evaluation service (`evaluation.py`) creates an `EEGWindowDataset` without passing `label_files`, causing all ground truth labels to be zeros. This resulted in completely invalid evaluation metrics.

## Symptom

After training for 78 epochs (~6 weeks), evaluation on held-out test set returned:
- AUROC: 0.5 (random chance)
- Sensitivity@10FA: 0.0%
- All sensitivities: 0.0%

Initially appeared to be catastrophic overfitting, but was actually a bug in the eval code.

## Root Cause

In `src/brain_brr/cli/services/evaluation.py:142-145`:
```python
dataset = EEGWindowDataset(
    edf_files,
    cache_dir=cache_dir,
    # label_files NOT passed!
)
```

In `src/brain_brr/data/datasets.py:191-201`:
```python
labels = None
if self.label_files is not None:  # Always None in eval!
    label_path = self.label_files[file_idx]
    labels = self._load_labels(...)
```

And fallback at line 319-321:
```python
# ALWAYS return dict with zero labels when none exist
label_tensor = torch.zeros(...)
```

**Result**: All ground truth labels are zeros, metrics computed against zeros.

## Verification

Tested model on dev (validation) vs eval with actual seizure files:

**DEV (training used labels correctly):**
- Batch 1: 55,150 seizure samples → probs mean=0.62, max=0.99
- Model correctly detects seizures!

**EVAL (labels all zeros):**
- ALL batches: "no seizure samples"
- Labels are zeros even for files with known seizures in csv_bi

## Comparison to Training

Training correctly loads labels (`loop.py:826-828`):
```python
train_dataset = EEGWindowDataset(
    train_files,
    label_files=train_label_files,  # CORRECT!
    ...
)
```

## Bug #1: evaluation.py doesn't pass label_files

In `evaluation.py:create_dataloader()`, find csv_bi files for each EDF and pass them:

```python
# Find label files (csv_bi) for each EDF
label_files = [edf.with_suffix('.csv_bi') for edf in edf_files]
# Filter to only existing files
label_files = [lf if lf.exists() else None for lf in label_files]

dataset = EEGWindowDataset(
    edf_files,
    label_files=label_files,  # ADD THIS
    cache_dir=cache_dir,
)
```

**Fixed in**: `src/brain_brr/cli/services/evaluation.py`

## Bug #2: _load_labels doesn't recognize .csv_bi suffix

In `datasets.py:_load_labels()`, the suffix check was wrong:

```python
# WRONG - only matches .csv
if label_path.suffix.lower() == ".csv" and label_path.exists():

# FIXED - matches both .csv and .csv_bi
if label_path.suffix.lower() in (".csv", ".csv_bi") and label_path.exists():
```

The comment said "CSV_BI (Temple/TUSZ)" but the code only checked for `.csv`!

**Fixed in**: `src/brain_brr/data/datasets.py`

## Impact

- 6 weeks of training results VALID (model learned correctly)
- Validation metrics during training VALID (used labels)
- Only the final eval run was broken
- Rerun with fixes completed successfully

## Resolution

After fixing both bugs, eval was rerun on 2024-12-20:

**Final Results (TUSZ Eval, held-out test set)**:
- AUROC: **0.8654** (was 0.5 when broken)
- Sensitivity @ 10 FA/24h: **35.9%** (was 0.0%)
- Sensitivity @ 5 FA/24h: 27.1%
- Sensitivity @ 2.5 FA/24h: 18.6%
- Sensitivity @ 1 FA/24h: 5.8%
- ECE: 0.029
- Dataset: 836 recordings, 127.8 hours

The model works correctly. Training was valid. Only the eval code was broken.

## Related Issues

- Friction point #5 in `FRICTION_POINTS_2024_12.md`: evaluate expects EDF not cache
- The cache issue was a red herring - real bug was label loading
