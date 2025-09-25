# Release v3.0.1: CRITICAL Patient Leakage Fix

## 🚨 EMERGENCY RELEASE - ALL PREVIOUS MODELS INVALID

**Date**: 2025-09-24
**Type**: Critical Bug Fix
**Severity**: P0 BLOCKER

## WARNING: IMMEDIATE ACTION REQUIRED

If you have ANY models trained before this release, they are **scientifically invalid** due to patient-level data leakage between training and validation splits.

## What Happened

During a critical code review, we discovered that patient `aaaaagxr` (and potentially hundreds of others) appeared in BOTH training and validation splits with different recording sessions. This means:

1. **All validation metrics were artificially inflated**
2. **Models learned patient-specific patterns rather than generalizable seizure patterns**
3. **Any published results using these models are invalid**

## The Fix

### Patient-Level Disjoint Splits (P0 BLOCKER FIXED)
- **Before**: File-level alphabetical splitting that mixed patients across splits
- **After**: Using TUSZ official train/dev/eval splits with enforced patient disjointness
- **Verification**: Runtime checks that fail immediately if any patient appears in multiple splits

```python
# New validation at startup
✅ PATIENT DISJOINTNESS VERIFIED - No leakage!
Train: 579 patients, 4667 files
Val: 53 patients, 1832 files
```

### FA Curve Threshold Bug (P0 BLOCKER FIXED)
- **Before**: `sensitivity_at_fa_rates()` passed ignored threshold parameter
- **After**: Properly clones post_cfg and sets tau_on/off for each FA target
- **Impact**: FA curve values were inconsistent with actual thresholds used

## Additional Fixes

- **TensorBoard Import**: Now optional with try/except pattern (was breaking fresh installs)
- **TCN Config**: Removed unused `channels` field that was misleading
- **Manifest Handling**: NPZ files without labels now excluded with warnings
- **CLI Robustness**: Threshold export handles string/numeric key variations

## Required Migration Steps

### 1. Stop All Training Immediately
```bash
tmux kill-session -t train
modal app stop <app-id>
```

### 2. Delete Contaminated Cache
```bash
rm -rf cache/tusz/train_windows/
rm -rf cache/tusz/val_windows/
rm -rf /results/cache/tusz/  # Modal
```

### 3. Update Configuration
```yaml
# configs/local/train.yaml
data:
  data_dir: data_ext4/tusz/edf  # Parent directory
  split_policy: official_tusz    # REQUIRED
```

### 4. Rebuild Cache with Proper Splits
```bash
python -m src build-cache \
  --data-dir data_ext4/tusz/edf \
  --cache-dir cache/tusz
```

### 5. Restart Training from Scratch
```bash
# Local
make train-local

# Modal
modal run --detach deploy/modal/app.py \
  --action train --config configs/modal/train.yaml
```

## Verification Checklist

- [ ] Smoke test shows "PATIENT DISJOINTNESS VERIFIED"
- [ ] No patient IDs appear in both train and val logs
- [ ] Config uses `split_policy: official_tusz`
- [ ] Old cache directories deleted
- [ ] Training restarted from epoch 0

## Impact Assessment

### What This Means for You
- **Research**: Any results must be re-run with proper splits
- **Production**: Models in production are unreliable and must be replaced
- **Publications**: Consider retracting or updating any published results

### Why This Matters
Patient-level splitting is **fundamental** for medical ML. Without it:
- Models memorize specific patients rather than learning seizure patterns
- Validation metrics are meaningless
- Clinical deployment would fail catastrophically

## Lessons Learned

1. **ALWAYS use patient-level splits for medical data**
2. **NEVER split by files when files belong to patients**
3. **ALWAYS validate disjointness before training**
4. **Official splits exist for a reason - use them!**

## Technical Details

### Files Changed
- `src/brain_brr/data/tusz_splits.py` - NEW: Official split handling
- `src/brain_brr/train/loop.py` - FIXED: Uses official splits
- `src/brain_brr/eval/metrics.py` - FIXED: FA curve thresholds
- `src/brain_brr/data/cache_utils.py` - FIXED: Manifest strictness
- `src/brain_brr/cli/cli.py` - FIXED: Threshold export robustness
- All configs updated to use `split_policy: official_tusz`

### New Module: tusz_splits.py
```python
def validate_patient_disjointness(
    train_patients: set[str],
    dev_patients: set[str],
    eval_patients: set[str] | None = None
) -> None:
    """CRITICAL: Prevents patient leakage between splits!"""
    overlap = train_patients & dev_patients
    if overlap:
        raise ValueError(
            f"PATIENT LEAKAGE DETECTED! {len(overlap)} patients "
            f"in both train and dev: {sorted(overlap)[:10]}"
        )
```

## Acknowledgments

Thank you to the code reviewer who caught this critical bug during the audit. This discovery prevented potentially years of invalid research and clinical deployment failures.

## Contact

For questions about this release or help with migration:
- GitHub Issues: https://github.com/Clarity-Digital-Twin/brain-go-brr-v2/issues
- Emergency Support: [Create a critical issue with "PATIENT LEAKAGE" in title]

---

**Remember**: This bug invalidates ALL previous work. There are no shortcuts - you MUST retrain everything with proper patient-disjoint splits.

**Tag**: `v3.0.1-critical-patient-leakage-fix`
**Commit**: Will be tagged after verification