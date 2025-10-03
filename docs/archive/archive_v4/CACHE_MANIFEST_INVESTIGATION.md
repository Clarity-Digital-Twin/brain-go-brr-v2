# Cache & Manifest Investigation - Critical Findings

**Date**: October 3, 2025
**Status**: 🚨 ACTIVE INVESTIGATION - Training Halted
**Issue**: Manifests are incomplete, causing slow startup times and potential data loss

---

## Executive Summary

**PROBLEM**: Both train and dev manifests are missing ~5-13% of cache files despite all NPZ files being valid and loadable.

**IMPACT**:
- 229 train files missing from manifest (4438/4667 = 95.1% coverage)
- 224 dev files missing from manifest (1608/1832 = 87.8% coverage)
- Validation dataset rebuilds index every run (5-10 min startup delay)
- Missing files = missing training data (potential performance loss)

**ROOT CAUSE**: Unknown - `scan_existing_cache()` is skipping valid files

---

## Observed Behavior

### Current Training Pipeline

```
TRAINING DATASET (Train Set):
  ✅ Uses BalancedSeizureDataset
  ✅ Reads cache/tusz/train/manifest.json
  ✅ Loads INSTANTLY from manifest
  ❌ BUT: Manifest only has 4438/4667 files (229 missing!)

VALIDATION DATASET (Dev Set):
  ❌ Uses EEGWindowDataset (ignores manifest!)
  ❌ Rebuilds _dataset_index.json every run
  ❌ Scans all 1832 NPZ files (5-10 minutes)
  ❌ _dataset_index.json gets invalidated when switching smoke ↔ full
  ❌ Manifest exists (13MB) but is NEVER used
```

### What Happens on Startup

```bash
[2025-10-03 13:21:48] INFO [BalancedSeizureDataset] Created with 61616 windows
                                                       ↑ INSTANT (reads manifest)

[2025-10-03 13:21:48] INFO [DATA] Building dataset index for 1832 files...
[2025-10-03 13:21:51] INFO [DATA] Processing file 11/1832: aaaaaajy_s004_t000.edf
[2025-10-03 13:22:44] INFO [DATA] Processing file 121/1832: aaaaagbf_s006_t017.edf
...
[2025-10-03 13:24:55] INFO [DATA] Processing file 351/1832: aaaaahie_s023_t002.edf
                                                       ↑ 5-10 MINUTES (scanning NPZ files)
```

**Question**: Why does validation ignore the manifest and rebuild from scratch?

---

## Verification Results

### Test 1: NPZ File Integrity

```bash
TRAIN CACHE:
  ✅ All 4667 NPZ files load successfully
  ✅ All have "labels" key
  ✅ Zero corrupted files

DEV CACHE:
  ✅ All 1832 NPZ files load successfully
  ✅ All have "labels" key
  ✅ Zero corrupted files
```

**Conclusion**: All cache files are valid. The issue is NOT corruption.

---

### Test 2: Manifest Completeness

```bash
TRAIN MANIFEST (cache/tusz/train/manifest.json):
  Size: 27MB
  Date: Sep 27 00:12 (7 days old)
  Files on disk: 4667
  Files in manifest: 4438
  Missing: 229 (4.9%)

  Manifest breakdown:
    Full seizure windows: Unknown
    Partial seizure windows: Unknown
    No-seizure windows: Unknown
    TOTAL windows: 303,990

DEV MANIFEST (cache/tusz/dev/manifest.json):
  Size: 13MB
  Date: Sep 29 08:09 (4 days old)
  Files on disk: 1832
  Files in manifest: 1608
  Missing: 224 (12.2%)

  Manifest breakdown:
    Full seizure windows: 3,536
    Partial seizure windows: 7,944
    No-seizure windows: 136,744
    TOTAL windows: 148,224
```

**Question**: Why are manifests outdated? When were they last rebuilt?

---

### Test 3: Manifest Rebuild (Just Ran)

```bash
# Rebuilding TRAIN manifest...
scan_existing_cache(Path('cache/tusz/train'))
Result: 303,990 total windows

# Verifying coverage...
Files in new manifest: 4438/4667 (95.1%)
❌ STILL MISSING 229 FILES!

# Rebuilding DEV manifest...
scan_existing_cache(Path('cache/tusz/dev'))
Result: 148,224 total windows

# Verifying coverage...
Files in new manifest: 1608/1832 (87.8%)
❌ STILL MISSING 224 FILES!
```

**CRITICAL**: Even after fresh rebuild, manifests are STILL incomplete!

**Question**: Why is `scan_existing_cache()` skipping 453 valid files?

---

## Code Analysis

### scan_existing_cache() Logic (cache_utils.py:67-145)

```python
def scan_existing_cache(cache_dir: Path) -> dict:
    manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

    npz_files = sorted(cache_dir.glob("*.npz"))  # Gets ALL .npz files

    for npz_path in tqdm(npz_files):
        try:
            with np.load(npz_path) as data:
                if "labels" not in data:
                    # ⚠️ Skip files without labels (cache corruption)
                    warnings.warn(f"NPZ file {npz_path.name} has NO LABELS!")
                    continue  # ← Could this be the issue?

                labels = data["labels"]
        except (OSError, ValueError) as e:
            # Skip corrupted or inaccessible files
            logger.warning(f"Skipping {npz_path.name}: {e}")
            continue  # ← Or this?

        # Categorize windows by seizure content
        n_windows = int(labels.shape[0])
        for w_idx in range(n_windows):
            lbl = labels[w_idx]
            ratio = float((lbl > 0).mean())

            item = {"cache_file": npz_path.name, "window_idx": int(w_idx)}

            if ratio == 0.0:
                manifest["no_seizure"].append(item)
            elif ratio >= 0.99:
                manifest["full_seizure"].append(item)
            else:
                manifest["partial_seizure"].append(item)

    # Save manifest
    with (cache_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
```

**Observations**:
1. Uses `glob("*.npz")` - should match all files ✅
2. Checks for "labels" key - all our files have this ✅
3. Has two `continue` statements that skip files:
   - Missing labels (but we verified all have labels)
   - Load errors (but all load successfully)

**Question**: Are there silent exceptions being caught somewhere?

---

### Validation Dataset Choice (loop.py:539-546)

```python
# CURRENT CODE:
val_dataset = EEGWindowDataset(
    val_files,
    label_files=val_label_files,
    cache_dir=data_cache_root / val_split_name,  # ← cache/tusz/dev
    allow_on_demand=True,
)
```

**Issue**: `EEGWindowDataset` doesn't use manifests!

It:
1. Tries to load `_dataset_index.json` (per-file window counts)
2. If stale/missing, scans ALL NPZ files to build index
3. Never looks at `manifest.json` (which already has all the info!)

**Question**: Why doesn't validation use `BalancedSeizureDataset` like training?

---

## Missing Files Sample

### First 20 Missing from TRAIN Manifest

```
aaaaaaar_s003_t001_windows.npz
aaaaaaar_s003_t002_windows.npz
aaaaaaar_s004_t000_windows.npz
aaaaaaar_s004_t001_windows.npz
aaaaaaar_s005_t000_windows.npz
aaaaaaar_s005_t001_windows.npz
aaaaaaar_s005_t002_windows.npz
aaaaaaar_s006_t000_windows.npz
aaaaaaar_s006_t001_windows.npz
aaaaaaar_s006_t002_windows.npz
... (219 more)
```

**Pattern**: Not obviously related to file size, alphabetical order, or content

### First 20 Missing from DEV Manifest

```
aaaaaajy_s002_t000_windows.npz
aaaaaajy_s002_t001_windows.npz
aaaaadkb_s003_t002_windows.npz
aaaaadkb_s008_t000_windows.npz
aaaaadkb_s010_t001_windows.npz
aaaaagbf_s005_t000_windows.npz
aaaaagbf_s005_t001_windows.npz
aaaaagbf_s006_t001_windows.npz
aaaaagbf_s006_t002_windows.npz
aaaaagbf_s006_t004_windows.npz
... (214 more)
```

**Pattern**: Also no obvious pattern

---

## Questions to Investigate

### Critical Questions

1. **Why are 453 valid files being skipped during manifest build?**
   - All NPZ files load successfully
   - All have "labels" key
   - No exceptions logged
   - Yet manifest is incomplete

2. **Is tqdm silently swallowing exceptions?**
   - `scan_existing_cache()` uses tqdm progress bar
   - Could exceptions be caught and hidden?

3. **Is there a race condition or file locking issue?**
   - WSL2 filesystem behavior?
   - ext4 partition characteristics?

4. **Are window counts the issue?**
   - Does manifest skip files with zero windows?
   - Could label analysis be failing silently?

### Design Questions

5. **Why doesn't validation use manifests?**
   - Training: BalancedSeizureDataset (reads manifest, instant load)
   - Validation: EEGWindowDataset (ignores manifest, scans files)
   - Why the inconsistency?

6. **What is the purpose of _dataset_index.json?**
   - Stores per-file window counts
   - Gets invalidated when file list changes
   - Redundant with manifest.json?

7. **When should manifests be rebuilt?**
   - On cache creation?
   - Manually via script?
   - Automatically on startup?

---

## Proposed Investigation Steps

### Step 1: Debug Manifest Building

```python
# Add detailed logging to scan_existing_cache()
# - Log every file processed
# - Log every skip/continue
# - Log final counts vs expected
# - Capture ALL exceptions

# Run rebuild with debug output
# Identify EXACTLY which files are skipped and WHY
```

### Step 2: Fix Validation to Use Manifests

```python
# Option A: Quick fix (use BalancedSeizureDataset with no balancing)
if (data_cache_root / val_split_name / "manifest.json").exists():
    val_dataset = BalancedSeizureDataset(
        data_cache_root / val_split_name,
        full_ratio=1.0,      # Use ALL windows
        background_ratio=1.0, # Natural distribution
    )
else:
    val_dataset = EEGWindowDataset(...)  # Fallback

# Option B: Teach EEGWindowDataset to read manifests
# - Add manifest loading path to __init__
# - Fall back to index building if manifest missing
```

### Step 3: Verify Modal Pipeline

```python
# Once local is fixed:
# 1. Test on Modal with same cache structure
# 2. Verify manifest usage
# 3. Ensure startup is instant
```

---

## Impact Analysis

### Current State Impact

**Training Performance**:
- Missing 229/4667 train files = **4.9% data loss**
- Potentially ~15,000 missing windows (rough estimate)
- Could impact final model performance

**Development Velocity**:
- **5-10 min wait** every training run for validation index
- **Compounds** when switching between smoke ↔ full
- **Wastes ~30 min per development cycle** (3 runs)

**Modal Training**:
- Same issues will occur on Modal
- Expensive GPU time wasted on index building
- Need to fix local first, then port to Modal

### After Fix Impact

**Training Performance**:
- Full 100% data coverage
- Maximum model performance

**Development Velocity**:
- Instant validation load (uses manifest)
- Zero wasted time
- Smooth smoke ↔ full switching

**Modal Training**:
- Instant startup
- Maximize GPU utilization
- Cost savings

---

## Temporary Workarounds

### For Now (Continue Training)

1. **Accept 95% coverage** - train with current manifest
   - 4.9% data loss is not ideal but not catastrophic
   - Can retrain after fix

2. **Live with slow validation startup**
   - 5-10 min is annoying but not blocking
   - Training itself is 200+ hours anyway

3. **Document for future**
   - Add to POLISH_ITEMS.md as HIGH priority
   - Fix before Modal training run

### For Production

**MUST FIX BEFORE MODAL**:
1. Debug and fix manifest building (100% coverage)
2. Make validation use manifests (instant load)
3. Verify on local first
4. Then deploy to Modal

---

## Next Steps

1. **Add debug logging to `scan_existing_cache()`**
   - Identify EXACTLY why files are skipped
   - Log every decision point

2. **Manually inspect one missing file**
   - Load NPZ
   - Verify labels
   - Run categorization logic
   - Understand failure mode

3. **Fix manifest building**
   - Ensure 100% coverage
   - Rebuild both train + dev

4. **Fix validation dataset**
   - Use manifests instead of index
   - Test smoke and full modes

5. **Verify and document**
   - Run full training smoke test
   - Measure startup time improvement
   - Update documentation

---

## Files Involved

### Core Files
- `src/brain_brr/data/cache_utils.py` - Manifest building logic
- `src/brain_brr/data/datasets.py` - EEGWindowDataset and BalancedSeizureDataset
- `src/brain_brr/train/loop.py` - Dataset instantiation

### Cache Files
- `cache/tusz/train/manifest.json` - Train manifest (95% complete)
- `cache/tusz/dev/manifest.json` - Dev manifest (88% complete)
- `cache/tusz/dev/_dataset_index.json` - Dev index (gets rebuilt every run)

### Config Files
- `configs/local/train.yaml` - Local training config
- `configs/local/smoke.yaml` - Smoke test config

---

## Status

**Current**: Investigation paused, documented findings
**Training**: Halted (tmux session killed)
**Next**: Compact conversation, investigate together with user
**Timeline**: Fix before resuming training

---

**Last Updated**: October 3, 2025 13:30 UTC
**Investigator**: Claude Code (Sonnet 4.5)
**Priority**: 🚨 HIGH - Blocking production training
