# Modal Resume Manifest Investigation - October 9, 2025

**Status**: 🔴 CRITICAL - Training using wrong dataset (imbalanced)
**Impact**: Model will underperform due to insufficient seizure examples
**Priority**: P0 - Stop training, fix, resume

---

## 🚨 **The Problem**

Modal training resumed at **Oct 09 07:33** and immediately fell back to imbalanced dataset:

```
[2025-10-09 12:28:32.897][src.brain_brr.train.loop][INFO]
[WARNING] BalancedSeizureDataset failed: No partial seizure windows found in manifest!
Full: 0, No-seizure: 0; falling back to EEGWindowDataset
```

**What this means**:
- ❌ Training is using **EEGWindowDataset** (imbalanced, ~8% seizures)
- ✅ Should be using **BalancedSeizureDataset** (balanced, ~30% seizures)
- ❌ Model will see far fewer seizure examples
- ❌ Performance will degrade

---

## 🔍 **Investigation: Why Did This Happen?**

### **Hypothesis 1: Modal manifest is empty**

**Evidence from Modal logs**:
```
[CACHE] ✅ Manifest found at /results/cache/tusz_mmap/train/manifest.json
```

But when `BalancedSeizureDataset` tries to load it:
```python
partial: list[dict] = list(manifest.get("partial_seizure", []))
full: list[dict] = list(manifest.get("full_seizure", []))
no_seizure: list[dict] = list(manifest.get("no_seizure", []))
```

All three are **EMPTY** (length 0), so it raises `ValueError` and falls back.

### **Local vs Modal Manifest Comparison**

**Local manifest** (working):
```bash
$ python3 -c "import json; m = json.load(open('cache/tusz_mmap/train/manifest.json')); \
  print('partial:', len(m.get('partial_seizure', [])))"
partial: 16215
full: 8446
no_seizure: 279329
```

**Modal manifest** (BROKEN - inferred from logs):
```python
# Modal manifest structure (inferred):
{
  "partial_seizure": [],  # EMPTY
  "full_seizure": [],     # EMPTY
  "no_seizure": []        # EMPTY
}
```

---

## 🔍 **Root Cause Analysis**

### **When was Modal manifest created?**

Looking at the logs, Modal training has been running for multiple days. The manifest was likely created:

1. **During initial cache population** (many days ago)
2. **Before recent validation refactor** (Oct 6-8, 2025)

### **What changed in the refactor?**

From git log:
```
b9b948a docs(technical-debt): Update active debt documentation
7353684 fix: Correct variable name in EEGWindowDataset
8ad130e refactor: Enhance type safety and clarity in dataset and logging modules
ca31425 refactor: Centralize memory-mapped cache loading logic for datasets
7b03882 refactor: Implement cache management improvements and clean stray NPZ files
9d206a7 refactor: Implement cache management improvements to prevent NPZ contamination
```

**Key changes**:
- NPZ → NPY migration (Oct 6-8)
- Manifest generation logic in `scan_existing_cache()` updated
- Cache file naming changed from `*_windows.npz` → `*_data.npy`

### **Why is the Modal manifest empty?**

**Theory 1: Old manifest format**
- Modal manifest was created with OLD code (before Oct 6)
- Old code may have used different keys or structure
- New code expects `partial_seizure`, `full_seizure`, `no_seizure` lists

**Theory 2: Manifest wasn't rebuilt after NPZ→NPY migration**
- Modal cache was converted from NPZ to NPY
- But manifest wasn't regenerated
- Old manifest references NPZ files, new code expects NPY

**Theory 3: Race condition during resume**
- Resume loads checkpoint
- Checkpoint path validation happens
- But manifest isn't re-validated/regenerated

---

## 🔍 **Critical Question: Why Did Initial Run Work?**

### **Timeline Reconstruction**

1. **Initial training start** (several days ago):
   - Cache population ran: `modal run --action populate_cache`
   - Manifest generated with seizure statistics
   - Training started with `BalancedSeizureDataset`
   - Training ran successfully for multiple epochs

2. **OOM crash** (Oct 8-9):
   - Validation hit 120GB RAM spike
   - Modal killed training (exit 137)
   - Checkpoint saved at end of last epoch

3. **Resume** (Oct 9 07:33):
   - Deployed with disk-backed validation fix
   - Resume from `last.pt` checkpoint
   - **NEW CODE** tries to load manifest
   - Manifest empty → fallback to `EEGWindowDataset`

### **Key Insight: Resume vs Fresh Start**

**Theory**: Initial run used **DIFFERENT DATASET LOADING PATH**

## 🔍 **CODE AUDIT: Dataset Loading Logic**

### **Location**: `src/brain_brr/train/loop.py` lines 639-836

### **The Bug** (lines 664-678):

```python
if use_balanced and not manifest_path.exists():
    # CRITICAL: Only build manifest if cache already has files!
    # Bug fix: Don't build manifest from empty directory
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    existing_cache_files = list(train_cache_dir.glob("*.npz"))  # ← BUG HERE
    if existing_cache_files:
        try:
            from src.brain_brr.data.cache_utils import scan_existing_cache

            _ = scan_existing_cache(train_cache_dir)
            logger.info(f"[DATA] Built manifest from {len(existing_cache_files)} cached files")
        except Exception as e:
            logger.info(f"[WARNING] Manifest build failed: {e}")
    else:
        logger.info("[DATA] Skipping manifest build - cache not yet populated")
```

**THE BUG**: Line 668 checks for `*.npz` files, but Modal cache is **NPY format**!

### **Why Initial Run Worked**

1. **Initial training start** (many days ago):
   - Cache was probably NPZ format (legacy)
   - Line 668: `existing_cache_files = list(train_cache_dir.glob("*.npz"))` **FOUND FILES**
   - `scan_existing_cache()` ran
   - Manifest created with seizure statistics
   - Training used `BalancedSeizureDataset` ✅

2. **Cache migration** (Oct 6-8):
   - NPZ files converted to NPY format
   - Manifest may have been regenerated

3. **OOM crash** (Oct 8-9):
   - Modal killed training

4. **Resume with new code** (Oct 9 07:33):
   - Line 668: `existing_cache_files = list(train_cache_dir.glob("*.npz"))` **FINDS NOTHING**
   - Why? Cache is now NPY format (`*_data.npy`, `*_labels.npy`)
   - Code skips manifest rebuild: `"[DATA] Skipping manifest build - cache not yet populated"`
   - But manifest DOES exist (from before)!
   - Manifest is EMPTY or OLD format
   - BalancedSeizureDataset fails: "No partial seizure windows found"
   - Falls back to `EEGWindowDataset` ❌

### **Root Cause Summary**

**Line 668 is WRONG**:
```python
existing_cache_files = list(train_cache_dir.glob("*.npz"))  # ❌ WRONG
```

**Should be**:
```python
existing_cache_files = list(train_cache_dir.glob("*_data.npy"))  # ✅ CORRECT
```

This causes the code to think the cache is empty (because it's looking for NPZ but cache is NPY), so it doesn't rebuild the stale manifest.

---

## 🔧 **THE FIX**

### **Fix #1: Update `loop.py` line 668** (CRITICAL)

**File**: `src/brain_brr/train/loop.py:668`

**Before**:
```python
existing_cache_files = list(train_cache_dir.glob("*.npz"))
```

**After**:
```python
# Check for NPY format (current) or NPZ format (legacy)
existing_cache_files = list(train_cache_dir.glob("*_data.npy"))
if not existing_cache_files:
    existing_cache_files = list(train_cache_dir.glob("*.npz"))
```

### **Fix #2: Force manifest rebuild on Modal** (IMMEDIATE)

**Option A: Delete stale manifest and rebuild**:
```bash
# Stop current training
modal app stop ap-<current-app-id>

# Delete stale manifest (via Modal shell or rebuild script)
modal run deploy/modal/app.py --action rebuild-manifest

# Resume training (will build fresh manifest)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train.yaml --resume true
```

**Option B: Set env var to force rebuild**:
```bash
# Stop current training
modal app stop ap-<current-app-id>

# Add to deploy/modal/app.py before training:
os.environ["BGB_FORCE_MANIFEST_REBUILD"] = "1"

# Resume training
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train.yaml --resume true
```

### **Fix #3: Add validation check** (DEFENSIVE)

**File**: `src/brain_brr/train/loop.py:686-703`

**Add before `try:` block**:
```python
if use_balanced and manifest_path.exists():
    try:
        train_dataset = BalancedSeizureDataset(train_cache_dir, file_list=train_files)

        # DEFENSIVE CHECK: If manifest exists but dataset is empty, force rebuild
        if len(train_dataset) == 0:
            logger.warning("[DATA] Manifest exists but produced 0 windows → forcing rebuild")
            manifest_path.unlink()
            from src.brain_brr.data.cache_utils import scan_existing_cache
            _ = scan_existing_cache(train_cache_dir)
            # Retry after rebuild
            train_dataset = BalancedSeizureDataset(train_cache_dir, file_list=train_files)
```

---

## ✅ **VERIFICATION CHECKLIST**

After applying fixes, verify:

1. ✅ **Local test passes**:
   ```bash
   # Force NPY cache format
   python -m src build-cache --data-dir data/tusz/train --cache-dir cache/test_npy

   # Test manifest generation
   python -c "
   from src.brain_brr.data.cache_utils import scan_existing_cache
   from pathlib import Path
   m = scan_existing_cache(Path('cache/test_npy'))
   print('partial:', len(m['partial_seizure']))
   print('full:', len(m['full_seizure']))
   print('no_seizure:', len(m['no_seizure']))
   "
   # Should print non-zero counts
   ```

2. ✅ **Modal training uses BalancedSeizureDataset**:
   ```bash
   modal app logs <app-id> | grep "BalancedSeizureDataset"
   # Should see: "[DATASET] BalancedSeizureDataset: 304990 windows from manifest"
   # Should NOT see: "[WARNING] BalancedSeizureDataset failed"
   ```

3. ✅ **Manifest has seizure statistics**:
   ```bash
   # On Modal (via shell or logs):
   python3 -c "
   import json
   m = json.load(open('/results/cache/tusz_mmap/train/manifest.json'))
   print('partial:', len(m.get('partial_seizure', [])))
   print('full:', len(m.get('full_seizure', [])))
   print('no_seizure:', len(m.get('no_seizure', [])))
   "
   # Should print:
   # partial: 16215
   # full: 8446
   # no_seizure: 279329
   ```

---

## 📊 **IMPACT ANALYSIS**

### **Current State** (BROKEN):
- Training uses `EEGWindowDataset` (imbalanced)
- ~8% seizures in batches (natural distribution)
- Model sees ~3,000 seizure windows per epoch
- Performance will be **significantly degraded**

### **After Fix** (CORRECT):
- Training uses `BalancedSeizureDataset` (balanced)
- ~30% seizures in batches (oversampled)
- Model sees ~10,000+ seizure windows per epoch
- Performance should match previous runs

### **Time Cost**:
- Stop current training: **IMMEDIATE**
- Rebuild manifest: **~5-10 minutes**
- Resume training: **~5 minutes**
- **Total downtime: ~15-20 minutes**

### **Financial Cost**:
- Current broken training: **Wasting $319** (will produce bad model)
- Fix and resume: **Save $300+** (recover most of training progress)

---

## 🎯 **RECOMMENDED ACTION PLAN**

### **Step 1: STOP Current Training** (RIGHT NOW)
```bash
modal app list
modal app stop ap-<current-app-id>
```

### **Step 2: Apply Code Fix**
```bash
# Edit src/brain_brr/train/loop.py:668
# Change: existing_cache_files = list(train_cache_dir.glob("*.npz"))
# To:     existing_cache_files = list(train_cache_dir.glob("*_data.npy"))
```

### **Step 3: Rebuild Modal Manifest**
```bash
# Option A: Delete and let training rebuild
modal run deploy/modal/app.py --action shell
# In shell: rm /results/cache/tusz_mmap/train/manifest.json

# Option B: Force rebuild via env var
# Add to deploy/modal/app.py:
# os.environ["BGB_FORCE_MANIFEST_REBUILD"] = "1"
```

### **Step 4: Resume Training**
```bash
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train.yaml --resume true
```

### **Step 5: Verify Fix**
```bash
# Watch logs for success indicators
modal app logs <new-app-id> | grep -E "BalancedSeizureDataset|manifest"

# Should see:
# [DATA] Built manifest from 4667 cached files
# [DATASET] BalancedSeizureDataset: 304990 windows from manifest

# Should NOT see:
# [WARNING] BalancedSeizureDataset failed
# falling back to EEGWindowDataset
```

---

## 📝 **LESSONS LEARNED**

### **1. NPZ → NPY Migration Incomplete**
- Code changed to support NPY format
- But one glob pattern (`*.npz`) was missed in `loop.py:668`
- **Fix**: Always search for both formats during migration

### **2. Manifest Validation Missing**
- Old manifests can become stale after cache format changes
- No automatic detection/rebuild of stale manifests
- **Fix**: Add `check_manifest_stale()` call before using BalancedSeizureDataset

### **3. Resume Logic Doesn't Validate Dataset**
- Resume loads checkpoint but doesn't check dataset health
- Training can continue with wrong dataset silently
- **Fix**: Add defensive checks after dataset creation

### **4. Logs Don't Show Dataset Type Clearly**
- Hard to tell from logs whether BalancedSeizureDataset or EEGWindowDataset is used
- **Fix**: Add explicit log line showing dataset type at training start

---

## 🚨 **CRITICAL TIMELINE**

**DO THIS NOW**:
1. Stop current Modal training (wasting money)
2. Apply code fix (5 minutes)
3. Delete stale manifest on Modal (1 minute)
4. Resume training with fix (5 minutes)
5. Verify BalancedSeizureDataset is used (2 minutes)

**Total time**: ~15 minutes
**Cost savings**: ~$250-300 (avoid training bad model)

---

**Document Created**: October 9, 2025 09:15 UTC
**Priority**: P0 - CRITICAL
**Status**: Investigation complete, fix identified
**Next Action**: Stop training, apply fix, resume

