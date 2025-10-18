# EVAL CACHE BUILD FAILURE - DETECTIVE REPORT 🔍

**Date**: 2025-10-17
**Detective**: Claude (ML Codebase Investigator)
**Case**: Why the fuck can't we build eval cache when train/dev worked?

---

## 🎬 EXECUTIVE SUMMARY (TL;DR FOR BRODIE)

**Problem**: Can't build eval cache. Command crashes with "Cache not found" error.

**Root Cause**: Train/dev caches were built DIFFERENTLY than we thought!
- They were built as NPZ first, THEN converted to NPY using `scripts/convert_cache_to_mmap.py`
- We're trying to build eval cache directly to NPY, which the code **DOES NOT SUPPORT**
- The code EXPECTS cache files to already exist during dataset initialization
- My `allow_on_demand=True` edit is **USELESS** because crash happens in constructor BEFORE __getitem__()

**The Fix**: Need to modify `EEGWindowDataset.__init__()` to:
1. Catch FileNotFoundError during index building
2. If `allow_on_demand=True`, process file on-the-fly and save to cache
3. Continue with normal index building

**Quick Win**: Could also just set `cache_dir=None` and manually save files afterward (Solution C)

**Status**:
- ✅ Root cause identified
- ✅ Solution designed
- ⏸️ Waiting for approval to implement
- 🔄 Need to revert useless commit b17544ac

---

## 🎯 THE MYSTERY

Attempted to build eval cache using:
```bash
python -m src build-cache --data-dir data_ext4/tusz/edf/eval --cache-dir cache/tusz_mmap/eval
```

**Result**: FAILED with:
```
Cache not found for aaaaaaaq_s006_t000.edf at cache/tusz_mmap/eval.
Run populate_cache first: modal run deploy/modal/app.py --action populate-cache
```

---

## 🔎 KEY FINDINGS

### Finding 1: Train/Dev Caches Were NOT Built the Same Way

**Evidence**:
- `scripts/convert_cache_to_mmap.py` exists (lines 1-200)
- `docs/02-data/cache-layout.md` lines 38-50 show the REAL workflow:

```bash
# Convert NPZ cache to mmap format (one-time per split)
python scripts/convert_cache_to_mmap.py \
  --source cache/tusz/train \
  --dest cache/tusz_mmap/train

python scripts/convert_cache_to_mmap.py \
  --source cache/tusz/dev \
  --dest cache/tusz_mmap/dev

# Regenerate manifests and indices for the mmap cache
python -m src scan-cache --cache-dir cache/tusz_mmap/train
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
```

**What this means**:
- Train/dev caches were built in TWO steps:
  1. First: Build compressed NPZ caches (source: `cache/tusz/{train,dev}/`)
  2. Then: Convert NPZ → NPY using `convert_cache_to_mmap.py`

**Problem**: We're trying to build eval cache directly to NPY format, which the code DOES NOT SUPPORT!

---

### Finding 2: The Code Architecture Prevents Direct NPY Cache Building

**Evidence from `src/brain_brr/data/datasets.py` lines 100-116**:

```python
# Pre-compute or load window counts for each file
logger.info(f"[DATA] Building dataset index for {len(self.edf_files)} files...")
for i, edf_path in enumerate(self.edf_files):
    if i % 10 == 0:
        logger.info(
            f"[DATA] Processing file {i + 1}/{len(self.edf_files)}: {edf_path.name}"
        )
    if self.cache_dir is not None:
        cache_path = self.cache_dir / f"{edf_path.stem}_data.npy"
        try:
            windows_mmap, _labels_mmap = self._load_cache_for_worker(cache_path)
            n_windows = windows_mmap.shape[0]
        except FileNotFoundError:
            raise FileNotFoundError(  # <-- 💥 BOOM! Dies here during index building
                f"Cache not found for {edf_path.name} at {cache_path.parent}. "
                f"Run populate_cache first: "
                f"modal run deploy/modal/app.py --action populate-cache"
            ) from None
```

**The Problem**:
1. When `cache_dir` is set, the code EXPECTS cache files to ALREADY EXIST
2. It tries to load them during the **INDEX BUILDING** phase (constructor)
3. If files don't exist → FileNotFoundError → CRASH
4. The `allow_on_demand=True` flag I edited only affects `__getitem__()` (line 194-203)
5. But we never GET to `__getitem__()` because the constructor crashes first!

**My edit to cli.py line 248 was USELESS** because:
- `allow_on_demand=True` controls whether to process files on-the-fly in `__getitem__()`
- But the constructor crashes BEFORE any `__getitem__()` calls happen
- The crash happens at line 112 in datasets.py during index building

---

### Finding 3: No Old NPZ Eval Cache Exists

**Evidence**:
```bash
$ ls -la cache/
total 16
drwxr-xr-x  3 jj jj 4096 Oct 12 09:44 .
drwxrwxrwx 27 jj jj 4096 Oct 17 16:58 ..
-rw-r--r--  1 jj jj  668 Oct 12 09:44 .gitkeep
drwxrwxrwx  5 jj jj 4096 Oct 17 16:57 tusz_mmap

$ ls -la cache/tusz/ 2>/dev/null
<no such directory>
```

**What this means**:
- No old NPZ caches exist (likely deleted after conversion)
- We cannot use `convert_cache_to_mmap.py` because there's no source NPZ cache
- We need to build eval cache FROM SCRATCH from raw EDFs

---

### Finding 4: Modal's populate-cache Just Copies from S3

**Evidence from `deploy/modal/app.py` lines 183-213**:

```python
@app.function(...)
def populate_cache():
    """One-time copy of cache from S3 to Modal SSD volume.

    This copies ~507GB of memory-mapped NPY files from S3 to the Modal
    persistent SSD volume for fast, reliable training access.
    """
    src = Path("/s3_cache")  # S3 mount (memory-mapped NPY format)
    dst = Path("/results/cache/tusz_mmap")  # SSD volume (memory-mapped cache)
    # ... just copies files ...
```

**What this means**:
- Modal's `populate-cache` doesn't BUILD caches
- It just copies pre-built caches from S3 → Modal SSD
- Someone built the caches ELSEWHERE and uploaded them to S3

---

## 🧩 THE MISSING PIECE

**CRITICAL QUESTION**: How were the original NPZ caches built for train/dev?

**Hypothesis 1**: Old version of `build-cache` could create NPZ files directly
- Maybe the code was different before
- Maybe `allow_on_demand` worked differently
- Maybe there was a separate script

**Hypothesis 2**: Caches were built manually using a different process
- Maybe direct calls to `EEGWindowDataset` without cache_dir
- Maybe a custom script that no longer exists
- Maybe on a different machine

**Evidence needed**:
- Check git history for old versions of `build-cache` command
- Check for any deleted scripts in git history
- Check if there's documentation about the ORIGINAL cache building process

---

## 🎯 SOLUTIONS

### Solution A: Fix EEGWindowDataset to Support On-Demand Building

**What to change**:
```python
# In src/brain_brr/data/datasets.py lines 106-127
if self.cache_dir is not None:
    cache_path = self.cache_dir / f"{edf_path.stem}_data.npy"
    try:
        windows_mmap, _labels_mmap = self._load_cache_for_worker(cache_path)
        n_windows = windows_mmap.shape[0]
    except FileNotFoundError:
        # NEW: Check if on-demand building is allowed
        if not self.allow_on_demand:
            raise FileNotFoundError(...) from None

        # NEW: Process file on-demand and cache it
        windows_arr, labels_arr = self._process_file(edf_path, i)
        n_windows = windows_arr.shape[0]

        # NEW: Save to cache for next time
        data_file = self.cache_dir / f"{edf_path.stem}_data.npy"
        labels_file = self.cache_dir / f"{edf_path.stem}_labels.npy"
        np.save(data_file, windows_arr)
        if labels_arr is not None:
            np.save(labels_file, labels_arr)
```

**Pros**:
- Makes `build-cache` work as expected
- One-step process: raw EDFs → NPY cache
- No intermediate NPZ files needed

**Cons**:
- Changes core dataset logic
- Needs testing to ensure it doesn't break train/dev loading
- Need to handle cache writing errors

---

### Solution B: Create Two-Step Process with NPZ Intermediate

**Step 1**: Build NPZ cache first
```python
# Create a new script: scripts/build_npz_cache.py
# Or modify build-cache to support --format npz flag
dataset = EEGWindowDataset(
    edf_files,
    label_files=label_files,
    cache_dir=None,  # <-- No cache_dir = forces on-demand processing
    allow_on_demand=True
)

# Then manually iterate and save as NPZ:
for i in range(len(dataset)):
    item = dataset[i]
    # ... save to NPZ format ...
```

**Step 2**: Convert NPZ → NPY
```bash
python scripts/convert_cache_to_mmap.py \
  --source cache/tusz_npz/eval \
  --dest cache/tusz_mmap/eval
```

**Pros**:
- Matches the proven train/dev workflow
- No changes to core dataset code
- NPZ intermediate format can be backed up

**Cons**:
- Two-step process (slower)
- Requires disk space for both NPZ and NPY (2× storage)
- More complex workflow

---

### Solution C: Use EEGWindowDataset with cache_dir=None

**Direct approach**:
```python
# Modify cli.py build-cache command:
if not dest_cache_exists:
    # First pass: build index without cache
    temp_dataset = EEGWindowDataset(
        edf_files,
        label_files=label_files,
        cache_dir=None,  # <-- Forces on-demand processing
        allow_on_demand=True
    )

    # Second pass: iterate and save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(temp_dataset)):
        item = temp_dataset[idx]
        file_id = item['file_id']

        # Save window data
        # ... accumulate windows per file ...
        # ... save as NPY when file complete ...
```

**Pros**:
- Uses existing code without modifications
- Single-step process
- No intermediate NPZ files

**Cons**:
- More complex logic in CLI command
- Need to accumulate windows per file in memory
- Potential for high memory usage

---

## ✅ RECOMMENDED SOLUTION

**Use Solution A** (Fix EEGWindowDataset) because:

1. **Cleanest**: Makes `build-cache` work as users expect
2. **Consistent**: `allow_on_demand=True` should mean "build if missing"
3. **Future-proof**: Any future cache building will benefit
4. **Minimal risk**: Only affects the index building when cache is missing

**Implementation checklist**:
- [ ] Modify `EEGWindowDataset.__init__()` lines 106-127
- [ ] Add cache writing logic when `allow_on_demand=True`
- [ ] Add error handling for cache write failures
- [ ] Test with a small eval subset (3 files)
- [ ] Test that train/dev loading still works
- [ ] Document the updated workflow

---

## 🔍 GIT HISTORY FINDINGS

**Commit b17544ac (TODAY)**:
```diff
-        _ = EEGWindowDataset(edf_files, label_files=label_files, cache_dir=cache_dir)
+        _ = EEGWindowDataset(edf_files, label_files=label_files, cache_dir=cache_dir, allow_on_demand=True)
```

**Verdict**: My edit added `allow_on_demand=True` but this **DOES NOT FIX THE PROBLEM** because:
- The crash happens in `__init__()` during index building (line 112 of datasets.py)
- `allow_on_demand` only affects `__getitem__()` behavior (line 194-203 of datasets.py)
- We never reach `__getitem__()` because the constructor crashes first

**Other key commits**:
- `fe2b6e5b` (2024): "Add script to convert NPZ cache to memory-mapped NPY format"
  - Confirms NPZ caches existed BEFORE NPY conversion
  - NPZ → NPY was a one-time migration, not the original build process
- `2623a1e0` (2024): "Update cache configurations to use memory-mapped NPY format"
  - This was when we switched from NPZ to NPY production format

**The smoking gun**: No commits show how the ORIGINAL NPZ caches were built!
- Likely built on a different machine or with old code that's been deleted
- Or built manually with direct Python scripts that weren't committed

---

## 🚨 ACTION ITEMS

1. **REVERT my cli.py edit** (commit b17544ac) - it does nothing useful
2. **Implement Solution A** in datasets.py to fix the root cause
3. **Test with smoke test** (`BGB_LIMIT_FILES=3`)
4. **Build full eval cache** once verified
5. **Update docs** with new workflow
6. **Document the CORRECT cache building process** for future reference

---

## 📚 REFERENCES

- `scripts/convert_cache_to_mmap.py` - Shows train/dev were converted from NPZ
- `docs/02-data/cache-layout.md` lines 38-50 - Documents conversion workflow
- `src/brain_brr/data/datasets.py` lines 100-116 - Where the crash happens
- `deploy/modal/app.py` lines 183-213 - Shows Modal just copies from S3

---

**STATUS**: Investigation complete. Root cause identified. Solution recommended.

**NEXT**: Implement Solution A and test.
