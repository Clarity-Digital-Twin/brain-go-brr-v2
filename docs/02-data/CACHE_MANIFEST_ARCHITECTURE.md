# Cache & Manifest Architecture - Complete Documentation

**Status**: 2025-09-30 - Documenting current state before fixes
**Author**: Investigation after discovering local/Modal divergence
**Purpose**: Understand WTF is happening with manifests, indexes, and caches

---

## 🚨 CRITICAL FINDINGS - Current State

### Local Cache State (cache/tusz/)

```
cache/tusz/
├── .cache_metadata.json          (282 bytes, Sep 26)
├── train/
│   ├── manifest.json              (27 MB, Sep 27) ✅ EXISTS
│   ├── _dataset_index.json        (282 bytes, Sep 28) ⚠️ STALE (3 files from smoke test)
│   └── *.npz                      (4667 files, 306 GB)
└── dev/
    ├── manifest.json              (13 MB, Sep 29) ✅ EXISTS
    ├── _dataset_index.json        (148 KB, Sep 30) ✅ COMPLETE (1832 files)
    └── *.npz                      (1832 files, 143 GB)
```

### Modal Cache State (/results/cache/tusz/)

**Unknown** - Need to verify what's actually on Modal SSD volume:
- Are manifests there?
- Are indexes there?
- When were they last updated?

---

## 📋 File Types & Purposes

### 1. `manifest.json` (TRAIN ONLY - Used by BalancedSeizureDataset)

**Purpose**: Enables balanced sampling for training
**Created by**: `src/brain_brr/data/cache_utils.py::scan_existing_cache()`
**Used by**: `BalancedSeizureDataset.__init__()` (line 242-257 in datasets.py)

**Structure** (27 MB for 4667 files):
```json
{
  "partial_seizure": [
    {"cache_file": "path/to/file_windows.npz", "indices": [0, 5, 12, ...]},
    ...
  ],
  "full_seizure": [...],
  "no_seizure": [...]
}
```

**Key Facts**:
- ✅ Lists ALL window indices categorized by seizure type
- ✅ Allows oversampling partial/full seizures without loading NPZ files
- ✅ ONLY used for TRAINING (not validation)
- ⚠️ Large file (27 MB) - takes time to generate

**Code Reference**:
```python
# src/brain_brr/data/datasets.py:242-257
class BalancedSeizureDataset(Dataset):
    def __init__(self, cache_dir: Path, ...):
        manifest_path = self.cache_dir / "manifest.json"
        if ensure_manifest and not manifest_path.exists():
            _ = scan_existing_cache(self.cache_dir)  # Generate it

        with manifest_path.open() as f:
            manifest = json.load(f)
```

---

### 2. `_dataset_index.json` (BOTH SPLITS - Used by EEGWindowDataset)

**Purpose**: Fast startup - avoids opening NPZ files to count windows
**Created by**: `EEGWindowDataset.__init__()` (line 53-78 in datasets.py)
**Used by**: `EEGWindowDataset` (validation dataset)

**Structure** (282 bytes for 3 files, 148 KB for 1832 files):
```json
{
  "files": [
    "data_ext4/tusz/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf",
    "data_ext4/tusz/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t001.edf",
    ...
  ],
  "window_counts": [25, 18, 21, ...]
}
```

**Key Facts**:
- ✅ Stores window counts per EDF file (not per-window metadata)
- ✅ Allows computing cumulative offsets without loading NPZ
- ✅ Used for BOTH training and validation datasets
- ⚠️ MUST match the exact file list passed to `EEGWindowDataset.__init__()`
- ⚠️ Invalidated when file list changes (smoke test → full training)

**Code Reference**:
```python
# src/brain_brr/data/datasets.py:53-78
class EEGWindowDataset:
    def __init__(self, edf_files: list[Path], cache_dir: Path, ...):
        index_cache_path = self.cache_dir / "_dataset_index.json"
        if index_cache_path.exists():
            try:
                with open(index_cache_path) as f:
                    cached = json.load(f)
                    # Validate cache matches current file list
                    if cached["files"] == [str(p) for p in self.edf_files]:
                        self.window_counts = cached["window_counts"]
                        # ... (use cached counts)
            except Exception as e:
                logger.warning(f"Could not load cached index: {e}")

        # If cache miss or invalid, rebuild index by scanning NPZ files
        logger.info(f"[DATA] Building dataset index for {len(self.edf_files)} files...")
```

---

### 3. `.cache_metadata.json` (ROOT - Cache validation)

**Purpose**: Proves cache was built with correct split policy
**Created by**: Cache build script / populate_cache()
**Used by**: Validation checks in training scripts

**Structure**:
```json
{
  "split_policy": "official_tusz",
  "created": "2025-09-26T22:11:00",
  "timestamp": "1758939060",
  "note": "Cache built with patient-disjoint TUSZ official splits",
  "train_patients": 579,
  "dev_patients": 53,
  "train_files": 4667,
  "dev_files": 1832,
  "version": "v3.2.0"
}
```

---

## 🔄 Dataset Types & What They Use

### Training Dataset: `BalancedSeizureDataset`

**Files Used**:
- ✅ `train/manifest.json` (REQUIRED)
- ❌ `train/_dataset_index.json` (NOT USED)

**Behavior**:
1. Loads `manifest.json` at init
2. Samples windows with balancing:
   - 100% of partial seizure windows (16,215)
   - 30% of full seizure windows (4,864)
   - 250% of partial seizure = 40,537 no-seizure windows
   - Total: ~61,616 windows (vs 1.3M unbalanced)
3. Each `__getitem__()` loads from NPZ based on manifest index

**Config**: `use_balanced_sampling: true` (CRITICAL for training)

---

### Validation Dataset: `EEGWindowDataset`

**Files Used**:
- ✅ `dev/_dataset_index.json` (OPTIONAL - speeds up init)
- ❌ `dev/manifest.json` (NOT USED)

**Behavior**:
1. Tries to load `_dataset_index.json` for fast init
2. If cache miss or file list changed: rebuilds index by scanning NPZ files
3. Uses natural distribution (~8% seizures, ~92% background)
4. Each `__getitem__()` loads from NPZ based on cumulative offset

**Config**: `use_balanced_sampling: false` (for smoke tests) or ignored (validation uses `EEGWindowDataset` regardless)

---

## 🐛 Current Bugs & Inconsistencies

### Bug #1: Stale `train/_dataset_index.json`

**Problem**:
- File created Sep 28 during smoke test (3 files)
- Current training needs 4667 files
- Dataset code correctly detects mismatch and rebuilds
- **Result**: Training processes 4667 files to rebuild index ✅ CORRECT BEHAVIOR

**Fix Needed**: None - working as designed

---

### Bug #2: Modal `populate_cache()` Missing Manifests

**Problem**:
```python
# deploy/modal/app.py:197-227
def populate_cache():
    # Copies *.npz files ✅
    shutil.copytree(train_src, train_dst)

    # Copies .cache_metadata.json ✅
    shutil.copy2(metadata_src, metadata_dst)

    # DOES NOT COPY manifest.json ❌
    # DOES NOT COPY _dataset_index.json ❌
```

**Impact**:
- Modal training rebuilds `train/manifest.json` on first run (slow! ~5-10 min)
- Modal training rebuilds `dev/_dataset_index.json` on first run (slower! ~40 min)
- Subsequent runs use cached versions ✅

**Questions**:
1. Are manifests/indexes ALREADY on Modal SSD from manual transfers?
2. Should `populate_cache()` copy them?
3. Or should Modal regenerate them (safer - ensures consistency)?

---

### Bug #3: Train Dataset Uses Manifest, But Also Has Index?

**Confusion**:
- `BalancedSeizureDataset` uses `manifest.json` ✅
- But `train/_dataset_index.json` exists (282 bytes, stale)
- Is train index used anywhere? **NO**

**Explanation**:
- Smoke test with `use_balanced_sampling: false` created train index
- Full training with `use_balanced_sampling: true` uses manifest instead
- Train index is **UNUSED and can be deleted**

---

## 📊 File Size Reference

| File | Size | Generation Time | Used By |
|------|------|----------------|---------|
| `train/*.npz` (4667 files) | 306 GB | Hours (one-time) | Both datasets |
| `dev/*.npz` (1832 files) | 143 GB | Hours (one-time) | Validation |
| `train/manifest.json` | 27 MB | ~5-10 min | `BalancedSeizureDataset` |
| `dev/manifest.json` | 13 MB | ~3-5 min | **UNUSED** (validation uses `EEGWindowDataset`) |
| `train/_dataset_index.json` | 282 B - 1 MB | <1 min | **UNUSED** (training uses manifest) |
| `dev/_dataset_index.json` | 148 KB | ~40 min (1832 NPZ opens) | `EEGWindowDataset` |
| `.cache_metadata.json` | 282 B | <1 sec | Validation |

---

## 🔍 What Modal Needs

### Minimum Required Files (Modal SSD: /results/cache/tusz/)

```
/results/cache/tusz/
├── .cache_metadata.json          ✅ COPIED by populate_cache()
├── train/
│   ├── manifest.json              ❌ NOT COPIED (regenerated on first run)
│   └── *.npz (4667 files)         ✅ COPIED by populate_cache()
└── dev/
    ├── _dataset_index.json        ❌ NOT COPIED (regenerated on first run)
    └── *.npz (1832 files)         ✅ COPIED by populate_cache()
```

### Optional But Helpful

- `dev/manifest.json` - **NOT USED** (validation doesn't use balanced sampling)
- `train/_dataset_index.json` - **NOT USED** (training uses manifest)

---

## 🎯 Recommended Actions (NOT IMPLEMENTED YET)

### Option A: Copy Manifests from Local → S3 → Modal (Fastest)

**Pros**:
- Saves 5-10 min (train manifest) + 40 min (dev index) on first Modal run
- Ensures consistency with local training

**Cons**:
- Manual sync required when cache rebuilds
- Risk of stale manifests if NPZ files change

**Steps**:
```bash
# 1. Local: Upload manifests to S3
aws s3 cp cache/tusz/train/manifest.json s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/
aws s3 cp cache/tusz/dev/_dataset_index.json s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/

# 2. Modal: Update populate_cache() to copy manifests
# (see code changes below)

# 3. Modal: Re-run populate_cache
modal run deploy/modal/app.py --action populate_cache
```

---

### Option B: Let Modal Regenerate (Safer, Slower)

**Pros**:
- Guaranteed consistency (manifests match NPZ files)
- No manual sync needed

**Cons**:
- First run takes extra 45-50 minutes
- Subsequent runs are fast (manifests cached on SSD)

**Steps**:
- Do nothing - current behavior works correctly

---

## 🔧 Code Changes Needed (Option A)

### 1. Update `populate_cache()` to Copy Manifests

```python
# deploy/modal/app.py:230+ (after metadata copy)

# Copy train manifest (for BalancedSeizureDataset)
train_manifest_src = src / "train" / "manifest.json"
train_manifest_dst = dst / "train" / "manifest.json"
if train_manifest_src.exists():
    logger.info(f"[COPY] Copying train manifest...")
    shutil.copy2(train_manifest_src, train_manifest_dst)
    logger.info(f"[COPY] ✅ Copied train manifest")
else:
    logger.info(f"[WARNING] No train manifest found at {train_manifest_src}")
    logger.info(f"[WARNING] Will regenerate on first training run (~5-10 min)")

# Copy dev index (for EEGWindowDataset - speeds up init)
dev_index_src = src / "dev" / "_dataset_index.json"
dev_index_dst = dst / "dev" / "_dataset_index.json"
if dev_index_src.exists():
    logger.info(f"[COPY] Copying dev dataset index...")
    shutil.copy2(dev_index_src, dev_index_dst)
    logger.info(f"[COPY] ✅ Copied dev dataset index")
else:
    logger.info(f"[WARNING] No dev index found at {dev_index_src}")
    logger.info(f"[WARNING] Will regenerate on first training run (~40 min)")
```

---

## 📚 Documentation Needed

### Files to Update

1. **`docs/02-data/cache-building.md`** (or create)
   - Explain manifest vs index
   - Document generation times
   - Show validation commands

2. **`docs/05-training/local.md`**
   - Add section on manifest/index regeneration triggers
   - Warn about stale indexes after smoke tests

3. **`docs/05-training/modal.md`**
   - Document `populate_cache()` behavior
   - Add troubleshooting for missing manifests
   - Show how to verify cache completeness

4. **`CLAUDE.md`** (root)
   - Add cache files to "Critical Notes" section
   - Document expected file sizes
   - Link to detailed docs

5. **`README.md`** (potentially)
   - Quick reference for cache structure

---

## 🧪 Verification Commands

### Local: Check Cache Completeness

```bash
# Check all cache files exist
ls -lh cache/tusz/.cache_metadata.json
ls -lh cache/tusz/train/manifest.json
ls -lh cache/tusz/dev/_dataset_index.json

# Check file counts
echo "Train NPZ:" && ls -1 cache/tusz/train/*.npz | wc -l  # Should be 4667
echo "Dev NPZ:" && ls -1 cache/tusz/dev/*.npz | wc -l      # Should be 1832

# Validate manifest structure
head -100 cache/tusz/train/manifest.json | python3 -m json.tool

# Validate index structure
cat cache/tusz/dev/_dataset_index.json | python3 -m json.tool | head -50
```

### Modal: Check Cache Completeness (via app.py)

```python
@app.function(volumes={"/results": results_volume})
def check_cache():
    """Verify Modal SSD cache completeness."""
    from pathlib import Path

    cache = Path("/results/cache/tusz")

    print(f"✅ Metadata: {(cache / '.cache_metadata.json').exists()}")
    print(f"✅ Train manifest: {(cache / 'train' / 'manifest.json').exists()}")
    print(f"✅ Dev index: {(cache / 'dev' / '_dataset_index.json').exists()}")

    train_npz = list((cache / "train").glob("*.npz"))
    dev_npz = list((cache / "dev").glob("*.npz"))

    print(f"Train NPZ: {len(train_npz)} (expected 4667)")
    print(f"Dev NPZ: {len(dev_npz)} (expected 1832)")
```

Usage: `modal run deploy/modal/app.py::check_cache`

---

## 🚨 Critical Questions to Answer

1. **Are manifests/indexes ALREADY on Modal SSD?**
   - Need to run `check_cache()` to verify
   - If yes: were they manually transferred? When?
   - If no: first training run will regenerate (45-50 min delay)

2. **Should we sync local manifests to S3?**
   - Pro: Faster Modal startup
   - Con: Manual process, risk of staleness

3. **Should populate_cache() copy manifests?**
   - Pro: Automated, consistent
   - Con: Requires S3 sync first

4. **Is dev/manifest.json used anywhere?**
   - **Answer: NO** - validation uses `EEGWindowDataset`, not `BalancedSeizureDataset`
   - Can be deleted or ignored

5. **When does train/_dataset_index.json get used?**
   - **Answer: NEVER** - training uses `BalancedSeizureDataset` with manifest
   - Only created during smoke tests with `use_balanced_sampling: false`

---

## 📝 Next Steps (DO NOT EXECUTE YET)

1. **Verify Modal cache state**
   - Add `check_cache()` function to app.py
   - Run and document results

2. **Decide on sync strategy**
   - Option A: Copy manifests (faster, manual)
   - Option B: Regenerate (slower, automatic)

3. **Update documentation**
   - Create/update all docs listed above
   - Add troubleshooting guides

4. **Implement chosen solution**
   - Update code if needed
   - Test on Modal
   - Verify training starts correctly

5. **Add validation to CI/CD**
   - Check manifest exists before training
   - Warn if index is stale
   - Validate file counts match metadata

---

## 🔗 Related Files

- `src/brain_brr/data/datasets.py` - Dataset implementations
- `src/brain_brr/data/cache_utils.py` - Manifest generation
- `deploy/modal/app.py` - Modal cache population
- `configs/local/train.yaml` - Local training config
- `configs/modal/train.yaml` - Modal training config

---

**End of Documentation - Ready for Review**
