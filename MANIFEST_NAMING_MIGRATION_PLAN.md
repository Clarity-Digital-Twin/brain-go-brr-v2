# Manifest Naming Migration Plan

**Version**: 1.0
**Date**: 2025-10-06
**Author**: AI Planning Agent
**Status**: 🟡 DRAFT - Awaiting External Validation

---

## Executive Summary

**OBJECTIVE**: Eliminate NPZ-style naming from manifests and codebase, aligning with NPY-only cache reality.

**IMPACT**:
- ✅ Cleaner code (remove 11 `.replace("_windows", "")` calls)
- ✅ Better maintainability (single source of truth for naming)
- ✅ Zero performance impact (same I/O patterns)
- ⚠️ Requires manifest regeneration (train: 61,616 entries, dev: 148,224 entries)

**RISK LEVEL**: 🟢 LOW
- Non-critical change (cosmetic/maintenance)
- Training can continue during preparation
- Full rollback capability
- Extensive pre-flight validation

---

## 1. Current State Audit

### 1.1 Cache Structure (Disk)

**Train Split** (`cache/tusz_mmap/train/`):
```
✅ 4667 *_data.npy files (actual cache, memory-mapped)
✅ 4667 *_labels.npy files (actual cache, memory-mapped)
⚠️ 3 *_windows.npz files (STRAY - legacy contamination, UNUSED)
✅ 1 manifest.json (61,616 entries with NPZ-style naming)
✅ 1 _dataset_index.json (metadata)
```

**Dev Split** (`cache/tusz_mmap/dev/`):
```
✅ 1832 *_data.npy files (actual cache, memory-mapped)
✅ 1832 *_labels.npy files (actual cache, memory-mapped)
✅ 0 *_windows.npz files (clean!)
✅ 1 manifest.json (148,224 entries with NPZ-style naming)
✅ 1 _dataset_index.json (metadata)
```

**Stray NPZ Files** (train only, DELETE these):
```
/home/jj/proj/brain-go-brr-v2/cache/tusz_mmap/train/aaaaaaac_s001_t000_windows.npz
/home/jj/proj/brain-go-brr-v2/cache/tusz_mmap/train/aaaaaaac_s001_t001_windows.npz
/home/jj/proj/brain-go-brr-v2/cache/tusz_mmap/train/aaaaaaac_s002_t000_windows.npz
```

### 1.2 Manifest Format (Current - NPZ-style)

**Train manifest** (`cache/tusz_mmap/train/manifest.json`):
```json
{
  "partial_seizure": [
    {"cache_file": "aaaaaaac_s001_t000_windows.npz", "window_idx": 0}
  ],
  "full_seizure": [
    {"cache_file": "aaaaaaac_s001_t000_windows.npz", "window_idx": 4}
  ]
}
```

**Problem**: References `*_windows.npz` but code loads `*_data.npy` + `*_labels.npy`

### 1.3 Code Locations Doing Name Conversion

**Total Conversions**: 11 locations across 3 files

#### `src/brain_brr/data/cache_utils.py` (4 locations)

**Line 45** - NPY path construction:
```python
return cache_dir / f"{edf_path.stem}_windows_data.npy"
```

**Line 95** - Strip `_windows` for NPY loading:
```python
stem = cache_path.stem.replace("_windows", "")
windows_file = cache_path.parent / f"{stem}_data.npy"
```

**Line 208** - Manifest uses NPZ-style naming:
```python
manifest_filename = f"{stem}_windows.npz"  # Keep NPZ-style naming for compatibility
```

**Line 289** - Convert NPY → NPZ style for verification:
```python
npy_set = {p.stem.replace("_data", "") + "_windows.npz" for p in npy_data_files}
```

#### `src/brain_brr/data/datasets.py` (6 locations)

**Line 107** - EEGWindowDataset constructs NPZ-style path:
```python
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"
```

**Line 275** - EEGWindowDataset (non-cached path):
```python
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"
```

**Line 390** - BalancedSeizureDataset strips `_windows`:
```python
stem = cache_path.stem.replace("_windows", "")
```

**Line 526** - BalancedSeizureDataset extracts file_id:
```python
file_id = cache_file.stem.replace("_windows", "")
```

**Line 607** - ValidationDataset strips `_windows`:
```python
stem = cache_path.stem.replace("_windows", "")
```

**Line 693** - ValidationDataset extracts file_id:
```python
file_id = cache_file.stem.replace("_windows", "")
```

#### `src/brain_brr/train/loop.py` (1 location)

**Line 629** - Constructs NPZ-style names for validation files:
```python
{f"{val_file.stem}_windows.npz" for val_file in val_files} if val_files else None
```

---

## 2. Target State Design

### 2.1 New Manifest Format (NPY-style)

**Train manifest** (proposed):
```json
{
  "partial_seizure": [
    {"cache_file": "aaaaaaac_s001_t000_data.npy", "window_idx": 0}
  ],
  "full_seizure": [
    {"cache_file": "aaaaaaac_s001_t000_data.npy", "window_idx": 4}
  ]
}
```

**Key Change**: `cache_file` references `*_data.npy` directly (reality, not legacy NPZ)

### 2.2 Code Simplification

#### `src/brain_brr/data/cache_utils.py`

**BEFORE** (line 95):
```python
stem = cache_path.stem.replace("_windows", "")  # Strip legacy suffix
windows_file = cache_path.parent / f"{stem}_data.npy"
```

**AFTER**:
```python
# Manifest already uses *_data.npy naming, no conversion needed
windows_file = cache_path
```

**BEFORE** (line 208):
```python
manifest_filename = f"{stem}_windows.npz"  # Keep NPZ-style naming for compatibility
```

**AFTER**:
```python
manifest_filename = f"{stem}_data.npy"  # Direct NPY naming
```

#### `src/brain_brr/data/datasets.py`

**BEFORE** (line 526, 693):
```python
file_id = cache_file.stem.replace("_windows", "")  # Strip suffix
```

**AFTER**:
```python
file_id = cache_file.stem.replace("_data", "")  # Extract base stem
```

**BEFORE** (line 390, 607):
```python
stem = cache_path.stem.replace("_windows", "")
windows_file = cache_path.parent / f"{stem}_data.npy"
```

**AFTER**:
```python
# cache_path already points to *_data.npy from manifest
windows_file = cache_path
```

#### `src/brain_brr/train/loop.py`

**BEFORE** (line 629):
```python
{f"{val_file.stem}_windows.npz" for val_file in val_files}
```

**AFTER**:
```python
{f"{val_file.stem}_data.npy" for val_file in val_files}
```

### 2.3 Cache Cleanup

**Delete stray NPZ files**:
```bash
rm -f cache/tusz_mmap/train/aaaaaaac_s001_t000_windows.npz
rm -f cache/tusz_mmap/train/aaaaaaac_s001_t001_windows.npz
rm -f cache/tusz_mmap/train/aaaaaaac_s002_t000_windows.npz
```

---

## 3. Migration Execution Plan

### ⚠️ CRITICAL SEQUENCING NOTE

**MUST UPDATE CODE BEFORE REGENERATING MANIFESTS!**

Why? `build_manifest()` is hardcoded to emit NPZ-style naming (line 208):
```python
manifest_filename = f"{stem}_windows.npz"  # Current behavior
```

If we rebuild BEFORE fixing code → generates NPZ-style manifests again (migration fails!)

**CORRECT ORDER**: Phase 1 → Phase 2 (code) → Phase 3 (manifests) → Phase 4 (cleanup)

---

### Phase 1: Pre-Flight Validation (30 min)

**Goal**: Verify current state before changes

**Steps**:
1. **Backup manifests**:
   ```bash
   cp cache/tusz_mmap/train/manifest.json cache/tusz_mmap/train/manifest.json.backup_npz
   cp cache/tusz_mmap/dev/manifest.json cache/tusz_mmap/dev/manifest.json.backup_npz
   ```

2. **Verify NPY cache integrity**:
   ```bash
   # Count data/label pairs (must be equal)
   ls cache/tusz_mmap/train/*_data.npy | wc -l    # Expect: 4667
   ls cache/tusz_mmap/train/*_labels.npy | wc -l  # Expect: 4667
   ls cache/tusz_mmap/dev/*_data.npy | wc -l      # Expect: 1832
   ls cache/tusz_mmap/dev/*_labels.npy | wc -l    # Expect: 1832
   ```

3. **Verify every *_data.npy has matching *_labels.npy**:
   ```bash
   # If this outputs anything, we have orphaned files
   for f in cache/tusz_mmap/train/*_data.npy; do
     labels="${f/_data.npy/_labels.npy}"
     [ ! -f "$labels" ] && echo "MISSING: $labels"
   done

   for f in cache/tusz_mmap/dev/*_data.npy; do
     labels="${f/_data.npy/_labels.npy}"
     [ ! -f "$labels" ] && echo "MISSING: $labels"
   done
   ```

4. **Verify current training still works**:
   ```bash
   # Smoke test with current NPZ-style manifests
   export BGB_SMOKE_TEST=1
   python -m src train configs/local/smoke.yaml
   ```

**VALIDATION GATE**: All checks must pass before proceeding

---

### Phase 2: Code Updates (60 min)

**Goal**: Update all 11 code locations to use NPY-style naming

**⚠️ CRITICAL**: Must complete BEFORE manifest regeneration!

#### Step 2.1: Update `src/brain_brr/data/cache_utils.py`

**Edit 1** - Line 45 (function `get_cache_path`):
```python
# OLD:
return cache_dir / f"{edf_path.stem}_windows_data.npy"

# NEW:
return cache_dir / f"{edf_path.stem}_data.npy"
```

**Edit 2** - Lines 92-96 (function `_load_mmap_cache`):
```python
# OLD:
if cache_path not in mmap_handles:
    # Convert NPZ path to NPY paths
    # Format: aaaaaajy_s001_t000_data.npy + aaaaaajy_s001_t000_labels.npy (mmap)
    stem = cache_path.stem.replace("_windows", "")
    windows_file = cache_path.parent / f"{stem}_data.npy"
    labels_file = cache_path.parent / f"{stem}_labels.npy"

# NEW:
if cache_path not in mmap_handles:
    # Manifest now uses *_data.npy naming directly
    # Format: aaaaaajy_s001_t000_data.npy + aaaaaajy_s001_t000_labels.npy (mmap)
    windows_file = cache_path  # cache_path IS *_data.npy from manifest
    stem = cache_path.stem.replace("_data", "")  # CRITICAL: Extract base for labels
    labels_file = cache_path.parent / f"{stem}_labels.npy"
```

**⚠️ LABELS PATH DERIVATION**: After migration, cache_path.stem = "file_data", so we MUST do `.replace("_data", "")` to get base stem "file" for constructing "file_labels.npy"

**Edit 3** - Lines 205-209 (function `build_manifest`):
```python
# OLD:
if npz_path.exists():
    stem = npz_path.stem
    manifest_filename = f"{stem}_windows.npz"  # Keep NPZ-style naming for compatibility

# NEW:
if npz_path.exists():
    stem = npz_path.stem
    manifest_filename = f"{stem}_data.npy"  # Direct NPY naming
```

**Edit 4** - Lines 287-289 (function `build_manifest`):
```python
# OLD:
# Verify all manifest files exist in cache
# NPZ format: filename_windows.npz
#   → Manifest references filename_windows.npz (for compatibility)
#   → Code loads filename_data.npy + filename_labels.npy (actual format)
npy_set = {p.stem.replace("_data", "") + "_windows.npz" for p in npy_data_files}

# NEW:
# Verify all manifest files exist in cache
# NPY format: filename_data.npy + filename_labels.npy
npy_set = {p.name for p in npy_data_files}  # Direct NPY naming
```

#### Step 2.2: Update `src/brain_brr/data/datasets.py`

**Edit 5** - Line 107 (EEGWindowDataset `__init__`):
```python
# OLD:
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

# NEW:
cache_path = self.cache_dir / f"{edf_path.stem}_data.npy"
```

**Edit 6** - Line 275 (EEGWindowDataset fallback):
```python
# OLD:
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

# NEW:
cache_path = self.cache_dir / f"{edf_path.stem}_data.npy"
```

**Edit 7** - Lines 389-391 (BalancedSeizureDataset `__getitem__`):
```python
# OLD:
# NPY format: convert a_windows stem → a_data.npy
stem = cache_path.stem.replace("_windows", "")
windows_mmap, labels_mmap = _load_mmap_cache(cache_path, self._mmap_handles)

# NEW:
# NPY format: manifest already uses *_data.npy naming
windows_mmap, labels_mmap = _load_mmap_cache(cache_path, self._mmap_handles)
```

**Edit 8** - Lines 525-526 (BalancedSeizureDataset `__getitem__`):
```python
# OLD:
# Extract file_id from cache filename (remove _windows suffix)
file_id = cache_file.stem.replace("_windows", "")

# NEW:
# Extract file_id from cache filename (remove _data suffix)
file_id = cache_file.stem.replace("_data", "")
```

**Edit 9** - Lines 606-608 (ValidationDataset `_build_index`):
```python
# OLD:
"""
Manifest references legacy NPZ naming (*_windows) but actual files
are NPY pairs (*_data.npy + *_labels.npy). Convert here.
"""
stem = cache_path.stem.replace("_windows", "")

# NEW:
# Manifest uses NPY naming directly (*_data.npy)
# Extract base stem for labels file lookup
stem = cache_path.stem.replace("_data", "")
```

**Edit 10** - Lines 692-693 (ValidationDataset `__getitem__`):
```python
# OLD:
# Extract file_id from cache filename (remove _windows suffix)
file_id = cache_file.stem.replace("_windows", "")

# NEW:
# Extract file_id from cache filename (remove _data suffix)
file_id = cache_file.stem.replace("_data", "")
```

#### Step 2.3: Update `src/brain_brr/train/loop.py`

**Edit 11** - Line 629:
```python
# OLD:
{f"{val_file.stem}_windows.npz" for val_file in val_files} if val_files else None

# NEW:
{f"{val_file.stem}_data.npy" for val_file in val_files} if val_files else None
```

---

### Phase 3: Manifest Regeneration (45 min)

**Goal**: Rebuild manifests with NPY-style naming using UPDATED code

**⚠️ PREREQUISITE**: Phase 2 (code updates) MUST be complete!

Now that `build_manifest()` emits NPY-style naming (line 208 fixed), we can rebuild:

```bash
# Force rebuild with NPY naming (code now emits *_data.npy)
export BGB_FORCE_MANIFEST_REBUILD=1
python -c "
from pathlib import Path
from src.brain_brr.data.cache_utils import build_manifest

# Train manifest
train_cache = Path('cache/tusz_mmap/train')
print('[TRAIN] Rebuilding manifest with NPY naming...')
build_manifest(
    cache_dir=train_cache,
    split_name='train',
    require_labels=True,
    force_rebuild=True
)

# Dev manifest
dev_cache = Path('cache/tusz_mmap/dev')
print('[DEV] Rebuilding manifest with NPY naming...')
build_manifest(
    cache_dir=dev_cache,
    split_name='dev',
    require_labels=True,
    force_rebuild=True
)
print('[SUCCESS] Manifests regenerated with NPY naming!')
"
```

**Verification**:
```bash
# Check manifest format (should show *_data.npy)
head -20 cache/tusz_mmap/train/manifest.json
```

**Expected Output**:
```json
{
  "partial_seizure": [
    {"cache_file": "aaaaaaac_s001_t000_data.npy", "window_idx": 0}
  ],
  ...
}
```

---

### Phase 4: Cache Cleanup (5 min)

**Goal**: Remove NPZ files (AFTER manifest conversion!)

**⚠️ CRITICAL TIMING**: Must delete AFTER Phase 3 (manifest regen)!

Why? These 3 NPZ files are currently referenced in old manifests:
- `aaaaaaac_s001_t000_windows.npz` (64 references)
- `aaaaaaac_s001_t001_windows.npz`
- `aaaaaaac_s002_t000_windows.npz`

Deleting before manifest update = rollback will fail!

```bash
# Verify files before deletion
ls -lh cache/tusz_mmap/train/*_windows.npz

# Delete (train only - dev has none)
rm -f cache/tusz_mmap/train/aaaaaaac_s001_t000_windows.npz
rm -f cache/tusz_mmap/train/aaaaaaac_s001_t001_windows.npz
rm -f cache/tusz_mmap/train/aaaaaaac_s002_t000_windows.npz

# Verify deletion
ls cache/tusz_mmap/train/*_windows.npz 2>/dev/null || echo "✓ All NPZ files removed"
```

### Phase 5: Validation Testing (45 min)

**Goal**: Verify migration succeeded

#### Test 1: Smoke Test (Local)
```bash
export BGB_SMOKE_TEST=1
python -m src train configs/local/smoke.yaml
```

**Expected**:
- ✅ Loads 3 files successfully
- ✅ BalancedSeizureDataset uses manifest (instant load)
- ✅ ValidationDataset uses manifest (instant load)
- ✅ No warnings about missing files
- ✅ Training completes 1 epoch

#### Test 2: Full Test Suite
```bash
make test
```

**Expected**:
- ✅ All 104+ tests pass
- ✅ Dataset tests verify NPY loading
- ✅ Manifest tests verify new format

#### Test 3: Manifest Integrity
```python
import json
from pathlib import Path

def verify_manifest(manifest_path: Path, cache_dir: Path) -> None:
    """Verify all manifest entries point to real NPY files."""
    with open(manifest_path) as f:
        data = json.load(f)

    errors = []
    for category in ["partial_seizure", "full_seizure"]:
        for entry in data.get(category, []):
            cache_file = entry["cache_file"]

            # Must be *_data.npy format
            if not cache_file.endswith("_data.npy"):
                errors.append(f"Wrong format: {cache_file}")

            # Must exist on disk
            full_path = cache_dir / cache_file
            if not full_path.exists():
                errors.append(f"Missing: {full_path}")

            # Must have matching labels file
            labels_path = full_path.parent / full_path.name.replace("_data.npy", "_labels.npy")
            if not labels_path.exists():
                errors.append(f"Missing labels: {labels_path}")

    if errors:
        print(f"[✗] {manifest_path}: {len(errors)} errors")
        for e in errors[:10]:  # Show first 10
            print(f"  - {e}")
    else:
        print(f"[✓] {manifest_path}: All entries valid")

# Verify both splits
verify_manifest(
    Path("cache/tusz_mmap/train/manifest.json"),
    Path("cache/tusz_mmap/train")
)
verify_manifest(
    Path("cache/tusz_mmap/dev/manifest.json"),
    Path("cache/tusz_mmap/dev")
)
```

### Phase 6: Documentation & Release (30 min)

#### Update Documentation

**File**: `docs/09-technical-debt/active-debt.md`
- Mark P3-1 as RESOLVED
- Add resolution details

**File**: `CHANGELOG.md`
- Add v3.8.3 entry documenting manifest migration

**File**: `src/brain_brr/__init__.py`
- Bump version to 3.8.3
- Update docstring

#### Create Release
```bash
# Quality check
make q

# Run full tests
make test

# Commit changes
git add -A
git commit -m "$(cat <<'EOF'
fix(manifests): Migrate from NPZ-style to NPY-style naming

BREAKING: Manifests now use *_data.npy naming (matches cache reality)

Changes:
- Regenerated manifests with NPY naming (train: 61,616, dev: 148,224)
- Removed 11 .replace("_windows", "") conversions
- Deleted 3 stray NPZ files from train cache
- Simplified cache_utils.py and datasets.py

Impact:
- Cleaner code, single source of truth
- Zero performance change (same I/O)
- Better maintainability

Resolves: P3-1 (Manifest naming mismatch)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# Tag release
git tag v3.8.3-clean-manifest-naming

# Push
git push origin development --tags
```

---

## 4. Risk Analysis & Mitigation

### Risk 1: Manifest Regeneration Errors

**Probability**: LOW
**Impact**: HIGH (training fails to start)

**Mitigation**:
- ✅ Backup original manifests first
- ✅ Validate NPY cache integrity before rebuild
- ✅ Use proven `build_manifest()` function (already tested)
- ✅ Verify manifest integrity after rebuild
- ✅ Test with smoke test before full deployment

**Rollback**:
```bash
# Restore backups
cp cache/tusz_mmap/train/manifest.json.backup_npz cache/tusz_mmap/train/manifest.json
cp cache/tusz_mmap/dev/manifest.json.backup_npz cache/tusz_mmap/dev/manifest.json

# Revert code changes
git checkout HEAD~1 -- src/brain_brr/data/cache_utils.py
git checkout HEAD~1 -- src/brain_brr/data/datasets.py
git checkout HEAD~1 -- src/brain_brr/train/loop.py
```

### Risk 2: Missed Code Locations

**Probability**: LOW
**Impact**: MEDIUM (runtime errors during training)

**Mitigation**:
- ✅ Comprehensive grep audit (completed above)
- ✅ Full test suite validation
- ✅ Smoke test validates end-to-end flow
- ✅ Type checking (mypy) catches path issues

**Detection**:
- Test suite will fail if any location missed
- Smoke test will crash if paths wrong

### Risk 3: NPY Cache Corruption

**Probability**: VERY LOW
**Impact**: HIGH (data loss)

**Mitigation**:
- ✅ NOT touching cache files (only manifests)
- ✅ Pre-flight validation checks NPY integrity
- ✅ Manifests only reference existing files

**Safety**: Cache files are read-only during migration

### Risk 4: Breaking Modal Training

**Probability**: LOW
**Impact**: HIGH (production disruption)

**Mitigation**:
- ✅ Wait for current v3.8.2 training to complete
- ✅ Test locally first (phases 1-5)
- ✅ Deploy to Modal only after local validation
- ✅ Modal cache regeneration may be needed

**Modal-Specific Steps**:
```bash
# After local validation passes
# Deploy new code to Modal
git push origin development

# Regenerate Modal cache manifests
modal run deploy/modal/app.py --action rebuild-manifests

# Smoke test on Modal
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

---

## 5. Validation Checklist

### Pre-Flight (Before Any Changes)

- [ ] Backup train manifest: `cache/tusz_mmap/train/manifest.json.backup_npz`
- [ ] Backup dev manifest: `cache/tusz_mmap/dev/manifest.json.backup_npz`
- [ ] Verify NPY cache integrity (4667 train, 1832 dev pairs)
- [ ] Verify current smoke test passes with NPZ-style manifests
- [ ] Identify all 3 stray NPZ files for deletion

### Post-Migration (After Changes)

- [ ] New manifests use `*_data.npy` naming (verified via grep)
- [ ] All 11 code locations updated (verified via diff)
- [ ] All 3 stray NPZ files deleted (verified via ls)
- [ ] Manifest integrity check passes (all files exist)
- [ ] Smoke test passes with NPY-style manifests
- [ ] Full test suite passes (`make test`)
- [ ] Quality checks pass (`make q`)
- [ ] Documentation updated (CHANGELOG, tech debt, version)

### Modal Deployment (After Local Validation)

- [ ] Code pushed to GitHub
- [ ] Modal manifests regenerated
- [ ] Modal smoke test passes (50 files)
- [ ] Modal full training starts successfully
- [ ] First epoch completes without errors

---

## 6. Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Pre-flight validation | 30 min | 0:30 |
| 2. Code updates (11 locations) | 60 min | 1:30 |
| 3. Manifest regeneration | 45 min | 2:15 |
| 4. Cache cleanup | 5 min | 2:20 |
| 5. Validation testing | 45 min | 3:05 |
| 6. Documentation & release | 30 min | 3:35 |

**Total**: ~3.5 hours (single developer, focused work)

**⚠️ SEQUENCING**: Code updates (Phase 2) MUST complete before manifest regen (Phase 3)

---

## 7. Success Criteria

### Must Have (Blockers)
- ✅ All manifests use `*_data.npy` naming
- ✅ All code uses NPY naming (zero `.replace("_windows", "")` calls)
- ✅ All tests pass (local + CI)
- ✅ Smoke test passes end-to-end
- ✅ Zero stray NPZ files remain

### Should Have (Quality)
- ✅ Full test coverage maintained (>80%)
- ✅ Type checking passes (mypy)
- ✅ Code quality passes (ruff)
- ✅ Documentation updated

### Nice to Have (Polish)
- ✅ Clean git history (single commit)
- ✅ Release tagged (v3.8.3)
- ✅ Modal deployment validated

---

## 8. Rollback Plan

### If Validation Fails

**Scenario**: Smoke test fails after migration

**Steps**:
1. Restore manifest backups:
   ```bash
   cp cache/tusz_mmap/train/manifest.json.backup_npz cache/tusz_mmap/train/manifest.json
   cp cache/tusz_mmap/dev/manifest.json.backup_npz cache/tusz_mmap/dev/manifest.json
   ```

2. Revert code changes:
   ```bash
   git checkout HEAD~1 -- src/brain_brr/data/cache_utils.py
   git checkout HEAD~1 -- src/brain_brr/data/datasets.py
   git checkout HEAD~1 -- src/brain_brr/train/loop.py
   ```

3. Verify rollback:
   ```bash
   export BGB_SMOKE_TEST=1
   python -m src train configs/local/smoke.yaml
   ```

**Time to Rollback**: <5 minutes

### If Modal Training Breaks

**Scenario**: Modal training fails after deployment

**Steps**:
1. Revert GitHub commit:
   ```bash
   git revert HEAD
   git push origin development
   ```

2. Modal will auto-pull reverted code
3. Restart training with old code

**Time to Rollback**: <10 minutes

---

## 9. External Validation Questions

**For Review by External AI Agent / Human Developer**:

### Completeness
1. Did we identify ALL code locations doing NPZ→NPY conversion?
2. Are there any edge cases in the migration logic?
3. Should we check test files for hardcoded NPZ paths?

### Correctness
4. Is the new manifest format correct (`*_data.npy` entries)?
5. Are the code edits functionally equivalent?
6. Will this break any integration tests?

### Risk Management
7. Is the rollback plan sufficient?
8. Are there additional validation steps needed?
9. Should we stage this as v3.8.3 or v3.9.0 (minor bump)?

### Performance
10. Will manifest regeneration impact cache hot paths?
11. Should we benchmark before/after?

### Documentation
12. Is the migration plan clear enough for execution?
13. Are there missing edge cases in the plan?

---

## 10. Next Steps

### Immediate (Awaiting Approval)
1. **External validation**: Review this plan with another AI agent
2. **Address feedback**: Incorporate review comments
3. **Get user approval**: Confirm migration timing

### Execution (After Approval)
1. **Wait for Modal training completion**: Don't interrupt v3.8.2 run
2. **Execute Phase 1-6**: Follow plan sequentially
3. **Deploy to Modal**: After local validation passes

### Post-Migration
1. **Monitor first full training**: Ensure no degradation
2. **Update onboarding docs**: Reflect new naming convention
3. **Close tech debt issue**: Mark P3-1 as RESOLVED

---

## Appendix A: Quick Reference Commands

### Verification Commands
```bash
# Check current manifest format
head -20 cache/tusz_mmap/train/manifest.json

# Count NPY pairs
ls cache/tusz_mmap/train/*_data.npy | wc -l
ls cache/tusz_mmap/dev/*_data.npy | wc -l

# Find NPZ files
find cache/tusz_mmap -name "*_windows.npz"

# Verify code patterns
grep -r "_windows" src/brain_brr/data/ src/brain_brr/train/
```

### Migration Commands
```bash
# Backup manifests
cp cache/tusz_mmap/train/manifest.json{,.backup_npz}
cp cache/tusz_mmap/dev/manifest.json{,.backup_npz}

# Rebuild manifests (after code changes)
export BGB_FORCE_MANIFEST_REBUILD=1
python -c "from pathlib import Path; from src.brain_brr.data.cache_utils import build_manifest; \
  build_manifest(Path('cache/tusz_mmap/train'), 'train', True, True); \
  build_manifest(Path('cache/tusz_mmap/dev'), 'dev', True, True)"

# Smoke test
export BGB_SMOKE_TEST=1
python -m src train configs/local/smoke.yaml
```

### Rollback Commands
```bash
# Restore manifests
cp cache/tusz_mmap/train/manifest.json{.backup_npz,}
cp cache/tusz_mmap/dev/manifest.json{.backup_npz,}

# Revert code
git checkout HEAD~1 -- src/brain_brr/data/cache_utils.py
git checkout HEAD~1 -- src/brain_brr/data/datasets.py
git checkout HEAD~1 -- src/brain_brr/train/loop.py
```

---

**END OF PLAN**

**APPROVAL REQUIRED**: This plan must be validated by external AI agent before execution.

**Questions? Contact**: Project maintainer or escalate to technical lead.
