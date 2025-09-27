# v3.1.1 - Critical Data Integrity Fix

## 🚨 CRITICAL: Cache Rebuild Required

**ALL training caches built before v3.1.1 have 44 missing seizures and must be rebuilt.**

## What's Fixed

### 🧠 Data Integrity (CRITICAL)
- **Fixed missing seizures**: Added `mysz` (myoclonic) seizure type that was causing 44 seizures to be mislabeled as background
- **Fixed EEG outliers**: Clipping to ±10σ prevents numerical overflow from extreme values (up to 121σ found in data)
- **Fixed NaN crashes**: Complete 3-tier clamping system prevents non-finite values during training

### 📁 Naming Consistency
- **Standardized on 'dev'**: All code/configs/docs now use 'dev' (not 'val') to match TUSZ official naming
- **20+ files updated**: Complete consistency across entire codebase
- **Clear documentation**: Added `CRITICAL-NAMING-CONVENTION.md` explaining why

### 🛠️ CLI Improvements
- Fixed evaluate command checkpoint handling
- Added `--limit-files` option to build-cache
- Fixed CSV export stride-aware timing
- Better error messages and handling

### ⚡ Performance
- Adjusted test thresholds for V3 dual-stream architecture (~50ms inference expected)

## Breaking Changes

**Cache rebuild is MANDATORY** - old caches have missing seizures:

```bash
# 1. Remove old cache
rm -rf cache/tusz

# 2. Rebuild with ALL seizures properly labeled
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train --split train
python -m src build-cache --data-dir data_ext4/tusz/edf/dev --cache-dir cache/tusz/dev --split dev

# 3. Upload to S3
./scripts/upload_cache_to_s3.sh

# 4. Populate Modal
modal run deploy/modal/app.py --action populate-cache
```

## Quick Validation

After rebuild, verify:
- ✅ 4667 train + 1832 dev NPZ files
- ✅ Manifest shows seizure windows present
- ✅ All paths use 'dev' naming
- ✅ Training runs without NaN losses

## Full Changelog

### Added
- `mysz` seizure type to label set
- `--limit-files` option for build-cache
- Comprehensive S3/Modal upload documentation
- P0-P3 blockers audit document

### Changed
- All 'val' references → 'dev' throughout codebase
- Performance test thresholds for V3 architecture
- Enhanced error handling in CLI commands

### Fixed
- 44 missing myoclonic seizures now properly labeled
- EEG outlier clipping prevents overflow
- Output sanitization prevents non-finite logits
- CLI evaluate checkpoint config handling
- CSV export stride-aware timing

### Documentation
- `CRITICAL-NAMING-CONVENTION.md` - Why we use 'dev'
- `S3-MODAL-CACHE-UPLOAD-PROCEDURE.md` - Complete upload guide
- `P0-P3-BLOCKERS.md` - All known issues tracked
- Updated configs, README, and all docs for consistency

## Commits Since v3.1.0

28 commits addressing critical issues:
- Data integrity fixes (mysz seizures, outliers, NaN prevention)
- Complete dev/val → dev naming standardization
- CLI improvements and bug fixes
- Documentation and clarity updates

## Installation

```bash
git fetch
git checkout v3.1.1
make setup && make setup-gpu
```

## Training Recommendation

Enable gradient sanitization for extra stability:
```bash
export BGB_SANITIZE_GRADS=1
python -m src train configs/local/train.yaml
```

---

**Priority: CRITICAL** - Rebuild your cache immediately to include all seizures.

**Impact**: Training on old cache will miss 44 seizures, reducing model sensitivity.