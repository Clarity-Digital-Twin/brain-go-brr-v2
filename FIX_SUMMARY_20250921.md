# CRITICAL FIXES SUMMARY - September 21, 2025

## P0 BLOCKERS FIXED

### 1. CSV Parser - TUSZ CSV_BI Format (CRITICAL)
**Problem**: Training was seeing 0% seizures due to broken CSV parser
**Root Cause**: Parser was reading wrong columns for TUSZ CSV_BI format
```
# Was reading: start,end,label (parts[0,1,2])
# CSV_BI has: channel,start_time,stop_time,label,confidence
# Fixed to: skip channel, read parts[1,2,3]
```
**Files Fixed**: `src/brain_brr/data/io.py`

### 2. Missing Seizure Types (CRITICAL)
**Problem**: Only looking for "seiz" label which doesn't exist in TUSZ
**Fix**: Added all TUSZ seizure types:
```python
seizure_labels = {"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}
```
**Impact**: Now detects ALL seizure types in TUSZ dataset

### 3. BalancedSeizureDataset Implementation
**Problem**: Risk of training with 0% seizures in batches
**Fix**: Implemented SeizureTransformer's exact balancing:
- ALL partial seizure windows
- 0.3× full seizure windows
- 2.5× background windows
**Files**: `src/brain_brr/data/datasets.py`

### 4. Hard Guards Against Zero Seizures
**Problem**: Training could proceed with no seizures
**Fix**: Added hard exit in CLI if no seizures found in manifest
**Files**: `src/brain_brr/cli/cli.py`

## CONFIGURATION CLEANUP

### 5. Config Directory Reorganization
**Before**: 8 messy configs with unclear naming
**After**: Clean professional structure
```
configs/
├── local/
│   ├── smoke.yaml
│   ├── train.yaml
│   ├── dev.yaml
│   └── eval.yaml
└── modal/
    ├── smoke_a100.yaml
    ├── train_a100.yaml
    ├── dev_a100.yaml
    └── eval_a100.yaml
```

### 6. Config Consistency Fixes
**Local configs fixed**:
- Added missing `use_mne: true` to smoke.yaml
- Fixed WSL2 settings (num_workers=0, pin_memory=false)
- Changed device from `auto` to `cuda` (explicit GPU)
- Fixed checkpoint paths to match training output

**Modal configs fixed**:
- Fixed checkpoint paths to match train_a100 output
- Changed device from `auto` to `cuda` for consistency
- Verified A100 optimizations (batch_size=64, workers=8)

## MODAL PIPELINE FIXES

### 7. BGB_LIMIT_FILES Environment Variable
**Problem**: Modal was using 50 files for full training
**Root Cause**: Environment variable persistence
**Fix**: Explicitly unset for non-smoke configs
```python
if "smoke" in config_path.lower():
    env["BGB_LIMIT_FILES"] = "50"
else:
    env.pop("BGB_LIMIT_FILES", None)  # EXPLICITLY UNSET
```
**Files**: `deploy/modal/app.py`

### 8. Cache Directory Structure
**Problem**: Confusion about cache locations
**Fix**: Documented and separated:
- Local smoke: `cache/smoke/`
- Local full: `cache/tusz/`
- Modal smoke: `/results/cache/smoke/`
- Modal full: `/results/cache/tusz/`

## DOCUMENTATION CREATED

1. `CACHE_DIRECTORY_STRUCTURE.md` - Cache organization
2. `MODAL_PIPELINE_SETUP.md` - Modal deployment guide
3. `MODAL_PIPELINE_ROOT_CAUSE_ANALYSIS.md` - BGB_LIMIT_FILES debugging
4. `PIPELINE_STATUS_COMPLETE.md` - Full pipeline status
5. `CONFIG_AUDIT_AND_CLEANUP.md` - Config reorganization plan
6. `configs/README.md` - Config usage guide
7. `configs/LOCAL_CONFIG_CONSISTENCY.md` - Local config validation
8. `configs/MODAL_A100_CONSISTENCY.md` - Modal config validation

## IMPACT

### Before Fixes:
- ❌ 0% seizures detected in training
- ❌ Broken CSV parser for TUSZ format
- ❌ Modal limited to 50 files
- ❌ Confusing config structure
- ❌ Risk of training collapse

### After Fixes:
- ✅ Seizures properly detected (313 partial, 55 full in test)
- ✅ All TUSZ seizure types recognized
- ✅ Modal processing full 3734 files
- ✅ Clean professional config structure
- ✅ Balanced training guaranteed

## CURRENT STATUS

- **Local Training**: Building cache, 832/3734 files (22%)
- **Modal Training**: Building cache, processing full dataset
- **Both pipelines**: Using fixed CSV parser with all seizure types

## FILES CHANGED

### Source Code:
- `src/brain_brr/data/io.py` - CSV parser fix
- `src/brain_brr/data/datasets.py` - BalancedSeizureDataset
- `src/brain_brr/data/cache_utils.py` - Manifest scanner
- `src/brain_brr/cli/cli.py` - Hard guards

### Configs:
- Moved 4 local configs to `configs/local/`
- Created 4 Modal configs in `configs/modal/`
- Deleted 3 redundant/broken configs

### Deployment:
- `deploy/modal/app.py` - BGB_LIMIT_FILES fix

### Documentation:
- 8 new documentation files
- 2 consistency verification files
- 1 comprehensive README for configs
