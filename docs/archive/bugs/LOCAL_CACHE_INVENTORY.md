# LOCAL CACHE INVENTORY - ALL LOCATIONS

**Generated**: 2025-09-20 20:10 EDT
**Total cache files**: 6780 .npz files

## CACHE DIRECTORIES

### 1. cache/train/ (27GB - 246 files)
**Path**: `/home/jj/proj/brain-go-brr-v2/cache/train/`
**Size**: 27GB
**Files**: 246 .npz files
**Purpose**: Training cache from smoke test runs
**Status**: ACTIVE - being used by local training
**Created**: Sep 18-20, 2025
**Notable**: This is what got archived to tusz_train_cache_256hz_10-20_v1_20250920.tar.gz

### 2. cache/data/ (449GB - 6509 files)
**Path**: `/home/jj/proj/brain-go-brr-v2/cache/data/`
**Size**: 449GB (MASSIVE)
**Subdirectories**:
  - `cache/data/train/` - 6509 files
  - `cache/data/val/` - 25 files
**Purpose**: Old cache from different config/experiment
**Status**: NOT IN USE - old data
**Created**: Sep 18-19, 2025

### 3. cache/val/ (305MB - 10 files)
**Path**: `/home/jj/proj/brain-go-brr-v2/cache/val/`
**Size**: 305MB
**Files**: 10 .npz files
**Purpose**: Validation cache
**Status**: May be used for validation
**Created**: Sep 19, 2025

### 4. cache/test/ (Empty)
**Path**: `/home/jj/proj/brain-go-brr-v2/cache/test/`
**Size**: 4KB (empty)
**Files**: 0
**Purpose**: Test cache (never used)
**Status**: EMPTY

## ARCHIVED CACHES

### tusz_train_cache_256hz_10-20_v1_20250920.tar.gz
**Path**: `/home/jj/proj/brain-go-brr-v2/tusz_train_cache_256hz_10-20_v1_20250920.tar.gz`
**Size**: 27GB compressed
**Contents**: All 246 files from cache/train/
**Created**: Sep 20, 2025 19:27 EDT
**Purpose**: Archive of training cache for Modal deployment

## TOTAL STORAGE USED

- **cache/data/**: 449GB (OLD - CAN BE DELETED)
- **cache/train/**: 27GB (ACTIVE)
- **cache/val/**: 305MB
- **cache/test/**: 4KB
- **Archive**: 27GB
- **TOTAL**: ~503GB

## WHAT'S ACTUALLY BEING USED

### LOCAL TRAINING (WSL2)
- **Using**: `cache/train/` (246 files, 27GB)
- **Config expects**: `cache/tusz/` (doesn't exist)
- **Actually uses**: `cache/train/` (hardcoded in dataset)

### MODAL TRAINING (Cloud)
- **Building to**: `/results/cache/tusz/`
- **Will have**: All 3734 files when done
- **Persistent**: Yes, stays on Modal volume

## RECOMMENDATIONS

1. **DELETE cache/data/** - 449GB of old unused cache
   ```bash
   rm -rf cache/data/
   ```

2. **Keep cache/train/** - Active local cache (27GB)

3. **Keep archive** - tusz_train_cache_256hz_10-20_v1_20250920.tar.gz (27GB)

## SUMMARY

You have:
- 6780 total .npz cache files
- 503GB of cache data
- Only 27GB actually being used
- 449GB can be deleted (cache/data/)