# Docker Implementation Plan V2 - SSOT

**Status:** 🎯 READY FOR AUDIT
**Date:** October 4, 2025
**Version:** v3.6.1

---

## Ground Truth Investigation Results

### Dataset Architecture (VERIFIED from code + docs)

**Three Dataset Types:**

1. **BalancedSeizureDataset** (Training with balanced sampling)
   - Constructor: `__init__(cache_dir: Path, ...)`
   - Files needed: `manifest.json` + `*.npz` files
   - **NO EDF FILES NEEDED** ✅

2. **ValidationDataset** (Validation with natural distribution)
   - Constructor: `__init__(cache_dir: Path, allowed_cache_files, ...)`
   - Files needed: `manifest.json` + `*.npz` files
   - **NO EDF FILES NEEDED** ✅

3. **EEGWindowDataset** (Smoke tests or fallback)
   - Constructor: `__init__(edf_files: list[Path], label_files, cache_dir, ...)`
   - Files needed: Original EDF files + CSV labels (or NPZ cache)
   - Optional: `_dataset_index.json` for fast startup
   - **NEEDS EDF FILES** if `allow_on_demand=True` ❌

### Local Training Setup (VERIFIED)

**File Structure:**
```
/home/jj/proj/brain-go-brr-v2/
├── data_ext4/tusz/edf/          # 109GB - Original EDFs + CSV labels
│   ├── train/                   # 4667 files
│   ├── dev/                     # 1832 files
│   └── eval/
├── cache/tusz/                  # 449GB - Preprocessed NPZ + manifests
│   ├── .cache_metadata.json
│   ├── train/
│   │   ├── manifest.json        # 27MB - Required for BalancedSeizureDataset
│   │   ├── _dataset_index.json  # 282B - Stale (smoke test artifact)
│   │   └── *.npz                # 4667 files, 306GB
│   └── dev/
│       ├── manifest.json        # 13MB - Required for ValidationDataset
│       ├── _dataset_index.json  # 148KB - Valid for 1832 files
│       └── *.npz                # 1832 files, 143GB
└── results/                     # Training outputs
```

**Configs:**
- `configs/local/train.yaml`:
  - `data_dir: data_ext4/tusz/edf` (original EDFs)
  - `cache_dir: cache/tusz` (preprocessed)
  - `use_balanced_sampling: true` → Uses BalancedSeizureDataset

- `configs/local/smoke.yaml`:
  - `data_dir: data_ext4/tusz/edf` (original EDFs)
  - `cache_dir: cache/tusz` (preprocessed)
  - `use_balanced_sampling: false` → Uses EEGWindowDataset + BGB_SMOKE_TEST=1

### Modal Training Setup (VERIFIED from deploy/modal/app.py)

**Volume Mounts:**
```python
data_mount = modal.CloudBucketMount(
    "s3://brain-go-brr-eeg-data-20250919",
    key_prefix="tusz/",  # → Mounted at /data/{train,dev,eval}/
)
results_volume = modal.Volume.from_name("brain-go-brr-results")  # SSD cache
```

**Paths:**
- `/data/edf/` - Original EDFs from S3 (109GB, read-only mount)
- `/results/cache/tusz/` - Preprocessed NPZ + manifests (449GB, persistent SSD)
- `/results/` - Training outputs

**Configs:**
- `configs/modal/smoke.yaml`:
  - `data_dir: /data/edf`
  - `cache_dir: /results/cache/tusz`
  - `use_balanced_sampling: false` + `BGB_LIMIT_FILES=50`

- `configs/modal/train.yaml`:
  - `data_dir: /data/edf`
  - `cache_dir: /results/cache/tusz`
  - `use_balanced_sampling: true`

**Key Insight:** Modal mounts BOTH original EDFs AND preprocessed cache!

---

## Why Docker V1 Failed

### Root Cause

**Docker V1 mounted:**
- ✅ `/app/cache/tusz/` → Preprocessed cache (449GB)
- ❌ NO `data_ext4/` mount!

**smoke.yaml uses:**
- `use_balanced_sampling: false` → Triggers EEGWindowDataset
- EEGWindowDataset needs: `data_ext4/tusz/edf/train/` EDFs
- Docker container ERROR: `Split directory not found: data_ext4/tusz/edf/train`

### Why This Wasn't Obvious

1. **Assumption**: "Preprocessed cache means no EDFs needed"
   - TRUE for: BalancedSeizureDataset + ValidationDataset
   - FALSE for: EEGWindowDataset (used by smoke tests)

2. **Incomplete .dockerignore**: Excluded `data_ext4/` from build context
   - Correct for build context (don't copy 109GB into image)
   - Wrong for runtime (need volume mount!)

3. **Config confusion**: `data_dir` vs `cache_dir`
   - `data_dir`: Original EDFs (109GB) - needed by EEGWindowDataset
   - `cache_dir`: Preprocessed NPZ (449GB) - needed by ALL datasets

---

## Docker V2 Solution - Match Local/Modal Exactly

### Volume Mount Strategy

**Mount THREE directories:**

1. **Original EDFs** (read-only, 109GB)
   ```yaml
   - ./data_ext4:/app/data_ext4:ro
   ```
   - Why: Enables smoke tests with EEGWindowDataset
   - Used by: `data_dir: data_ext4/tusz/edf` in configs

2. **Preprocessed cache** (read-only, 449GB)
   ```yaml
   - ./cache/tusz:/app/cache/tusz:ro
   ```
   - Why: Primary data source for training
   - Used by: `cache_dir: cache/tusz` in configs

3. **Results** (read-write)
   ```yaml
   - ./results:/app/results:rw
   ```
   - Why: Checkpoints, logs, outputs

**Total mounted:** ~560GB (all read-only except results)

### Config Changes

**NO CONFIG CHANGES NEEDED!**

Local configs already expect:
- `data_dir: data_ext4/tusz/edf` → `/app/data_ext4/tusz/edf` (mounted)
- `cache_dir: cache/tusz` → `/app/cache/tusz` (mounted)

Paths match exactly!

---

## Implementation Steps

### 1. Update docker-compose.yml

```yaml
services:
  train-base:
    # ... (keep existing build/image/runtime config)

    volumes:
      # CRITICAL: Mount BOTH data directories (matches local setup)

      # Original EDFs (read-only) - 109GB
      # Required for: smoke tests with EEGWindowDataset
      - ./data_ext4:/app/data_ext4:ro

      # Preprocessed cache (read-only) - 449GB
      # Required for: ALL training (BalancedSeizureDataset, ValidationDataset)
      - ./cache/tusz:/app/cache/tusz:ro

      # Results (read-write)
      # Required for: checkpoints, logs, outputs
      - ./results:/app/results:rw

      # Configs (read-only)
      - ./configs:/app/configs:ro
```

### 2. Update .env

```bash
# Docker Compose environment variables
# NO CHANGES NEEDED - paths are relative to /app workdir in container
```

### 3. Test Smoke

```bash
# Should work identically to local:
docker compose up smoke-test

# Expected behavior:
# - Loads from cache/tusz/ (fast)
# - Uses EEGWindowDataset with BGB_SMOKE_TEST=1 (3 files)
# - Completes 1 epoch in ~5 minutes
```

### 4. Test Full Training

```bash
docker compose up train

# Expected behavior:
# - Loads from cache/tusz/train/manifest.json (instant)
# - Uses BalancedSeizureDataset (61,616 windows)
# - Validation uses dev/manifest.json (instant)
# - Same performance as local training
```

---

## File Dependency Matrix

| Use Case | Dataset | data_dir (EDFs) | cache_dir (NPZ) | manifest.json | _dataset_index.json |
|----------|---------|-----------------|-----------------|---------------|---------------------|
| **Local smoke** | EEGWindowDataset | ✅ REQUIRED | ✅ REQUIRED | ❌ | ⚠️ Optional (fast startup) |
| **Local train** | BalancedSeizureDataset | ❌ | ✅ REQUIRED | ✅ REQUIRED | ❌ |
| **Local val** | ValidationDataset | ❌ | ✅ REQUIRED | ✅ REQUIRED | ❌ |
| **Modal smoke** | EEGWindowDataset | ✅ REQUIRED | ✅ REQUIRED | ❌ | ⚠️ Optional |
| **Modal train** | BalancedSeizureDataset | ❌ | ✅ REQUIRED | ✅ REQUIRED | ❌ |
| **Docker V1 (FAILED)** | EEGWindowDataset | ❌ **MISSING!** | ✅ | ❌ | ❌ |
| **Docker V2 (FIX)** | EEGWindowDataset | ✅ MOUNTED | ✅ MOUNTED | ✅ | ✅ |

---

## Validation Checklist

Before running Docker:

```bash
# 1. Verify data_ext4/ exists
ls -lh data_ext4/tusz/edf/train/ | head -5
# Should show: 4667 directories (patients)

# 2. Verify cache exists with manifests
ls -lh cache/tusz/train/manifest.json
ls -lh cache/tusz/dev/manifest.json
# Should show: 27MB and 13MB files

# 3. Verify NPZ files
ls -1 cache/tusz/train/*.npz | wc -l  # Should be 4667
ls -1 cache/tusz/dev/*.npz | wc -l    # Should be 1832

# 4. Check disk space for volume mounts
df -h .
# Need: 560GB available (109GB data_ext4 + 449GB cache + headroom)
```

---

## Alternative: Cache-Only Docker (NOT RECOMMENDED)

**If you want to skip smoke tests:**

1. Remove `data_ext4/` mount
2. Change smoke.yaml to `use_balanced_sampling: true`
3. Smoke test will use BalancedSeizureDataset (full manifest, slower)

**Why not recommended:**
- Breaks parity with local smoke tests
- Can't test EEGWindowDataset pathway
- Modal uses both pathways, Docker should too

---

## Success Criteria

✅ **Docker V2 is correct when:**

1. `docker compose up smoke-test` completes without "Split directory not found"
2. Smoke test uses 3 files (BGB_SMOKE_TEST=1)
3. Full training uses BalancedSeizureDataset (61,616 windows)
4. Validation uses ValidationDataset with manifest (instant load)
5. Performance matches local training (99%+ speed)
6. No adhoc hacks or workarounds

---

## Files to Modify

1. **docker-compose.yml** - Add data_ext4 volume mount
2. **DOCKER_IMPLEMENTATION_PLAN.md** - Archive (replace with this V2)
3. **(Optional) README.md** - Update Docker quick start

---

## Questions for Senior Audit

1. ✅ Is mounting 560GB via volumes acceptable? (Answer: Yes, read-only mounts are efficient)
2. ✅ Should we match Modal's dual-mount approach? (Answer: Yes, for consistency)
3. ✅ Any concerns about path `/app/data_ext4/` vs `/data/`? (Answer: No, configs use relative paths)
4. ✅ Should smoke test work identically in Docker? (Answer: Yes, full parity required)

---

**Ready for implementation!** 🚀
