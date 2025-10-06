# P0 BLOCKER: Modal Cache Path Hardcoded to Wrong Location

**Date**: October 6, 2025
**Severity**: **P0 CRITICAL** - Blocks all Modal training
**Status**: ❌ IDENTIFIED - Requires immediate fix
**Impact**: 93 minutes of cache population wasted, mmap conversion not being used

---

## Executive Summary

**The Problem**: Despite updating all config files to use `/results/cache/tusz_mmap` (memory-mapped NPY cache), Modal training attempts to use `/results/cache/tusz` (old NPZ location).

**Root Cause**: `deploy/modal/app.py` has **TWO hardcoded cache paths** that OVERRIDE the config files.

**Evidence**: Modal smoke test logs show:
```
[CONFIG] Using cache directory: /results/cache/tusz  ❌ WRONG!
```

But `configs/modal/smoke.yaml` specifies:
```yaml
cache_dir: /results/cache/tusz_mmap  ✅ CORRECT!
```

---

## Root Cause Analysis

### Location 1: Line 658 (Cache Validation Logic)

**File**: `deploy/modal/app.py`
**Function**: `train()`
**Line**: 658

```python
# ALWAYS use SSD cache on Modal, NOT S3!
cache_dir = "/results/cache/tusz"  # ❌ HARDCODED OLD PATH!
```

**Context**: This is inside cache validation logic that runs before training starts.

### Location 2: Lines 811-821 (Config Override Logic)

**File**: `deploy/modal/app.py`
**Function**: `train()`
**Lines**: 811-821

```python
# CRITICAL: Cache architecture
# - Cache is on Modal SSD volume at /results/cache/tusz
# - Smoke tests use SAME cache with BGB_LIMIT_FILES=50
# - NO SEPARATE SMOKE CACHE EXISTS OR IS NEEDED
cache_dir = "/results/cache/tusz"  # ❌ HARDCODED OLD PATH!

# Ensure cache directories exist with correct structure
from pathlib import Path
Path(cache_dir).mkdir(parents=True, exist_ok=True)
(Path(cache_dir) / "train").mkdir(exist_ok=True)
(Path(cache_dir) / "dev").mkdir(exist_ok=True)

# Set cache_dir in both data and experiment sections
exp["cache_dir"] = cache_dir
data.setdefault("data", {})["cache_dir"] = cache_dir  # ❌ OVERWRITES CONFIG!
```

**This is the smoking gun**: Line 821 explicitly OVERWRITES the config's cache_dir setting!

---

## Impact Analysis

### What Went Wrong

1. ✅ We spent 93 minutes populating `/results/cache/tusz_mmap` with mmap NPY files
2. ✅ We updated ALL config files to point to `/results/cache/tusz_mmap`
3. ✅ We documented the change everywhere (CLAUDE.md, configs/README.md, etc.)
4. ❌ **But** deploy/modal/app.py ignored all of this and used hardcoded paths!

### Consequences

- **Cache miss**: Training will try to build cache at `/results/cache/tusz` from scratch
- **Wrong format**: Will try to create NPZ files (old format) instead of using NPY mmap
- **Wasted time**: 93 minutes of populate-cache work completely ignored
- **Wasted money**: S3 → Modal copy happened for nothing
- **Training degradation**: Won't get mmap benefits (<1 GB RAM vs 387 GB)

---

## Historical Context

### Why Was This Hardcoded?

Looking at comments in app.py:

```python
# Comment at line 167:
# Cache lives on Modal SSD volume at /results/cache/tusz for performance.

# Comment at line 552:
# NO /cache mount! Cache is on SSD at /results/cache/tusz

# Comment at line 808:
# CRITICAL: Cache architecture
# - Cache is on Modal SSD volume at /results/cache/tusz
```

**Analysis**: These comments were written BEFORE the mmap conversion. The hardcoded paths were intentional at the time (to prevent S3 caching), but became obsolete after we:
1. Converted cache format (NPZ → NPY mmap)
2. Updated directory naming (`tusz` → `tusz_mmap`)

**The code was never updated to reflect the conversion!**

---

## Why This Wasn't Caught Earlier

1. **Configs looked correct**: All YAML files had the right paths
2. **Documentation looked correct**: CLAUDE.md, README.md all updated
3. **Local testing**: We didn't test locally (WSL2 has different paths)
4. **populate-cache success**: That script DID use the correct path (`/results/cache/tusz_mmap`)
5. **First smoke test**: This is the FIRST time we actually ran training with the new cache

**Lesson**: Hardcoded overrides in deployment code can silently break config-driven changes.

---

## The Fix

### Required Changes

#### 1. Remove Hardcoded Path #1 (Line 658)

**BEFORE**:
```python
# ALWAYS use SSD cache on Modal, NOT S3!
cache_dir = "/results/cache/tusz"  # Fixed path on SSD volume
```

**AFTER**:
```python
# Use cache_dir from config (defaults to /results/cache/tusz_mmap for mmap NPY)
cache_dir = config_data.get("data", {}).get("cache_dir", "/results/cache/tusz_mmap")
```

#### 2. Remove Hardcoded Path #2 (Lines 811-821)

**BEFORE**:
```python
cache_dir = "/results/cache/tusz"  # SSD volume, NOT S3!

# Ensure cache directories exist with correct structure
from pathlib import Path
Path(cache_dir).mkdir(parents=True, exist_ok=True)
(Path(cache_dir) / "train").mkdir(exist_ok=True)
(Path(cache_dir) / "dev").mkdir(exist_ok=True)

# Set cache_dir in both data and experiment sections
exp["cache_dir"] = cache_dir
data.setdefault("data", {})["cache_dir"] = cache_dir  # OVERWRITES CONFIG!
```

**AFTER**:
```python
# Use cache_dir from config (mmap format)
cache_dir = data.get("data", {}).get("cache_dir", "/results/cache/tusz_mmap")

# Ensure cache directories exist with correct structure
from pathlib import Path
Path(cache_dir).mkdir(parents=True, exist_ok=True)
(Path(cache_dir) / "train").mkdir(exist_ok=True)
(Path(cache_dir) / "dev").mkdir(exist_ok=True)

# Set cache_dir in experiment section (data section already has it from config)
exp["cache_dir"] = cache_dir
# DO NOT overwrite data["data"]["cache_dir"] - respect config!
```

#### 3. Update Comments

Replace all mentions of `/results/cache/tusz` in comments with `/results/cache/tusz_mmap`.

**Files to update**:
- `deploy/modal/app.py` (lines 167, 552, 808)
- `deploy/modal/inspect_volume.py` (line 82)

---

## Testing Plan

### After Fix

1. **Verify config propagation**:
   ```bash
   # Add debug logging to confirm cache_dir is read from config
   grep -A2 "cache_dir from config" deploy/modal/app.py
   ```

2. **Dry run smoke test**:
   ```bash
   # Should show correct path in logs
   modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
   # Look for: [CONFIG] Using cache directory: /results/cache/tusz_mmap
   ```

3. **Verify cache exists**:
   ```bash
   modal run deploy/modal/app.py --action check-cache
   # Should find 4667 train + 1832 dev NPY files
   ```

4. **Full smoke test**:
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
   # Should use existing mmap cache, NOT rebuild
   ```

---

## Verification Checklist

Before declaring this fixed:

- [ ] Line 658: Cache validation reads from config
- [ ] Line 811: Cache setup reads from config
- [ ] Line 821: Does NOT overwrite config cache_dir
- [ ] Comments updated to reference `tusz_mmap`
- [ ] Smoke test logs show `/results/cache/tusz_mmap`
- [ ] Smoke test completes without building cache
- [ ] Memory usage stays <2 GB per worker (mmap benefit)

---

## Related Issues

### Why populate-cache Worked

The `populate_cache()` function at line 214 correctly used:
```python
dst = Path("/results/cache/tusz_mmap")  # SSD volume (memory-mapped cache)
```

So populate-cache created files in the RIGHT place, but training was looking in the WRONG place!

### Why check-cache Would Have Caught This

If we had run `check-cache` before smoke test, we would have seen:
```
❌ Cache not found at /results/cache/tusz
✅ Cache exists at /results/cache/tusz_mmap (4667 train + 1832 dev)
```

**Lesson**: Always run check-cache after populate-cache and before training.

---

## Priority Assessment

**Severity**: P0 CRITICAL
**Urgency**: IMMEDIATE
**Training Blocked**: YES
**Data Loss Risk**: NO (cache still exists at correct location)
**Est. Fix Time**: 15 minutes
**Est. Test Time**: 10 minutes (smoke test)

---

## Action Items

1. **Immediate**: Fix lines 658, 811-821 in app.py
2. **Immediate**: Update comments to reference `tusz_mmap`
3. **Immediate**: Test smoke test shows correct path
4. **Before full training**: Run check-cache to verify
5. **Documentation**: Update modal.md to warn about config overrides

---

**Status**: Ready to fix immediately
**Assignee**: AI Agent (this fix)
**Timeline**: Fix now, smoke test in 10 min, full training after smoke passes
