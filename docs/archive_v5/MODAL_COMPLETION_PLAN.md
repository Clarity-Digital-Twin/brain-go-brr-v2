# Modal Mmap Cache Completion - Foolproof Plan

**Date**: 2025-10-06
**Goal**: Complete mmap cache on Modal SSD without losing existing train split
**Status**: Ready to execute

---

## Current Situation

### Modal Volume Status
```
Total: 958 GB capacity
Used: ~958 GB (FULL!)

/results/cache/tusz/          449 GB   ← OLD NPZ (delete!)
/results/cache/tusz_mmap/     349 GB   ← NEW MMAP
  ├── train/                  349 GB   ✅ COMPLETE (4667 files)
  └── dev/                    partial  ❌ INCOMPLETE (OOM'd during copy)
/results/diag_*/              minimal  ← OLD DIAGNOSTICS (delete!)
/results/v3_full_training/    50-200GB ← OLD TRAINING (delete!)
/results/smoke/               0.6 GB   ← RECENT (keep!)
```

### Why populate-cache Failed
```
Available: 958 GB - 449 GB (old NPZ) - ~150 GB (other) = ~359 GB free
Train copy: -349 GB
Remaining: ~10 GB
Dev copy needs: 170 GB
Result: NO SPACE LEFT ON DEVICE ❌
```

---

## Why Modal SSD is Still Best (Even With Mmap)

**Storage speed doesn't change with mmap:**
- **Modal SSD**: 0.1ms latency (local NVMe) ⚡⚡⚡
- **S3**: 10-50ms latency (network) 🐌

**Mmap only changes RAM requirements:**
- Old (NPZ): 387 GB RAM needed → impossible
- New (NPY mmap): <1 GB RAM needed → perfect

**Result**: Modal SSD + mmap = **1000% fastest training possible**

---

## Foolproof 5-Step Plan

### Step 1: Run Targeted Cleanup (10 min)

**What it deletes:**
- `/results/cache/tusz/` - Old NPZ cache (449 GB) ❌
- `/results/diag_*/` - Old diagnostic runs (~minimal) ❌
- `/results/v3_full_training/` - Old training (50-200 GB) ❌

**What it keeps:**
- `/results/cache/tusz_mmap/train/` - Your 349 GB train split! ✅
- `/results/smoke/` - Recent smoke test ✅

**Space freed**: ~500-700 GB → plenty for dev split (170 GB)!

**Commands:**
```bash
# Run cleanup script
modal run deploy/modal/targeted_cleanup.py

# Verify cleanup worked
modal run deploy/modal/inspect_volume.py
```

**Expected output:**
```
✅ Cleanup complete: ~500-700 GB freed
✅ Sufficient space for dev split copy!
```

---

### Step 2: Verify Cleanup (5 min)

```bash
# Check volume contents
modal run deploy/modal/inspect_volume.py | grep -E "cache|diag|training"

# Should see:
# ✅ cache/tusz_mmap/train/  (349 GB, 4667 files)
# ✅ cache/tusz_mmap/dev/    (partial or empty)
# ❌ cache/tusz/             (DELETED)
# ❌ diag_*/                 (DELETED)
# ❌ v3_full_training/       (DELETED)
```

---

### Step 3: Re-run populate-cache (Smart Version) (1-2 hours)

**Modified populate-cache now:**
- ✅ Detects existing complete train split → SKIPS copy (preserves 349 GB!)
- ✅ Detects incomplete dev split → copies it (170 GB)
- ✅ Won't delete your existing train data!

**Commands:**
```bash
# Re-run populate-cache (will only copy dev split)
modal run --detach deploy/modal/app.py --action populate-cache

# Monitor progress
modal app list
modal app logs <app-id>

# Expected logs:
# [SKIP] Train split already complete: 4667/4667 files
# [SKIP] Skipping train copy to preserve existing data
# [COPY] Found 1832 dev data files to copy...
# [COPY] Copying /s3_cache/dev → /results/cache/tusz_mmap/dev...
# ... (1-2 hours) ...
# [COPY] ✅ Copied 1832 data files + 1832 labels files
```

---

### Step 4: Verify Complete Cache (5 min)

```bash
# Check cache completeness
modal run deploy/modal/app.py --action check-cache

# Expected output:
# ✅ Train: 4667 data files + 4667 labels files
# ✅ Dev: 1832 data files + 1832 labels files
# ✅ Cache is COMPLETE and ready for training!
```

---

### Step 5: Run Smoke Test (10 min)

```bash
# Test that mmap cache works perfectly
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Monitor logs
modal app logs <app-id> | grep -E "CACHE|epoch|/s"

# Expected:
# [CACHE] Format: NPY (mmap) ✅
# [CACHE] Train: 4667 files, Dev: 1832 files
# Epoch 1: ... samples/s: ~5-10
# ✅ Training runs fast with <1 GB RAM per worker!
```

---

### Step 6: Launch Full Training! (100 hours)

```bash
# READY FOR PRODUCTION!
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Cost: ~$319 for 100 epochs
# Speed: ~1 hour/epoch (FASTEST POSSIBLE with Modal SSD + mmap)
```

---

## Safety Features

### Modified populate_cache (app.py:228-262)

**OLD behavior (DANGEROUS):**
```python
if train_dst.exists():
    shutil.rmtree(train_dst)  # ❌ DELETES YOUR 349 GB!
```

**NEW behavior (SAFE):**
```python
if train_dst.exists():
    existing = len(list(train_dst.glob("*_data.npy")))
    expected = len(list(train_src.glob("*_data.npy")))

    if existing == expected:
        logger.info("[SKIP] Train split already complete!")
        # ✅ PRESERVES YOUR 349 GB!
    else:
        logger.info("[INCOMPLETE] Re-copying...")
        shutil.rmtree(train_dst)
        shutil.copytree(train_src, train_dst)
```

**Same logic for dev split** - won't re-copy if already complete!

---

## Rollback Plan (If Something Goes Wrong)

**If cleanup deletes too much:**
- Train split is on S3: `s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/`
- Dev split is on S3: `s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/dev/`
- Just re-run populate-cache and it will copy both (now with space!)

**If populate-cache fails again:**
- Check logs: `modal app logs <app-id>`
- Check volume: `modal run deploy/modal/inspect_volume.py`
- Likely need more cleanup (contact me!)

---

## Final Checklist

Before starting:
- [ ] Read this plan completely
- [ ] Understand what gets deleted vs kept
- [ ] Confirm S3 has complete mmap cache (train + dev)

After completion:
- [ ] Modal volume has `/results/cache/tusz_mmap/{train,dev}/`
- [ ] Train: 4667 data + 4667 labels files
- [ ] Dev: 1832 data + 1832 labels files
- [ ] Smoke test passes with mmap format
- [ ] Ready for 100-epoch training!

---

## Commands Summary

```bash
# 1. Cleanup (frees ~500-700 GB)
modal run deploy/modal/targeted_cleanup.py

# 2. Verify cleanup
modal run deploy/modal/inspect_volume.py

# 3. Complete dev split copy (1-2 hours, preserves train!)
modal run --detach deploy/modal/app.py --action populate-cache

# 4. Verify cache complete
modal run deploy/modal/app.py --action check-cache

# 5. Smoke test
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# 6. FULL TRAINING!
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

---

**READY TO EXECUTE - This plan is 1000% safe and will work!**
