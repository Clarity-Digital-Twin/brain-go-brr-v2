# Modal Cache Architecture: First-Principles Analysis

**Date**: 2025-10-06
**Status**: CRITICAL DECISION POINT
**Question**: S3 direct mount vs Modal SSD persistence for mmap cache?

---

## Executive Summary

**Current Situation:**
- Modal volume: **958 GiB total size**
- Mmap cache needed: **519 GB** (349 GB train + 170 GB dev)
- Populate-cache job: **FAILED** with "No space left on device"
- Root cause: **Volume is FULL of old training runs** (not insufficient capacity)

**Critical Fork in Road:**
1. **Option A**: Clean up old junk, finish copying to Modal SSD (traditional approach)
2. **Option B**: Abandon Modal SSD copy, use S3 CloudBucketMount directly (new approach)

---

## Historical Context: Why Modal SSD Was Chosen (NPZ Era)

### The Old Cache Format (NPZ Compressed)
- **Format**: Compressed NumPy archives (.npz files)
- **Size on disk**: 449 GB
- **Size in RAM**: 387 GB when fully decompressed
- **Access pattern**: Must decompress entire file to access any window

### The S3 Performance Problem (NPZ Era)
**KEY INSIGHT FROM DOCS** (`docs/08-operations/modal-volume-architecture.md:101-111`):

**Why NOT S3 Mount for Cache?**
- **S3 is SLOW**: Network latency kills training performance (50-100ms per file)
- **S3 throttling**: Can hit rate limits with parallel data loading
- **S3 costs**: Egress charges for repeatedly reading 450GB cache
- **Reliability**: Network hiccups can crash training

**Why Modal SSD Volume?**
- **FAST**: Local NVMe SSD with microsecond latency
- **Reliable**: No network issues
- **Persistent**: Survives between runs
- **Cost-effective**: One-time population, then free reads

**The Problem with NPZ + S3:**
1. **S3 → Modal latency**: ~50-100ms per file open
2. **Random access**: Balanced sampling touches ~1,000+ files per epoch
3. **Decompression overhead**: Each S3 read downloads entire file (85 MB avg)
4. **Worker coordination**: 4 workers × 1,000 files = 4,000 S3 requests/epoch
5. **Total overhead**: 4,000 × 100ms = 400 seconds = 6.7 minutes JUST for S3 latency per epoch!

**Result**: Training was too slow with S3 direct access (NPZ format)

**Solution**: Copy NPZ cache to Modal SSD once, then train fast

---

## The Game Changer: Memory-Mapped NPY Format

### New Cache Format (NPY Uncompressed)
- **Format**: Uncompressed NumPy arrays (.npy files)
- **Size on disk**: 519 GB (1.16x larger than NPZ)
- **Size in RAM**: **<1 GB per worker** (OS manages paging!)
- **Access pattern**: `mmap_mode='r'` - OS pages data in/out automatically

### Why Mmap Changes Everything

**Traditional file I/O (NPZ approach):**
```
S3 → Download → Decompress → RAM → Process
(High latency, requires full file in RAM)
```

**Memory-mapped I/O (NPY approach):**
```
Storage → Page fault → OS pages in 4KB chunks → Process
(OS handles caching, only active pages in RAM)
```

**Critical difference:**
- **NPZ**: Must copy entire file into RAM (85 MB per file)
- **NPY mmap**: OS pages data as needed (typically <10 MB active per file)

---

## First-Principles Analysis: S3 vs Modal SSD for Mmap

### Option A: Modal SSD (After Cleanup)

**Pros:**
- ✅ **Lowest latency**: Local NVMe SSD (~0.1ms access time)
- ✅ **Proven approach**: We know this works
- ✅ **Predictable performance**: No network variability
- ✅ **Free egress**: No S3 → Modal data transfer during training

**Cons:**
- ❌ **3-hour initial copy**: Must populate cache before training
- ❌ **Volume management**: Must clean up old runs regularly
- ❌ **Higher storage cost**: Modal SSD ~$0.10-0.15/GB/month vs S3 ~$0.023/GB/month
- ❌ **One-time setup**: Every new cache rebuild requires full re-copy

**Cost:**
- Storage: 519 GB × $0.10/GB/month = ~$52/month
- One-time copy: ~$1-2 compute time

### Option B: S3 CloudBucketMount (Direct)

**Pros:**
- ✅ **Zero copy time**: Start training immediately
- ✅ **Lower storage cost**: $12/month vs $52/month for 519 GB
- ✅ **Automatic persistence**: S3 is durable by design
- ✅ **Easy cache updates**: Just upload to S3, no re-populate needed
- ✅ **Mmap-friendly**: OS caches hot pages in RAM, cold pages stay in S3

**Cons:**
- ❌ **Higher latency**: S3 → Modal ~10-50ms vs SSD ~0.1ms
- ❌ **Network variability**: Could have occasional slowdowns
- ❌ **Unknown with mmap**: Haven't tested S3 + mmap performance
- ❌ **S3 request costs**: Potentially thousands of GET requests per epoch

**Cost:**
- Storage: 519 GB × $0.023/GB/month = ~$12/month
- S3 requests: ~$0.01-0.10 per training run (GET requests)

---

## The Critical Question: Does Mmap Make S3 Viable?

### Theory: Mmap Should Work Well with S3

**Why mmap + S3 might be fine:**

1. **Page-level caching**: OS caches frequently accessed pages in RAM
2. **Sequential access**: Training often accesses windows sequentially within a file
3. **Worker locality**: Each worker tends to stick to subset of files
4. **CloudBucketMount optimizations**: Modal may prefetch or cache at kernel level

**Expected behavior:**
- First epoch: Slow (cold cache, many S3 reads)
- Subsequent epochs: Fast (hot pages cached in RAM, minimal S3 reads)

### Theory: Mmap Might Still Be Slow with S3

**Why mmap + S3 might fail:**

1. **Page faults**: Each new page triggers S3 GET request
2. **Random access**: Balanced sampling jumps across many files randomly
3. **RAM pressure**: 96 GB RAM might not cache enough hot pages
4. **CloudBucketMount limitations**: May not optimize for mmap workloads

**Expected behavior:**
- All epochs: Slow (constant page faults → S3 reads)
- Training time: 2-3x slower than Modal SSD

---

## What's Actually on the Modal Volume Right Now?

**Volume contents (as of Oct 6, 2025):**
```
Total: 958 GiB used

/results/cache/tusz/          ~449 GB   🚨 OLD NPZ CACHE - NO LONGER NEEDED! 🚨
/results/cache/tusz_mmap/     ~349 GB   NEW mmap cache (train complete + partial dev)
/results/diag_1a_amp_off/     ~minimal  (just tensorboard events, 7 days old)
/results/diag_1b_fallback/    ~minimal  (just tensorboard events, 7 days old)
/results/diag_2a_blocking/    ~minimal  (just tensorboard events, 7 days old)
/results/smoke/               ~0.6 GB   (3 checkpoints @ 190 MB each, recent)
/results/v3_full_training/    ~50-200 GB? (7 days old - likely has 100 checkpoints)
```

**SMOKING GUN DISCOVERED:**
The old NPZ cache (`/results/cache/tusz/`) is wasting **449 GB**!

This is why populate-cache failed:
```
Available space: 958 GB - 449 GB (old NPZ) - ~150 GB (other junk) = ~359 GB free
Train mmap copy: -349 GB
Remaining: ~10 GB
Dev mmap copy needs: 170 GB
Result: NO SPACE LEFT ON DEVICE ❌
```

**Quick fix to enable Modal SSD approach:**
Delete `/results/cache/tusz/` (old NPZ cache) → frees 449 GB → plenty of space for dev split!

---

## Root Cause of populate-cache OOM

**What actually happened:**

1. **Started**: Copy train split (349 GB) → Success ✅
2. **Runtime**: 3 hours for train split
3. **Started**: Copy dev split (170 GB) → **FAILED** ❌
4. **Error**: "No space left on device"

**Why it failed:**

Modal volume had:
- 958 GB total capacity
- ~600-800 GB already used by old runs
- 349 GB copied (train)
- Started copying dev (170 GB) → Not enough free space!

**Math:**
```
Available before copy: 958 GB - ~600 GB (old junk) = ~358 GB free
Train copy: -349 GB
Remaining: ~9 GB free
Dev copy needs: 170 GB
Result: OOM ❌
```

---

## Decision Framework

### If We Choose Modal SSD (Option A):

**Steps:**
1. Investigate volume contents (via inspect_volume.py or Modal UI)
2. Delete old diagnostic runs (diag_*/)
3. Delete old training run (v3_full_training/)
4. Keep recent smoke test (smoke/) for reference
5. Re-run populate-cache to copy dev split (170 GB)
6. Verify cache completeness
7. Train

**Time investment**: ~1-2 hours
**Risk**: Low (proven approach)
**Ongoing cost**: $52/month storage

### If We Choose S3 Direct (Option B):

**Steps:**
1. Update deploy/modal/app.py:
   - Remove populate_cache() function
   - Add S3 mmap cache mount to train() function
   - Update configs to point to /cache (S3 mount) instead of /results/cache
2. Run Modal smoke test (50 files) to measure performance
3. Compare epoch time vs historical SSD performance
4. If acceptable (<2x slowdown), proceed with full training
5. If too slow, fall back to Option A

**Time investment**: ~2-3 hours (includes testing)
**Risk**: Medium (unproven with mmap)
**Ongoing cost**: $12/month storage

---

## Recommendation

**RECOMMENDATION: Test S3 first, then decide based on data**

**Critical information FOUND:**
1. ✅ **Volume contents breakdown** - Old NPZ cache wasting 449 GB (can be deleted)
2. ✅ **Historical S3 performance docs** - S3 rejected due to NPZ latency issues
3. ❓ **Mmap + S3 performance test** - UNKNOWN (need to test!)

**Key insight: NPZ != NPY for S3 performance**

**NPZ + S3 was slow because:**
- Each file access downloads entire 85 MB file
- 1,000+ files × 100ms latency = 100+ seconds overhead per epoch

**NPY mmap + S3 might be fast because:**
- OS pages data in 4KB chunks as needed
- Hot pages cached in RAM, cold pages stay in S3
- 96 GB RAM can cache ~20% of dataset (most-accessed files)

**Recommended plan:**

### Phase 1: Quick S3 Performance Test (1 hour) 🚀 START HERE

**Goal**: Determine if mmap + S3 is fast enough for training

```bash
# 1. Modify app.py to mount S3 mmap cache
# 2. Run Modal smoke test (50 files, 1 epoch)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# 3. Monitor performance
modal app logs <app-id> | grep "Epoch\|batch\|/s"

# 4. Measure critical metrics:
# - Epoch time (should be <15 min for smoke)
# - Batch processing speed (should be >5 samples/sec)
# - RAM usage (should be <50 GB)
# - Any S3 throttling errors
```

**Decision criteria:**
```
If epoch time < 15 min && no S3 errors:
  → ✅ Use S3 direct (Option B)
  → Save $40/month, zero copy time, simpler architecture

If epoch time > 15 min OR S3 throttling:
  → ✅ Use Modal SSD (Option A)
  → Worth the cost for performance
```

### Phase 2: If S3 Works - Deploy! (10 min)

```bash
# Update configs to use /cache (S3 mount) instead of /results/cache
# Run full training immediately (no cache copy needed!)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

### Phase 3: If S3 Fails - Clean & Copy (2 hours)

```bash
# 1. Delete old NPZ cache (frees 449 GB)
modal run deploy/modal/cleanup_volume.py  # Or manual deletion via UI

# 2. Re-run populate-cache (will succeed now with space)
modal run --detach deploy/modal/app.py --action populate-cache

# 3. Verify cache completeness
modal run deploy/modal/app.py --action check-cache

# 4. Train
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

---

## My Strong Recommendation

**Test S3 performance first** (Option B smoke test) because:

1. ✅ **Low risk**: 1-hour test, easy to pivot
2. ✅ **High reward**: If it works, saves $40/month + eliminates 3-hour copy step
3. ✅ **New information**: Mmap changes I/O characteristics fundamentally vs NPZ
4. ✅ **Modal might optimize**: CloudBucketMount may have prefetching/caching for mmap workloads
5. ✅ **RAM is large**: 96 GB can cache most hot pages even with 4 workers

**Why this is different from historical S3 rejection:**
- **Then (NPZ)**: Each access downloads entire 85 MB file → S3 latency disaster
- **Now (NPY mmap)**: OS pages 4KB chunks → S3 latency amortized across many samples

**Worst case if S3 is too slow:**
- We lose 1 hour testing
- We fall back to Modal SSD cleanup approach (proven to work)
- We still save compared to guessing wrong and cleaning volume first

**Best case if S3 works:**
- Start training immediately (no 3-hour copy wait)
- Save $40/month ongoing
- Simpler architecture (no populate-cache step)
- Easier cache updates (just upload to S3)

---

## NEXT STEP (User Decision Required)

**Option 1 (RECOMMENDED): Test S3 first**
```bash
# I'll modify app.py to mount S3 mmap cache and run smoke test
# Takes ~1 hour, gives definitive answer
```

**Option 2 (CONSERVATIVE): Clean Modal SSD first**
```bash
# I'll clean up old NPZ cache, finish populate-cache
# Takes ~3 hours, proven approach
```

**Please confirm which path you prefer, or if you want me to proceed with Option 1 (test S3 first).**
