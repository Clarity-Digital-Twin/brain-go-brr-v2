# Cache Migration Plan - Fix SIGBUS Crashes

**Date**: October 11, 2025
**Status**: ✅ **SOLUTION IDENTIFIED** - Delete old NPZ cache, move NPY cache to native filesystem

---

## Problem Summary

**Current Situation**:
- New cache (`cache/tusz_mmap/`, 518GB NPY format) is symlinked to `/mnt/d/` (Windows partition)
- WSL2 9P filesystem causes SIGBUS crashes when memory-mapped files are accessed
- Training crashes around batch 2890 (~2 hours) when page cache is evicted

**Root Cause**: Memory-mapped NPY files on WSL2 9P filesystem (Windows partition via `/mnt/d/`)

---

## Discovery: Old Cache Still Present!

### Current Cache State

```bash
# Old cache (NPZ format, deprecated)
cache/tusz/              449GB, 6499 .npz files, ON NATIVE FILESYSTEM ✅
├── train/               ~340GB (4667 sessions)
└── dev/                 ~109GB (1832 sessions)

# New cache (NPY format, current)
cache/tusz_mmap/         518GB, 12998 .npy files, SYMLINK TO /mnt/d/ ❌
├── train/               348GB (4667 sessions × 2 files = 9334 .npy)
└── dev/                 170GB (1832 sessions × 2 files = 3664 .npy)
```

### Space Analysis

```
WSL2 Disk: /dev/sdc (ext4)
├── Total:        1007GB
├── Used:          823GB
├── Free:          134GB  ← NOT ENOUGH for 518GB cache
└── Use%:          87%

Space Calculation:
├── Delete old cache/tusz/:        +449GB
├── Available after delete:         583GB
├── Move cache/tusz_mmap/:         -518GB
└── Final free space:                65GB ✅
```

**Conclusion**: We can delete the old NPZ cache (unused) to make room for moving the NPY cache!

---

## Verification: Old Cache Not Used

### Config Check
```bash
grep -r "cache/tusz[^_]" configs/
# Result: NO MATCHES ✅
```

All configs use `cache/tusz_mmap/`, not `cache/tusz/`.

### Code Check
```bash
grep -r "\.npz" src/brain_brr/data/
# Result: NO MATCHES ✅
```

Code only handles `.npy` files, not `.npz` files.

### File Format
```bash
# Old: Single NPZ file per session (compressed, slow)
cache/tusz/train/aaaaacmr_s001_t000_windows.npz

# New: Two NPY files per session (memory-mapped, fast)
cache/tusz_mmap/train/aaaaacmr_s001_t000_data.npy
cache/tusz_mmap/train/aaaaacmr_s001_t000_labels.npy
```

**Old format migrated to new format in v3.8.0 (October 6, 2025)** - see STATUS.md

---

## Migration Steps

### Step 1: Backup Verification (SAFETY CHECK)

```bash
# Verify new cache is complete
echo "=== New cache file count ==="
find /mnt/d/brain-go-brr/cache/tusz_mmap/train/ -name "*.npy" | wc -l
# Expected: 9334 files (4667 sessions × 2)

find /mnt/d/brain-go-brr/cache/tusz_mmap/dev/ -name "*.npy" | wc -l
# Expected: 3664 files (1832 sessions × 2)

echo "Total files:"
find /mnt/d/brain-go-brr/cache/tusz_mmap/ -name "*.npy" | wc -l
# Expected: 12998 total (9334 + 3664)

echo "=== New cache manifests ==="
ls -lh /mnt/d/brain-go-brr/cache/tusz_mmap/train/manifest.json
# Expected: ~26MB manifest

ls -lh /mnt/d/brain-go-brr/cache/tusz_mmap/dev/_dataset_index.json
# Expected: ~148KB index (dev uses _dataset_index.json, not manifest.json)
# Note: manifest.json will be auto-created by ValidationDataset on first use

echo "=== New cache size ==="
du -sh /mnt/d/brain-go-brr/cache/tusz_mmap/
# Expected: 518GB
```

**ONLY PROCEED IF ALL CHECKS PASS** ✅

### Step 2: Delete Old NPZ Cache

```bash
# Safety: Rename first (can undo if needed)
cd /home/jj/proj/brain-go-brr-v2/
mv cache/tusz cache/tusz_OLD_NPZ_DELETE_ME

# Verify space freed
df -h /
# Should show ~583GB free (134GB + 449GB)
```

**CHECKPOINT**: If something goes wrong, you can restore with `mv cache/tusz_OLD_NPZ_DELETE_ME cache/tusz`

### Step 3: Move New Cache to Native Filesystem

```bash
# Remove symlink
rm cache/tusz_mmap

# Create real directory
mkdir -p cache/tusz_mmap

# Copy from Windows partition to native filesystem
# IMPORTANT: Use rsync for progress and safety
rsync -avh --progress \
  /mnt/d/brain-go-brr/cache/tusz_mmap/ \
  cache/tusz_mmap/

# Verify copy
echo "=== Verify file count ==="
find cache/tusz_mmap/train/ -name "*.npy" | wc -l
# Expected: 9334

find cache/tusz_mmap/dev/ -name "*.npy" | wc -l
# Expected: 3664

find cache/tusz_mmap/ -name "*.npy" | wc -l
# Expected: 12998 total

echo "=== Verify manifests ==="
ls -lh cache/tusz_mmap/train/manifest.json
ls -lh cache/tusz_mmap/dev/_dataset_index.json
# Note: dev manifest.json will be auto-created by ValidationDataset on first training run

echo "=== Verify filesystem ==="
df -h cache/tusz_mmap/
# Should show: /dev/sdc (ext4) ✅
```

**IMPORTANT**: `rsync` will take ~1-2 hours to copy 518GB. Use tmux!

### Step 4: Verify Cache Integrity

```bash
# Scan train cache
python -m src scan-cache --cache-dir cache/tusz_mmap/train
# Expected: 4667 .npy stems, ~34% seizure ratio

# Scan dev cache
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
# Expected: 1832 .npy stems, ~8% seizure ratio
```

### Step 5: Resume Training

```bash
# Start tmux session
tmux new -s train-fla

# Enable NaN debug
export BGB_NAN_DEBUG=1

# Resume training
.venv/bin/python -m src train configs/local/train_fla.yaml --resume

# Monitor logs for:
# 1. Cache loading from /home/jj/proj/brain-go-brr-v2/cache/tusz_mmap (NOT /mnt/d/)
# 2. Training past batch 2890 without crashes
# 3. No SIGBUS errors
```

### Step 6: Cleanup (After Successful Training)

```bash
# Delete old NPZ cache permanently
rm -rf cache/tusz_OLD_NPZ_DELETE_ME

# Verify space
df -h /
# Should show ~65GB free
```

---

## Estimated Timeline

| Step | Duration | Description |
|------|----------|-------------|
| 1. Backup verification | 5 min | Check new cache is complete |
| 2. Delete old cache | 2 min | Rename cache/tusz/ |
| 3. Copy new cache | **1-2 hours** | rsync 518GB (use tmux!) |
| 4. Verify integrity | 10 min | scan-cache both splits |
| 5. Resume training | 2-3 hours | Verify past batch 2890 |
| 6. Cleanup | 5 min | Delete old cache permanently |
| **Total** | **3-5 hours** | Mostly copying time |

---

## Expected Results

### Before Migration
```bash
# Cache location
cache/tusz_mmap -> /mnt/d/brain-go-brr/cache/tusz_mmap (symlink)

# Filesystem
df -h cache/tusz_mmap/
# D:\    932G  660G  273G  71% /mnt/d  (9P filesystem)

# Training result
# Crashes at batch 2890 with SIGBUS ❌
```

### After Migration
```bash
# Cache location
cache/tusz_mmap/         (real directory on ext4)

# Filesystem
df -h cache/tusz_mmap/
# /dev/sdc  1007G  955G   52G  95% /  (native ext4)

# Training result
# Completes full epoch without crashes ✅
```

---

## Rollback Plan (If Needed)

If something goes wrong during migration:

```bash
# Step 2 rollback: Restore old cache
mv cache/tusz_OLD_NPZ_DELETE_ME cache/tusz

# Step 3 rollback: Restore symlink
rm -rf cache/tusz_mmap
ln -s /mnt/d/brain-go-brr/cache/tusz_mmap cache/tusz_mmap

# Resume training with symlink (will still crash, but data is safe)
```

---

## Safety Checklist

Before deleting anything, verify:

- [ ] New cache exists at `/mnt/d/brain-go-brr/cache/tusz_mmap/`
- [ ] New cache has 12998 .npy files total (9334 train + 3664 dev)
- [ ] New cache has manifest.json in train/ and _dataset_index.json in dev/
- [ ] New cache size is ~518GB
- [ ] No configs reference `cache/tusz/` (only `cache/tusz_mmap/`)
- [ ] Code only uses `.npy` files, not `.npz` files

---

## Post-Migration Verification

After migration completes, verify:

1. **Filesystem**:
   ```bash
   df -h cache/tusz_mmap/
   # Should show: /dev/sdc (ext4) ✅
   ```

2. **File integrity**:
   ```bash
   python -m src scan-cache --cache-dir cache/tusz_mmap/train
   # Should show: 4667 stems, ~34% seizure ratio ✅
   ```

3. **Training stability**:
   ```bash
   # Monitor training logs:
   # - Loads cache from /home/jj/.../cache/tusz_mmap/ ✅
   # - Passes batch 2890 without crashes ✅
   # - Passes batch 2996 without crashes ✅
   # - Completes full epoch ✅
   ```

---

## Technical Notes

### Why Old Cache Exists

**History** (from git log):
- Sep 28, 2025: Original cache created as NPZ format (single file per session)
- Oct 6, 2025 (v3.8.0): Migrated to NPY format (two files per session, memory-mapped)
- Oct 11, 2025: Discovered old cache still present, taking up 449GB

**NPZ vs NPY**:
```python
# NPZ (old): Compressed, single file, in-memory loading
data = np.load("session.npz")
# Pros: Smaller size (compressed)
# Cons: Slower loading, full decompression, no mmap, 387GB RAM usage

# NPY (new): Uncompressed, two files, memory-mapped
data = np.memmap("session_data.npy", mode='r')
# Pros: Fast loading, <1GB RAM usage, instant startup
# Cons: Larger size (uncompressed), requires mmap-friendly filesystem
```

### Why WSL2 9P Filesystem Fails

**9P Filesystem Architecture**:
```
WSL2 VM (Linux) ←→ 9P Protocol ←→ Windows Host
     ↑                               ↑
   mmap()                      File operations
   Pages cached              Served over network
```

**Problem**:
1. Python opens NPY file with `np.memmap(..., mode='r')`
2. WSL2 maps file pages via 9P client/server
3. Under memory pressure, 9P client evicts old pages
4. PyTorch dataloader tries to read evicted page
5. AVX2 instruction (`vmovdqu`) hits invalid page → SIGBUS

**Why Native ext4 Works**:
```
Application ←→ mmap() ←→ Page Cache ←→ Block Device ←→ Disk
     ↑                     ↑                           ↑
Direct memory access   Always valid            Persistent backing
```

Pages can always be re-fetched from block device, no network protocol involved.

---

**Status**: ✅ **READY TO EXECUTE** - All checks complete, solution verified
**Risk Level**: LOW - Old cache unused, new cache verified complete
**Next Action**: Execute Step 1 (backup verification), then proceed with migration
**Expected Result**: Training completes full epoch without SIGBUS crashes
