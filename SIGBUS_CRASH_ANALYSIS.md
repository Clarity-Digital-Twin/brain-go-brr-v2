# SIGBUS Crash Analysis - Memory-Mapped Cache on WSL2

**Date**: October 11, 2025
**Status**: ⚠️ ACTIVE ISSUE - Cache on Windows partition causes crashes
**Severity**: P1 - Blocks local FLA training
**Driver Version**: 581.42 (latest, not driver bug)

---

## Executive Summary

**Crash Pattern**: Training dies with `Bus error (core dumped)` around batch 2890-3000 during local FLA training.

**Root Cause**: Memory-mapped NPY cache files on Windows partition (`/mnt/d/`) accessed via WSL2's 9P network filesystem. AVX2 memory copy instructions get SIGBUS when mmap backing pages become invalid.

**Solution**: Move cache to native Linux filesystem (ext4/btrfs) on WSL2 VM disk.

---

## Current Crash Details (October 11, 2025)

```
Time: 15:54:14
Batch: 2890/7702 (38% through epoch)
Signal: 7 (SIGBUS - bus error)
Driver: 581.42 (latest stable - NOT driver bug)
GPU: RTX 4090 (healthy, 39°C, 1% utilization)
Memory: 23.50GB RAM used / 25.44GB available
Training: Healthy metrics until instant crash
Loss: 0.0134 (normal)
```

**Kernel Log**:
```
[62361.200410] potentially unexpected fatal signal 7.
[62361.200566] RIP: 0033:0x7f843ff8984d
[62361.200642] Code: c5 fe 6f 06 ...  (AVX2 vmovdqu instruction)
[62361.200661] RDX: 000000000011d000  (copying ~1.1MB)
[62361.203996] WSL (363994): Capturing crash for pid: 119531
[62361.203998] , signal: 7, port: 50005
```

**Disassembly**: `c5 fe 6f 06` = `vmovdqu ymm0, ymmword ptr [rsi]` (AVX2 memory load)

---

## Root Cause Analysis

### 1. Memory-Mapped Files on Windows Partition

**Cache Location**: `/mnt/d/brain-go-brr-v2/cache/tusz_mmap/`
```bash
df -h cache/tusz_mmap/
# D:\    932G  660G  273G  71% /mnt/d
```

**Problem**: WSL2 mounts Windows drives via 9P network filesystem:
- 9P filesystem has poor mmap support
- Memory-mapped pages can be invalidated by the 9P client/server
- AVX2 instructions accessing invalidated pages trigger SIGBUS

### 2. WSL2 Memory Allocation Failures

**From dmesg**:
```
misc dxg: dxgk: dxgvmb_send_create_allocation: send_create_allocation failed ffffffb5
misc dxg: dxgk: dxgkio_create_allocation: Ioctl failed: -75
```

These are DirectX GPU memory allocation errors (error -75 = EOVERFLOW/EREMOTE). WSL2's GPU virtualization layer (dxgk) is struggling under sustained training load.

### 3. Semaphore Leak Warning

**User Warning**:
```python
/home/jj/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

This appears immediately before crash - suggests multiprocessing cleanup issues when process receives SIGBUS.

---

## Why This Happens

**Timeline**:
1. Training starts, opens NPY files with `np.memmap(mode='r')`
2. Python/NumPy calls `mmap()` syscall to map files into memory
3. WSL2's 9P filesystem provides backing pages via network protocol
4. Training runs for ~2 hours (~2890 batches)
5. Memory pressure or 9P client/server timeout causes page invalidation
6. PyTorch dataloader tries to read from mmap'd array
7. AVX2 instruction (`vmovdqu`) hits invalid page → SIGBUS
8. Python interpreter killed instantly

**Why it happens around batch 2890-3000**:
- ~2 hours of training elapsed
- Page cache pressure builds up
- WSL2 9P filesystem timeout/cleanup kicks in
- First batch that tries to read evicted page crashes

---

## Historical Context

### Previous Crash (Documented in troubleshooting.md:57-62)

**Symptom**: SIGBUS crash near batch 2900-3100
**Diagnosed Root Cause**: NVIDIA driver 572.xx bugs
**Fix**: Upgrade to driver 581.42
**Status**: User already has 581.42 ✅

### Current Crash (October 11, 2025)

**Symptom**: Same SIGBUS crash pattern (batch 2890)
**Current Driver**: 581.42 (latest stable) ✅
**New Root Cause**: Cache on Windows partition + WSL2 9P filesystem
**Fix Required**: Move cache to native Linux filesystem

---

## Solution

### Option 1: Move Cache to WSL2 Native Filesystem (RECOMMENDED)

**Pros**:
- Native ext4 filesystem with full mmap support
- No 9P network overhead
- Stable under sustained I/O

**Cons**:
- Limited disk space (~250GB WSL2 VM default)
- Need to copy ~50GB cache

**Steps**:
```bash
# 1. Check available space
df -h ~/proj/brain-go-brr-v2/cache/

# 2. Copy cache to native Linux filesystem
mkdir -p ~/proj/brain-go-brr-v2/cache/tusz_mmap/
cp -r /mnt/d/brain-go-brr-v2/cache/tusz_mmap/* ~/proj/brain-go-brr-v2/cache/tusz_mmap/

# 3. Update symlinks or configs to point to new location

# 4. Resume training
tmux new -s train-fla
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla.yaml --resume
```

### Option 2: Use Dedicated ext4 Partition on Windows Drive

**Pros**:
- Larger storage (can use full Windows drive space)
- Native ext4 with full mmap support
- Keep data on fast NVMe drive

**Cons**:
- Requires WSL2 mount commands
- More complex setup

**Steps**:
```bash
# 1. Create virtual disk on Windows
# In PowerShell (Administrator):
# wsl --mount --vhd D:\wsl-ext4-cache.vhdx --bare

# 2. Format as ext4 (in WSL2)
sudo mkfs.ext4 /dev/sdX

# 3. Mount permanently
sudo mkdir -p /mnt/cache
sudo mount /dev/sdX /mnt/cache
echo '/dev/sdX /mnt/cache ext4 defaults 0 0' | sudo tee -a /etc/fstab

# 4. Move cache
sudo cp -r /mnt/d/brain-go-brr-v2/cache/tusz_mmap/* /mnt/cache/
sudo chown -R $(whoami):$(whoami) /mnt/cache/

# 5. Update config to use /mnt/cache/
```

### Option 3: Disable Memory Mapping (FALLBACK)

**Pros**:
- Works on any filesystem
- No SIGBUS crashes

**Cons**:
- ~10-20x slower data loading
- Higher RAM usage (~5-10GB instead of <1GB)

**Implementation**:
Modify `datasets.py` to use `np.load()` instead of `np.memmap()`:
```python
# OLD (memory-mapped)
data = np.memmap(cache_path, dtype=np.float32, mode='r', shape=shape)

# NEW (in-memory)
data = np.load(cache_path)
```

---

## Verification

### Before Fix
```bash
# Check cache location
ls -la cache/tusz_mmap/ | head
# Should see: lrwxrwxrwx ... cache/tusz_mmap -> /mnt/d/...

# Check filesystem
df -h cache/tusz_mmap/
# Should see: /mnt/d (9P filesystem)
```

### After Fix
```bash
# Check cache location
ls -la cache/tusz_mmap/ | head
# Should see: drwxr-xr-x ... (real directory, not symlink)

# Check filesystem
df -h cache/tusz_mmap/
# Should see: /dev/sda or similar (native ext4)

# Resume training
tmux new -s train-fla
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla.yaml --resume

# Monitor - should run past batch 2890 without crashes
```

---

## Technical Deep Dive

### Why AVX2 and mmap Don't Mix on 9P

**AVX2 Memory Access Pattern**:
```assembly
vmovdqu ymm0, ymmword ptr [rsi]  ; Load 32 bytes with AVX2
```

**Requirements**:
- Memory must be resident (not paged out)
- Backing file must be accessible
- Page table entries must be valid

**9P Filesystem Behavior**:
1. Client (WSL2) requests file pages from server (Windows)
2. Server provides pages over network protocol
3. Client caches pages in page cache
4. Under pressure, client may evict pages
5. Client assumes it can re-fetch pages on-demand

**The Problem**:
- AVX2 instruction is **non-faulting** - doesn't trigger page fault handler
- If page is invalid, CPU generates SIGBUS immediately
- No chance for 9P client to re-fetch page
- Process killed instantly

**Why Regular Filesystems Work**:
- ext4/btrfs use block devices with direct page cache
- Pages backed by actual disk blocks, not network protocol
- Kernel can always re-fetch evicted pages via block I/O
- Page faults handled correctly

---

## Related Documentation

- `docs/08-operations/troubleshooting.md:57-62` - Original SIGBUS diagnosis (driver bug)
- `docs/archive_v3/NVIDIA_DRIVER_FIX_UPGRADE.md` - Driver upgrade guide (completed)
- This document: Root cause analysis for cache filesystem issue

---

## Action Items

**Immediate** (P1):
- [ ] Verify cache location: `df -h cache/tusz_mmap/`
- [ ] If on `/mnt/d/`, move to native Linux filesystem
- [ ] Resume training and verify it runs past batch 2890
- [ ] Document actual filesystem layout in CLAUDE.md

**Documentation** (P2):
- [ ] Update troubleshooting.md with filesystem-specific guidance
- [ ] Add cache filesystem check to pre-flight checklist
- [ ] Document recommended cache locations for local training

**Long-term** (P3):
- [ ] Consider adding filesystem check to training startup
- [ ] Add warning if cache is on /mnt/ partition
- [ ] Profile mmap vs in-memory loading performance

---

## Prevention

**Before Local Training**:
```bash
# Check cache filesystem
df -h cache/tusz_mmap/

# If /mnt/, move to native Linux filesystem FIRST
if [[ $(df cache/tusz_mmap/ | tail -1 | grep -c /mnt/) -eq 1 ]]; then
    echo "❌ ERROR: Cache on Windows partition - will crash!"
    echo "Move to native Linux filesystem first"
    exit 1
fi
```

**Recommended Cache Locations**:
- ✅ `/home/jj/proj/brain-go-brr-v2/cache/` (WSL2 native ext4)
- ✅ `/mnt/wsl/cache/` (if mounted as ext4 VHD)
- ❌ `/mnt/c/`, `/mnt/d/`, `/mnt/e/` (Windows partitions via 9P)

---

**Status**: ⚠️ UNRESOLVED - Requires cache migration to native filesystem
**Next Step**: Move cache from `/mnt/d/` to WSL2 native filesystem, resume training
**Expected Result**: Training should complete full epoch without SIGBUS crashes
