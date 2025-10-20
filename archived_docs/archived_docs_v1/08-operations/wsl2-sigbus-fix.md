# WSL2 SIGBUS Fix Guide

**Last Updated**: October 12, 2025  
**Scope**: Local RTX 4090 training on Windows + WSL2 with the memory-mapped TUSZ cache

---

## 1. Summary

- **Symptom**: Flash Linear Attention (FLA) training crashes around batch 2,800–3,100 with `Signal 7 (SIGBUS)` or the Python process dies without traceback.
- **Root Cause**: The mmap cache (`*_data.npy`, `*_labels.npy`) lived on a Windows disk (`/mnt/c|d/`). WSL2’s 9P network filesystem evicts pages under memory pressure. AVX2 copy instructions inside FLA hit invalid pages → SIGBUS.
- **Fix**: Move the mmap cache to a native ext4 volume inside the WSL2 virtual disk. Raw EDF inputs can remain on Windows drives.
- **Status**: ✅ Verified — after migrating 518 GB of cache files to ext4, FLA training progressed past batch 5,400 with no crashes (previous failure point: 2,890).

See also: `docs/archive_v4/` for the complete investigation and audit trail.

---

## 2. Action Plan (Do This Once)

### Step 1 — Confirm Crash Signature
```bash
tmux attach -t train-fla   # or view logs
# Look for: "Signal 7 (SIGBUS)" near batch ~2900
```

### Step 2 — Verify Cache Location
```bash
readlink -f cache/tusz_mmap
df -h cache/tusz_mmap
# ✅ Expected: /dev/sdX … ext4
# ❌ Problem: /mnt/c/... or 9p filesystem
```

If the path resolves to `/mnt/c` or `/mnt/d`, you are using the Windows filesystem and MUST migrate.

### Step 3 — Create Native ext4 Storage (One-Time)
1. Open Windows “Manage Disks and Volumes”, shrink an existing volume, and create a new VHD (recommended ≥600 GB).
2. Mount the VHD inside WSL2:
   ```bash
   wsl --shutdown
   wsl --mount <path-to-vhd> --bare
   ```
3. Inside WSL2, format and mount:
   ```bash
   sudo mkfs.ext4 /dev/sdc            # replace with the device shown by `lsblk`
   sudo mkdir -p /mnt/cache_ext4
   sudo mount /dev/sdc /mnt/cache_ext4
   sudo chown $USER:$USER /mnt/cache_ext4
   ```
4. Persist the mount by adding an entry to `/etc/fstab`:
   ```
   /dev/sdc  /mnt/cache_ext4  ext4  defaults  0  2
   ```

### Step 4 — Migrate the Cache
```bash
# Stop training first.
rsync -avh --progress --delete /mnt/d/brain-go-brr/cache/tusz_mmap/ \
  /mnt/cache_ext4/tusz_mmap/

ln -sfn /mnt/cache_ext4/tusz_mmap cache/tusz_mmap
```

### Step 5 — Verify Integrity
- `find cache/tusz_mmap/train -maxdepth 1 -type f` → expect manifest + `_data.npy/_labels.npy` pairs only.
- Verify counts: `ls -1 cache/tusz_mmap/train/*_data.npy | wc -l` should show 4667.

### Step 6 — Resume Training
```bash
tmux new -s train-fla
make train-fla  # or python -m src train configs/local/train_fla.yaml --resume
```
Training should now run well past batch 3,000 without SIGBUS.

---

## 3. Verification Checklist

- ✅ `df -h cache/tusz_mmap` shows an ext4 device (e.g., `/dev/sdc`).
- ✅ FLA training reaches batch ≥5,400 without crashing.
- ✅ Modal/WSL2 logs show normal heartbeats and gradient statistics.
- ✅ `cache/tusz_mmap` contains only NPY pairs and manifest files (no NPZ artifacts).

Optional: capture logs to `/tmp/phase2_sigbus_fix.log` and link in status docs.

---

## 4. Investigation Evidence (Forensics)

This section preserves the complete investigation trail for future reference.

### Dual Root Causes (October 11, 2025)

**Discovery**: Two independent bugs both caused SIGBUS around batch 2900:

1. **Driver Bug** (Fixed Oct 11 morning):
   - NVIDIA driver 572.16 (January 2025) had widespread RTX 4090 stability issues
   - Upgraded to 581.42 (October 2025) → First crash pattern eliminated
   - Community reports confirmed 572.xx series problems; driver 576.02 (April 2025) resolved 40+ crash issues

2. **Filesystem Bug** (Fixed Oct 11 afternoon):
   - Memory-mapped cache lived on `/mnt/d/` (Windows partition via WSL2 9P filesystem)
   - Under memory pressure (~2 hours training), 9P client evicted mmap pages
   - AVX2 copy instructions in FLA hit invalid pages → SIGBUS
   - **Why missed initially**: Driver bug was happening first every time, masking filesystem bug

### Kernel Evidence (dmesg)

```
[62361.200410] potentially unexpected fatal signal 7.
[62361.200642] Code: c5 fe 6f 06 ...  (AVX2 vmovdqu instruction)
[62361.200661] RDX: 000000000011d000  (copying ~1.1MB)
```

- Signal 7 = SIGBUS (bus error)
- AVX2 `vmovdqu` = vectorized memory copy (likely NumPy loading from mmap)
- Non-faulting AVX2 instruction cannot trigger page fault handler → immediate SIGBUS when page invalid

### Cache Filesystem Verification

```bash
# Before fix
$ ls -la cache/tusz_mmap
lrwxrwxrwx  1 jj jj  35 Oct  5 17:09 cache/tusz_mmap -> /mnt/d/brain-go-brr/cache/tusz_mmap

$ df -h cache/tusz_mmap/
D:\    932G  660G  273G  71% /mnt/d    # ← 9P filesystem (problem!)

# After fix (cache migrated to ext4)
$ df -h cache/tusz_mmap/
/dev/sdc  1007G  823G  134G  87% /     # ← ext4 (solution)
```

### Space Discovery

**Critical Finding**: Old NPZ cache (449GB) unused since v3.8.0 migration allowed migration without disk expansion:

```bash
$ du -sh cache/tusz         # Old NPZ format
449G

$ du -sh /mnt/d/.../tusz_mmap  # New NPY format
518G

# Space calculation:
#   Current WSL2 free: 134GB
#   + Delete old cache: 449GB
#   = Available: 583GB
#   - New cache needs: 518GB
#   = Final free: 65GB ✅
```

**Verification**: No runtime code creates or requires NPZ files (legacy read-only compatibility only):
```bash
$ grep -rn "\.save.*npz\|savez\|np\.savez" src/
# (empty) ✅
```

### Crash Timeline

| Time | Batch | Driver | Root Cause | Action |
|------|-------|--------|------------|--------|
| Oct 11, 01:23 | 2996 | 572.16 | Driver bug | Upgraded to 581.42 |
| Oct 11, 15:54 | 2890 | 581.42 | Filesystem bug | Migrated cache to ext4 |
| Post-migration | 5401+ | 581.42 | None | Training stable ✅ |

**Verification**: After both fixes, FLA training progressed past batch 5,400 (previous failure point: 2,890) with zero SIGBUS events.

---

## 5. Frequently Asked Questions

**Q: Can the raw EDF dataset stay on `/mnt/d/`?**  
Yes. Sequential EDF reads tolerate 9P; only the mmap cache requires ext4.

**Q: Do BiMamba2 runs need the migration?**  
BiMamba2 kernels are less sensitive but share the same cache. Moving once protects both stacks.

**Q: Why not use NTFS from WSL2 directly?**  
NTFS → WSL2 via 9P lacks stable mmap semantics. Ext4 inside the WSL2 VM guarantees page pinning.

**Q: How large is the cache?**  
~518 GB for TUSZ train+dev. Ensure the ext4 volume has ≥600 GB free for checkpoints and manifests.

**Q: Do I need to rebuild the cache?**  
No. `rsync` preserves the mmap files. Rebuild only if `validate-cache` reports corruption.

---

## 6. Related Documentation

- `INSTALLATION.md` — GPU stack requirements and driver checklist.
- `docs/05-training/local.md` — local training pipeline.
- `docs/08-operations/troubleshooting.md` — SIGBUS troubleshooting entries.
- `docs/archive_v4/` — full investigation, timelines, and audit evidence.

---

**Outcome**: With the cache on ext4, local FLA training is stable, enabling the v4.0.0 dual-stack milestone (BiMamba2 + FLA). Delete or archive the old Windows-hosted cache to reclaim disk space once you confirm the migration.
