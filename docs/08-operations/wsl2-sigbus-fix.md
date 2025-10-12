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
- `find cache/tusz_mmap -maxdepth 1 -type f` → expect manifest + `_data.npy/_labels.npy` pairs only.
- `python -m src validate-cache --cache-dir cache/tusz_mmap/train` (optional sanity check).

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

## 4. Frequently Asked Questions

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

## 5. Related Documentation

- `INSTALLATION.md` — GPU stack requirements and driver checklist.
- `docs/01-installation/gpu-stack.md` — version matrix and WSL2 precautions.
- `docs/05-training/local.md` — local training pipeline (update to reflect ext4 requirement).
- `docs/08-operations/troubleshooting.md` — add SIGBUS troubleshooting entry pointing here.
- `docs/archive_v4/` — full investigation, timelines, and audit evidence.

---

**Outcome**: With the cache on ext4, local FLA training is stable, enabling the v4.0.0 dual-stack milestone (BiMamba2 + FLA). Delete or archive the old Windows-hosted cache to reclaim disk space once you confirm the migration.
