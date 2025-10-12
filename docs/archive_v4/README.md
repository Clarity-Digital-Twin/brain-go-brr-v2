# Archive v4: WSL2 SIGBUS Investigation

**Status**: ✅ RESOLVED - Critical WSL2 fix
**Archive Date**: October 12, 2025
**Context**: SIGBUS crash investigation and resolution (v4.0.0)

## Contents

This archive documents the **critical WSL2 SIGBUS bug** discovered during local FLA training. The production guide now lives in `docs/08-operations/wsl2-sigbus-fix.md`; keep this folder for the raw investigation notes and evidence trail.

### Investigation Documents

1. **SIGBUS_CRASH_ANALYSIS.md** - Root cause analysis
   - **Bug**: Memory-mapped NPY cache on Windows drives (/mnt/d/) via WSL2 9P filesystem
   - **Cause**: Page evictions under memory pressure → AVX2 instructions hit invalid pages → SIGBUS
   - **Impact**: FLA training crashed after ~2 hours (batch ~2890)
   - **Status**: ✅ Diagnosed

2. **CRASH_TIMELINE_ANALYSIS.md** - Detailed timeline of crashes
   - **Pattern**: Consistent crash at ~2h mark across multiple attempts
   - **Smoking gun**: dmesg showed page fault + segfault
   - **Evidence**: Training always crashed at same approximate point
   - **Status**: ✅ Analyzed

3. **CACHE_MIGRATION_PLAN.md** - Migration strategy
   - **Solution**: Move cache from Windows drive to native ext4 filesystem
   - **Steps**: rsync cache, verify integrity, update symlinks
   - **Result**: 518GB cache migrated successfully
   - **Status**: ✅ Executed

4. **AUDIT_FINDINGS.md** - Comprehensive cache audit
   - **Discovery**: Windows drives via WSL2 9P don't support mmap properly
   - **Analysis**: Raw EDF files (sequential reads) are safe on Windows drives
   - **Decision**: Only cache needs native ext4, raw data can stay on /mnt/d/
   - **Status**: ✅ Complete

## Impact Summary

This fix enabled:
- ✅ **Local FLA training works**: No more SIGBUS crashes
- ✅ **Verified past batch 2890**: Training reached batch 5401 (2511 batches past crash point!)
- ✅ **Production FLA stack**: Both BiMamba2 and FLA now training simultaneously
- ✅ **v4.0.0 milestone**: Dual-stack production capability

## The Fix

**Critical requirement**: Memory-mapped cache MUST be on native Linux filesystem (ext4/btrfs) inside WSL2 VM.

**DO NOT** use Windows drives (`/mnt/c/`, `/mnt/d/`) for memory-mapped files - WSL2's 9P network filesystem has poor mmap support.

**Documentation added**:
- INSTALLATION.md: Section 6 - WSL2 Cache Location
- CLAUDE.md: Common Issues table entry
- CACHE.md: WSL2 critical note
- cache/.gitkeep: WSL2 warning
- data_ext4/.gitkeep: Clarification (raw EDF files are safe)

## Current Status

**FIXED** - FLA training verified stable on local RTX 4090:
- Training reached Epoch 2, batch 940 (far past previous crash point)
- Modal BiMamba2 training also proceeding normally (Epoch 3)
- Both stacks now in production

## Current Documentation

For WSL2-specific guidance, see:
- **NEW**: `docs/08-operations/wsl2-sigbus-fix.md` (consolidated guide)
- Installation: `INSTALLATION.md` (Section 6)
- Quick ref: `CLAUDE.md` (Common Issues)
- Operations: `docs/08-operations/wsl2-notes.md`
