# Technical Debt - Resolved Issues (Archive)

**Purpose**: Historical record of technical debt that has been fully resolved.
**Status**: ✅ ALL RESOLVED - For reference only

This document archives previously tracked technical debt that has been completely resolved. For current open debt, see `/TECHNICAL_DEBT.md` in the project root.

---

## ✅ P0: RESOLVED BLOCKERS (October 6, 2025)

### P0-1: ValidationDataset Cannot Find Cache Files ✅ **FIXED**

**Root Cause**: Manifest references legacy NPZ naming, actual files use NPY naming

**Resolution** (v3.8.0):
- Added `cache_file_exists()` helper to ValidationDataset.__init__
- Helper translates legacy `*_windows` manifest naming to actual `*_data.npy` files
- ValidationDataset now correctly finds all 148,224 windows

---

### P0-2: Read-Only NumPy Array Warnings ✅ **FIXED**

**Root Cause**: NPY files opened with `mmap_mode='r'`, PyTorch tensors backed by read-only memory

**Resolution** (v3.8.1):
- Added `.clone()` to all three datasets' __getitem__ methods
- Creates writable tensor copies (~5-10ms CPU overhead per batch)
- Eliminates entire class of potential in-place operation bugs

---

### P0-3: LR Scheduler Order Warning ✅ **VERIFIED CORRECT**

**Root Cause**: PyTorch quirk during scheduler creation, not actual stepping order

**Resolution** (v3.8.2):
- Verified actual stepping order is correct (no fix needed)
- Added minimal targeted suppression at scheduler creation only
- Removed broad suppression from training loop

---

## ✅ P1: RESOLVED (October 7, 2025)

### P1-1: Manifest Cache File References Use Legacy Naming ✅ **FIXED**

**Root Cause**: Manifests referenced legacy NPZ naming, required workarounds throughout codebase

**Resolution** (v3.8.3):
- Regenerated both train/dev manifests with correct `*_data.npy` naming
- Removed all 11 `stem.replace("_windows", "")` workarounds from codebase
- Simplified cache_utils.py and datasets.py
- 100% NPY naming, 0 NPZ references

---

## ✅ P2: RESOLVED (October 6-11, 2025)

### P2-1: YAML Config Output Directory Conflict ✅ **FIXED**

**Root Cause**: Both BiMamba2 and FLA configs used same output directory, risking checkpoint contamination

**Resolution** (v3.11.0 - October 11, 2025):
- Changed FLA configs ONLY (BiMamba2 configs unchanged, already training)
- Modal FLA: `/results/v3_full_training` → `/results/v3_fla_training`
- Local FLA: `results/local_training` → `results/local_fla_training`
- BiMamba2 configs unchanged (already in use with checkpoints)
- Modal persistent volume auto-creates new directories at runtime
- Verified separation with grep verification command

**Why Critical**:
- Different model architectures (BiMamba2 vs GatedDeltaNet) have incompatible checkpoints
- Wrong config resume → checkpoint contamination → silent failures
- Happened once (Oct 11), killed immediately, no damage
- Now impossible to accidentally mix checkpoints

---

## ✅ P2 (HISTORICAL): RESOLVED (October 6-8, 2025)

### P2-1: psutil Swap Memory Warning ✅ **SUPPRESSED**

**Root Cause**: Container environment doesn't have `/proc/vmstat`

**Resolution** (v3.8.0): Suppressed warning, no functional impact

---

### P2-2: Code Duplication Between Datasets ✅ **RESOLVED**

**Root Cause**: Repeated patterns across three dataset classes

**Resolution** (v3.8.0): Refactored common patterns, improved maintainability

---

### P2-3: Type Safety Issues ✅ **RESOLVED**

**Root Cause**: Missing type hints in several modules

**Resolution** (v3.8.0): Added comprehensive type hints, mypy passing

---

## ✅ P3: RESOLVED (October 7, 2025)

### P3-1: Manifest Naming Mismatch ✅ **FIXED**

**Root Cause**: Same as P1-1 (promoted from P3 to P1)

**Resolution** (v3.8.3): See P1-1 above - complete manifest regeneration

---

## 🎉 Complete Resolution Timeline

- **v3.8.0** (Oct 6): Resolved NPZ contamination (P0), code duplication (P2), type safety (P2)
- **v3.8.1** (Oct 6): Completed tensor safety (P0-2), verified scheduler order (P0-3)
- **v3.8.2** (Oct 6): Eliminated all training warnings with professional PyTorch patterns
- **v3.8.3** (Oct 7): Manifest naming cleanup complete (P1/P3) → **ZERO DEBT ACHIEVED**
- **v3.9.0** (Oct 8): Bulletproof checkpoints + timeout guard → **PRODUCTION BASELINE**
- **v3.9.1** (Oct 9): Validation OOM fix → **MODAL TRAINING STABLE**
- **v3.10.0** (Oct 10): Auto-restart + three checkpoint fixes → **PRODUCTION READY**
- **v3.11.0** (Oct 11): StatefulDataLoader integration + YAML config separation (P2-1) → **HANDS-FREE TRAINING**

---

**Archive Date**: October 11, 2025
**Status**: All items in this archive have been fully resolved
**Current Debt**: See `/TECHNICAL_DEBT.md` for open items
