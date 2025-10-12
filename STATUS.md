# Brain-Go-Brr v4.0.0 – Current Status

**Last Updated:** 2025-10-12
**Branch:** `feature/flash-linear-attention`
**Version:** v4.0.0 (FLA Production + WSL2 Fix)
**Deployment:** DUAL PRODUCTION STACKS – BiMamba2 (Modal A100, Epoch 3) + FLA (Local RTX 4090, Epoch 2)

---

## Production Readiness

**🟢 READY FOR MODAL A100 TRAINING – ZERO TECHNICAL DEBT**

- ✅ **P0/P1/P2/P3:** 0 issues (all debt resolved)
- 🟡 **P4/P5:** Optional ideas (post-training optimization only)

**Quality Verification (2025-10-07)**:
- `make q` → ✅ PASS (lint + format + mypy + config validation)
- `make test` → ✅ PASS (104 tests, 83.80% coverage)
- Cache validation → ✅ PASS (4667 train + 1832 dev NPY files)
- Manifest validation → ✅ PASS (100% NPY naming, 0 NPZ references)

**Policy:** Maintain zero debt before every major training run. Any new debt must be paid down immediately.

---

## Latest Improvements

### v4.0.0 - FLA Production + WSL2 Fix (October 12, 2025)

**MAJOR RELEASE**: Dual production stacks + critical WSL2 fix

**New Capabilities**:
- ✅ **MAJOR: FLA stack production-ready**: BiGatedDeltaNet fully validated, training successfully on local RTX 4090
- ✅ **MAJOR: WSL2 SIGBUS fix**: Local FLA training now works after cache migration to native ext4 filesystem
- ✅ **MAJOR: Dual training stacks**: BiMamba2 (Modal, Epoch 3) + FLA (Local, Epoch 2) running simultaneously
- ✅ **Critical WSL2 discovery**: Memory-mapped cache MUST be on native ext4, not Windows drives (/mnt/d/)
- ✅ **Crash verification**: FLA training validated past previous crash point (batch 5401 vs crash at 2890)

**Documentation**:
- ✅ Comprehensive WSL2 fix guide: `docs/08-operations/wsl2-sigbus-fix.md` (NEW)
- ✅ Incident analysis: `docs/archive_v4/` (SIGBUS, timeline, migration, audit)
- ✅ Quick ref updates: INSTALLATION.md, CLAUDE.md, CACHE.md, .gitkeep files

**Impact**:
- **Research Milestone**: Both SSM architectures (BiMamba2 + FLA) now training in production
- **Local Training Enabled**: WSL2 users can now train FLA locally (previously impossible)
- **A/B Comparison**: Empirical comparison of BiMamba2 vs GatedDeltaNet on full TUSZ dataset

### v3.11.0 - StatefulDataLoader & Mid-Epoch Resume (October 10, 2025)

**New Features**:
- ✅ **StatefulDataLoader Integration**: PyTorch official dataloader state management for exact mid-epoch checkpoint resume
- ✅ **DataLoader State Persistence**: Saves/restores exact batch position in checkpoints (eliminates 1-2h wasted compute per restart)
- ✅ **Pydantic v2 Warning Fix**: Clean `Annotated` pattern for forward references (zero warnings in production logs)
- ✅ **Backward Compatibility**: Old checkpoints still work (logs warning, restarts from epoch start)
- ✅ **Documentation Updates**: pyproject.toml, README, CLAUDE.md, STATUS.md, CHANGELOG.md all updated

**Impact**:
- **Cost Savings**: Additional $150+ saved per 100 epochs (on top of v3.10.0 savings)
- **Resume Precision**: Resumes at exact batch (e.g., batch 512/1283) instead of batch 0
- **Code Quality**: Zero Pydantic warnings, clean type annotations with `Annotated` pattern
- **Production Ready**: Deployed to Modal with immediate benefits

### v3.10.0 - Auto-Restart & Checkpoint Fix (October 10, 2025)

**New Features**:
- ✅ **Auto-Restart Training**: `train_auto_restart()` function with `modal.Period(hours=23)` for hands-free 100-epoch training
- ✅ **Checkpoint Resume Fix**: Changed `save_checkpoint(..., epoch + 1, ...)` to prevent re-training completed epochs (saves $672 over 12 restarts)
- ✅ **Modal 1.0 Migration**: Updated `concurrency_limit` → `max_containers` for future compatibility
- ✅ **Modal CLI Reference**: New `MODAL_CLI_REFERENCE.md` with all updated commands and migration notes

**Impact**:
- **Operational**: Zero manual interventions after initial setup (vs. 12× manual resume for 100 epochs)
- **Cost**: $616 net savings ($672 saved - $56 one-time waste)
- **Human Time**: 10 min total vs. 60 min manual resume interventions

### v3.9.2 - CI/CD Stability (October 9, 2025)

**Fixes**:
- ✅ Confirmed O(n) event extraction fix mathematically correct (training weights unaffected)
- ✅ Fixed FLA test skip logic (module-level flag check)
- ✅ Fixed memory test flake (deterministic array properties)
- ✅ Added dedicated FLA CI job for continuous validation
- ✅ Documentation cleanup (moved config docs to archive)

**Impact**: All CI passing reliably, FLA code path validated continuously.

### v3.9.1 - Validation OOM Fix (October 9, 2025)

**Critical OOM Fix (P0 BLOCKER RESOLVED)**:
- ✅ **Validation OOM Eliminated**: Disk-backed storage replaces 120GB RAM requirement with <5GB overhead
- ✅ **Dataset Fallback Bug Fixed**: Manifest-based BalancedSeizureDataset loading confirmed
- ✅ **Manifest Verified**: 61,616 training windows (34.2% seizure ratio) from 4667 NPY files
- ✅ **Modal Training Validated**: Epoch 1 completed with <5GB validation memory overhead
- ✅ **Production Ready**: Full 100-epoch training now possible without OOM crashes

**Files Changed**:
- `src/brain_brr/train/loop.py:592-658` - Disk-backed validation storage
- `src/brain_brr/data/datasets.py` - Fixed manifest validation (all 3 dataset classes)

### v3.9.0 - Production Training Baseline (October 8, 2025)
- ✅ **Bulletproof Checkpoints**: Atomic saves (temp + fsync + rename), AMP scaler capture, RNG state persistence
- ✅ **Timeout Guard**: 23h wall-clock limit with 1h safety margin, graceful exit before Modal kill
- ✅ **Comprehensive Validation**: Pre-training validation report, metrics pipeline verified from first principles
- ✅ **Test Suite Enhancement**: Manifest validation tests, checkpoint robustness tests, 75%+ coverage maintained
- ✅ **Full Training Resumed**: Modal A100-80GB (100 epochs, disk-backed validation fix verified Oct 9)
- ✅ **Documentation**: PRE_TRAINING_VALIDATION.md (root) + updated `docs/05-training/modal.md`; full historical analysis archived at `docs/archive_v1/MODAL_TRAINING_DIAGNOSTICS.md`

### FLA Research Comparison (October 8-9, 2025)
- ✅ Phase 0–2 COMPLETE: Edge, node, and dual-stream GDN smoke tests passed (Phase 2 medium validation technical success)
- ✅ Full-stack configs: `configs/local/phase2_both_gdn.yaml` + `configs/local/phase2_medium_gdn.yaml`
- ✅ Roadmap documented: See `FLA_ROADMAP.md` / `FLA_QUICK_REFERENCE.md`
- 🔄 Modal A/B Plan: BiMamba2 baseline training LIVE; FLA Modal config queued post-baseline
- 📊 Research Goal: Empirical comparison of BiMamba2 vs Gated DeltaNet architectures; both results publishable regardless of outcome

### v3.8.3 - Manifest Naming Cleanup (October 7, 2025)
- ✅ Regenerated train manifest: 303,990 windows across 4438 NPY files
- ✅ Regenerated dev manifest: 148,224 windows (7.7% natural seizure ratio)
- ✅ Removed all 11 `.replace("_windows", "")` string manipulation workarounds
- ✅ Simplified cache_utils.py and datasets.py for better maintainability
- ✅ Verification: 100% NPY naming, 0 NPZ references

### Previous Improvements

#### v3.8.2 - Zero Warnings (October 6, 2025)
- ✅ NumPy copy-on-read tensors (Balanced, Validation, EEGWindow) – no read-only warnings
- ✅ AMP scheduler guard (main loop + accumulation flush) for accurate LR schedule
- ✅ Verified 0 warnings in Modal training logs

#### v3.8.1 - Complete Tensor Safety (October 6, 2025)
- ✅ Completed tensor safety across all 3 dataset classes
- ✅ EEGWindowDataset hardened with proper .clone() calls

#### v3.8.0 - NPZ Cache Cleanup (October 6, 2025)
- ✅ Cleaned 3 stray NPZ files from Modal cache
- ✅ Fixed datasets.py NPZ creation bug
- ✅ Fixed all type annotations (WandBRun, Console instead of Any)
- ✅ Extracted duplicate `_load_cache_for_worker` (120 lines eliminated)

#### Architecture (v3.3.0-v3.4.1)
- ✅ V3 dual-stream with edge similarity clamping
- ✅ Dynamic Laplacian PE (time-evolving graph structure)
- ✅ Detached eigenvectors (prevents gradient explosion)
- ✅ 3-tier NaN protection
- ✅ Unique Triton cache dirs (prevents XID 31 GPU crashes)

---

## 🚨 IMPORTANT - Auto-Restart Instructions

**After current manual resume completes Epoch 2 (~23h from 16:54 EDT Oct 10)**:

1. **Verify Epoch 2 completed successfully** (check Modal logs + W&B)

2. **Deploy auto-restart function** (one-time setup):
   ```bash
   modal deploy deploy/modal/app.py
   ```

3. **Start hands-free auto-restart** (Epochs 3-100, zero manual intervention):
   ```bash
   modal run --detach deploy/modal/app.py \
     --action schedule-training \
     --config configs/modal/train_bimamba.yaml
   ```

4. **Monitor** (optional):
   ```bash
   modal app list                    # See active scheduled job
   modal app logs brain-go-brr-v2    # Stream logs
   modal app stop brain-go-brr-v2    # Stop if needed
   ```

**Current vs. Auto-Restart**:
- ✅ **Current (manual)**: Tests checkpoint fixes, runs once, stops after Epoch 2
- 🔄 **Auto-restart**: Restarts every 23h via `modal.Period(hours=23)`, runs to epoch 100
- 🎉 **Result**: Zero manual resume from Epoch 3 → 100!

---

## Current Deployment

### DUAL PRODUCTION STACKS (v4.0.0)

**BiMamba2 - Modal Full Training (LIVE)**:
- Launch: Oct 10, 2025
- Config: 100 epochs, batch_size=48, A100-80GB, mixed_precision=true
- Cache: 4667 train + 1832 dev NPY files on Modal SSD volume
- Status: ✅ **EPOCH 3** - Progressing normally with StatefulDataLoader
- Stack: TCN + BiMamba2 + GNN + Dynamic LPE

**FLA - Local Full Training (LIVE)**:
- Launch: Oct 11, 2025 (after SIGBUS fix)
- Config: 100 epochs, batch_size=8, RTX 4090 (24GB VRAM), mixed_precision=false
- Cache: 4667 train + 1832 dev NPY files on native ext4 filesystem (WSL2)
- Status: ✅ **EPOCH 2** - Stable past previous crash point (batch 5401 vs crash at 2890)
- Stack: TCN + BiGatedDeltaNet (FLA) + GNN + Dynamic LPE

**Next Steps**:
1. Continue monitoring both training runs
2. Enable auto-restart for BiMamba2 if manual resume needed
3. Let FLA complete locally to gather empirical results
4. Compare final metrics: BiMamba2 vs FLA performance on full TUSZ dataset

---

## Outstanding Items

**Active Debt**: ✅ **ZERO** - All technical debt resolved (October 11, 2025)
- ✅ **P3-1 RESOLVED**: Pydantic Field warnings (schemas use correct `Annotated` pattern, local v2.11.9 has zero warnings)
- ✅ **P3-2 RESOLVED**: .gitkeep files added to 4 directories for OSS contributor clarity

**Optional Improvements** (post-training only):
- Profile `.item()` calls if profiling shows >1% GPU sync time
- Consider detector refactor if future features reduce readability

---

## Validation Checklist (should stay green)

```bash
make q                 # Lint + format + mypy + config validation
make test              # Full test suite with coverage
make test-performance  # GPU performance tests
```

---

## Quick Facts

**Training Commands**:
```bash
# BiMamba2 smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml

# FLA smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml

# BiMamba2 full training - Manual (100 epochs, ~100 hours)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# BiMamba2 full training - Auto-Restart (hands-free, recommended)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml

# FLA full training (100 epochs, ~100 hours)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_fla.yaml
```

**Architecture**:
- TCN: 8 layers, channels [64,128,256,512], stride_down=16
- BiMamba: 6 layers, d_model=64 per electrode, O(N) complexity
- GNN: SSGConv, α=0.05, 2 layers, Dynamic LPE (k=16)
- Fusion: Multi-head gated fusion (4 heads)
- Total: 31M parameters

**Cache System**:
- Format: NPY mmap (memory-efficient, <1 GB RAM vs 387 GB for NPZ)
- Modal: 4667 train + 1832 dev files at `/results/cache/tusz_mmap/`
- Startup: Manifest-based (99.6% faster than NPZ scan)
- Strategy: Read-only datasets, populate_cache sole writer

**Code Quality**:
- 65 source files, 83.80% test coverage
- 104 tests (64 integration + 40 clinical)
- Zero lint/format/type errors
- Zero active technical debt

---

Keep this document in sync with every deployment or training cycle.
