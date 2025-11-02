# Brain-Go-Brr v4.2.0 – Current Status

**Last Updated:** 2025-11-01
**Branch:** `development`
**Version:** v4.2.0 (SGDR Scheduler + Experimental Pipeline)
**Deployment:** FLA FOCUS – Baseline (Epoch 30+, plateaued), Exp4 ready to launch

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

### v4.2.0 - SGDR Scheduler + Experimental Pipeline (November 1, 2025)

**FEATURE RELEASE**: Cyclic learning rate scheduler + validated experimental framework

**New Capabilities**:
- ✅ **SGDR (Cosine Restarts) Scheduler**: Cyclic LR with warm restarts to escape local minima
  - Implements Loshchilov & Hutter (2017) SGDR with proper gradient accumulation awareness
  - Config parameters: `t_initial` (cycle length), `t_mult` (cycle multiplier), `eta_min` (min LR)
  - Fixed 4 critical bugs from initial implementation (agent validation)
  - Test coverage: `test_scheduler_cosine_restarts` validates warmup and restart behavior
- ✅ **Experimental Framework**: Validated hyperparameter experiment pipeline
  - Exp1 (Regularization): Completed, FAILED (-2.7% vs baseline 0.284)
  - Exp4 (Cyclic LR): Ready to launch after baseline early stop
  - Isolated output directories for clean A/B testing
- ✅ **Baseline Progress**: FLA training reached epoch 30+, plateaued at 0.257
  - Best: 0.284 @ epoch 9, 0.95 TAES
  - Plateau: 0.257 for 13 epochs (17-29), patience 13/20
  - Early stop expected: Epoch 36

**Documentation**:
- ✅ EXPERIMENTAL_PLAN.md: Concise experiment tracking (506→121 lines)
- ✅ README.md: Updated current status (7→3 bullets)
- ✅ BASELINE_METRICS.md: Epochs 1-29 tracked with plateau analysis

**Impact**:
- **Research Tools**: SGDR scheduler enables local minimum escape strategies
- **Validated Pipeline**: Exp1 proved regularization hypothesis false, model not overfitting
- **Next Step**: Launch Exp4 to test SGDR escape from 0.257 plateau

### v4.1.0 - NEDC Evaluation Pipeline + Documentation Optimization (October 20, 2025)

**FEATURE RELEASE**: Official clinical evaluation + agent-optimized documentation

**New Capabilities**:
- ✅ **NEDC Evaluation Pipeline**: Official TUSZ evaluation via nedc-bench integration
  - NEDCScorer wrapper (202 lines) with automatic csv_bi validation
  - CLI integration: `python -m src evaluate --checkpoint <path> --nedc`
  - Eval cache building: `python -m src build-cache eval`
  - 277 lines of tests (231 unit + 46 integration)
- ✅ **Documentation Optimization**: Agent-optimized for 2025 best practices
  - CLAUDE.md: 537→372 lines (-65% tokens, -66% context waste)
  - AGENTS.md: Updated for OpenAI Codex compatibility
  - 100% accurate project tree structure
  - Modal cache operations (populate-cache, check-cache, clean_stray_npz)
- ✅ **Hyperparameter Experiments**: FLA research framework
  - 3 experiment configs (regularization, learning rate, smaller model)
  - Experiment documentation framework

**Documentation**:
- ✅ NEDC evaluation docs: `docs/06-evaluation/NEDC_*.md` (overview, guide, reference)
- ✅ Optimized agent docs: CLAUDE.md, AGENTS.md (identical, multi-agent compatible)
- ✅ Training methodology: `docs/05-training/training-methodology.md`
- ✅ Hyperparameter experiments: `docs/05-training/HYPERPARAMETER_EXPERIMENTS.md`

**Impact**:
- **Clinical Validation**: Can now compute official NEDC metrics (sensitivity @ FA rates)
- **Agent Efficiency**: 66% reduction in context consumption per session
- **Research Framework**: Structured experiments for FLA optimization

### v4.0.0 - FLA Production + WSL2 Fix (October 12, 2025)

**MAJOR RELEASE**: Dual production stacks + critical WSL2 fix

**New Capabilities**:
- ✅ **MAJOR: FLA stack production-ready**: BiGatedDeltaNet fully validated, training successfully on local RTX 4090
- ✅ **MAJOR: WSL2 SIGBUS fix**: Local FLA training now works after cache migration to native ext4 filesystem
- ✅ **MAJOR: Dual training stacks**: BiMamba2 (Modal, PAUSED at Epoch 6) + FLA (Local, Epoch 7/100) training simultaneously
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

## Current Deployment

### FLA-FOCUSED TRAINING (v4.2.0)

**BiMamba2 - Modal Training (⏸️ PAUSED)**:
- Launch: Oct 10, 2025
- Stopped: Oct 13, 2025 (budget control decision)
- Progress: **PAUSED at Epoch 6** (5 complete epochs + 50% of epoch 6)
- Cost: **$1,118 spent**
- Checkpoints: ✅ Backed up to Modal SSD + local (`backups/modal_bimamba2_epoch6/`)
- Stack: TCN + BiMamba2 + GNN + Dynamic LPE
- Resumable: Yes, anytime via `modal run --detach --action train --config configs/modal/train_bimamba.yaml --resume`

**Rationale for Pause**:
- Budget-conscious decision as independent researcher
- FLA (Gated DeltaNet) is primary research focus
- BiMamba2 useful for comparison but not critical path
- Can resume incrementally when budget allows

**FLA Baseline - Local Training (🔄 RUNNING, PLATEAUED)**:
- Launch: Oct 16, 2025 (resumed from epoch 13)
- Current: **Epoch 30+** (running, will auto-stop ~epoch 36)
- Status: **Plateaued at 0.257** for 13 epochs (17-29)
- Best: **0.284 @ epoch 9** (0.95 TAES, 10 FA/24h)
- Patience: 13/20 (early stop triggers at 20)
- Expected stop: Epoch 36 (patience exhausted)
- Config: `configs/local/train_fla.yaml` (patience=20, min_epochs=30)
- Stack: TCN + BiGatedDeltaNet (FLA) + GNN + Dynamic LPE
- Cost: $0 (local RTX 4090)
- Checkpoints: `results/local_fla_training/checkpoints/`

**Timeline**:
- Oct 11-13: Initial training, stopped at epoch 13 (crash)
- Oct 16: Resumed from epoch 13 with patience=20
- Oct 16-Nov 1: Epochs 13-30+, plateaued at 0.257
- Expected: Nov 4-5: Early stop at epoch 36

**Hypothesis**: Stuck in local minimum at 0.257 (lost 0.284 peak after resume chaos)

**Experiments Completed**:
- ✅ **Exp1 (Regularization)**: FAILED (-2.7% vs baseline)
  - Best: 0.272 @ epoch 6 vs baseline 0.284
  - Config: Dropout 0.1→0.2, weight_decay 0.01→0.05
  - Conclusion: Model NOT overfitting, regularization hurts
  - Status: Completed, archived

**Experiments Ready to Launch**:
- 🚀 **Exp4 (Cyclic LR / SGDR)**: Ready after baseline stops
  - Config: `configs/local/train_fla_exp4_cyclic.yaml`
  - Strategy: SGDR restarts to escape 0.257 local minimum
  - Params: t_initial=10, t_mult=2, eta_min=1e-6, patience=15
  - Expected: Break plateau, potentially recover 0.28+ region
  - Launch: After baseline early stops (~epoch 36)

**Experiments Paused/Rejected**:
- ⏸️ **Exp2 (Lower LR)**: Available but not prioritized (Exp4 more promising)
- ❌ **Exp3 (Smaller Model)**: Rejected (capacity reduction not justified)

**Next Actions**:
1. ⏳ Wait for baseline early stop (~epoch 36, Nov 4-5)
2. 🚀 Launch Exp4 (SGDR) to test local minimum escape hypothesis
3. 📊 Monitor Exp4 epochs 1-15 for early signal (should hit 0.28+ if hypothesis correct)
4. 🎯 If Exp4 succeeds: New baseline >0.28
5. 🎯 If Exp4 fails: Accept 0.284 as architecture ceiling, consider BiMamba2 comparison

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
