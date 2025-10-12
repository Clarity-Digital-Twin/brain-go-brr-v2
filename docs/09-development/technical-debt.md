# Technical Debt & Cleanup Status

**Last Updated**: October 12, 2025 (v4.0.0)
**Status**: 🟢 **ZERO ACTIVE DEBT** - Dual stacks (BiMamba2 + FLA) live with deterministic resume, ext4 cache migration, and WSL2 SIGBUS fix integrated

## Executive Summary

| Priority | Count | Training Impact |
|----------|-------|-----------------|
| **P0 BLOCKER** | 0 | None |
| **P1 URGENT** | 0 | None |
| **P2 MEDIUM** | 0 | None |
| **P3 LOW** | 0 | None |
| **P4 OPTIONAL** | 3 | None (future enhancements) |

**Current Status**: All known technical debt has been eliminated. Codebase is clean and Modal A100 training is running.

Priority definitions (for quick reference):
- **P0** — Blocks training/inference or corrupts results.
- **P1** — High risk of silent degradation; not a hard block.
- **P2** — Medium risk/confusion; suboptimal defaults.
- **P3** — Low-risk polish and maintenance.
- **P4/P5** — Optional enhancements or research experiments.

---

## Recently Resolved (October 10–12, 2025 — v3.11.0 → v4.0.0)

- **Exact mid-epoch resume** (StatefulDataLoader + checkpoint metadata)
  - Added `torchdata.stateful_dataloader.StatefulDataLoader` for train/val loops, persisted loader state in checkpoints, and restored `resume_batch_idx` so Modal restarts skip straight to the saved batch.
  - `global_step` is now saved/restored alongside optimizer/scheduler state; warmup schedules, gradient logging, and W&B metrics continue without restarting from zero.
- **RNG device safety**
  - Checkpoint loads keep RNG tensors on CPU before calling `torch.set_rng_state`, preventing `RNG state must be a torch.ByteTensor` when resuming on CUDA.
  - Regression coverage lives in `tests/unit/train/test_checkpoint_rng_device.py`.
- **Pydantic warning sweep**
  - Optional config fields now use `Annotated[..., Field(...)]`, eliminating `UnsupportedFieldAttributeWarning` noise introduced in Pydantic 2.12.
  - `make q` runs warning-free while keeping schema validation strict.
- **Config/documentation realignment**
  - All tooling defaults, Docker `CMD`, docs, and tests reference `{smoke,train}_{bimamba,fla}.yaml` after the architecture split.
  - Modal quick commands and README examples now match the live file layout.
- **WSL2 SIGBUS mitigation**
  - Root cause traced to mmap cache living on `/mnt/d` (9P filesystem). Cache migrated to native ext4 volume; new guide lives in `docs/08-operations/wsl2-sigbus-fix.md`.
  - Driver baseline updated to 581.42; troubleshooting doc now captures both driver and filesystem fixes.
- **Oct 9 “Bug Hunt” audit**
  - Full repo sweep validated config paths, test coverage, and documentation parity (see `docs/09-development/bug-tracker.md`).
  - Outcome: zero lingering P0/P1 items; archives retained for historical evidence only.

## Recently Resolved (October 8, 2025 - v3.9.0)

- **Atomic checkpoint system** (checkpoint.py): temp-file + fsync + rename, plus AMP scaler & RNG capture for deterministic resume.
- **Timeout guard** (timeout_guard.py + loop.py): exits ~23 h with `timeout_exit.pt`, prevents Modal hard kill mid-save.
- **Operational polish**: metric key normalization (`metrics_utils.py`), W&B run persistence, and Modal docs refreshed (`docs/05-training/modal.md` / `modal-deployment.md`).

## Recently Resolved (October 6, 2025 - v3.8.2)

### Zero Warnings (P2 → Closed)
- ✅ Replaced `.clone()` calls with NumPy copy-on-read tensors across all datasets to eliminate read-only warnings without touching mmap cache
- ✅ Added AMP scheduler guard (main loop + accumulation flush) so `scheduler.step()` only runs after a successful optimizer update
- ✅ Verified `make q`, `make test`, and Modal training logs: 0 warnings, accurate LR schedule

### Complete Tensor Safety (P0, v3.8.1)
- ✅ Added `.clone()` to EEGWindowDataset (lines 307, 312) - missed in v3.8.0
- ✅ All 3 datasets now safe from read-only mmap tensor undefined behavior
- ✅ Removed broad warning suppression from train_step.py (paper-over eliminated)
- ✅ Verified scheduler step order is correct (optimizer → scheduler)
- ✅ Updated TECHNICAL_DEBT.md to reflect truth (verified vs fixed)

### Previous Fixes (v3.8.0)
- ✅ Cleaned 3 stray NPZ files from Modal cache (66.1 MiB freed)
- ✅ Fixed datasets.py NPZ creation bug (removed all `np.savez_compressed` calls)
- ✅ Updated cache validation to check NPY files (mmap format)
- ✅ Fixed test regression (cache_dir=None support restored)
- ✅ Fixed all type annotations (WandBRun, Console instead of Any)
- ✅ Extracted duplicate `_load_cache_for_worker` to shared function (120 lines eliminated)
- ✅ Updated NPZ references in comments to reflect NPY mmap format
- ✅ Fixed clean_cache() paths (cache/tusz → cache/tusz_mmap)

### Previous Debt Eliminated
- ✅ NPZ→NPY mmap conversion (387 GB RAM → <1 GB)
- ✅ ValidationDataset caching (49x speedup)
- ✅ Modal SSD cache population (train + dev splits)
- ✅ P0 Cache path hardcoding - Fixed app.py lines 658, 811, 821
- ✅ P1 Documentation paths - Updated all `cache/tusz` → `cache/tusz_mmap`
- ✅ P2 Magic numbers - Added `HEARTBEAT_INTERVAL_SEC` to constants
- ✅ P2 Env var naming - Renamed to `BGB_SKIP_PERF_TESTS`
- ✅ P3 YAML comments - Cleaned double comments in configs
- ✅ BCE mode removed - Focal-only production
- ✅ Deprecated env helpers removed
- ✅ Assertions → exceptions (all 11 in detector.py converted)
- ✅ Type ignore audit (21 → 17, all remaining documented)

---

## P4: Optional Future Enhancements (Non-Blocking)

**Not required for production, but nice to have:**

### 1. Smoke Test Validation Dataset Limitation
**Issue**: When `BGB_LIMIT_FILES=50`, validation dataset is empty
- Smoke test limits training to 50 files
- Validation tries to load 10 files from manifest
- Those 10 files aren't in the limited 50-file set
- Results: 0 validation windows, default metrics returned

**Impact**:
- ❌ Smoke test: Cannot validate metrics (expected behavior)
- ✅ Full training: Uses all 1832 dev files (works perfectly)

**Decision**: Accept current behavior. Smoke test validates:
- ✅ Pipeline execution
- ✅ Training loop completion
- ✅ Gradient stability
- ✅ Cache format correctness
- ✅ No GPU crashes

**Status**: 🟡 DOCUMENTED - Not blocking production training

### 2. Protection Environment Variables (Optional Enhancement)
- **Observation**: `BGB_SANITIZE_GRADS` and similar flags exist but are informational/debug-only
- **Current Protection**: Gradient clipping (0.5) provides primary NaN protection (works perfectly)
- **Optional Future**: Could implement additional opt-in debugging modes if needed
- **Status**: 🟡 OPTIONAL - Training is stable without these

### 3. Unused Constants (Already Documented)
- **Status**: 26 intentional reserves documented in v3.7.0 (LABEL_*, METRIC_*, etc.)
- **Impact**: None at runtime
- **Action**: Already tracked in constants.py header as future-use reserves

---

## Quality Verification

**All Checks Passing** (v3.8.2):
- ✅ `make q` - Lint + format + mypy + config validation → PASS
- ✅ `make test` - 104 tests, 83.80% coverage → PASS
- ✅ Modal cache - 4667 train + 1832 dev NPY files, 0 NPZ
- ✅ Smoke test - Completed successfully, all systems validated

---

**Status**: 🟢 **ZERO ACTIVE DEBT** - Ready for production training
**Next**: Full Modal A100 training (100 epochs)
**Smoke Test**: ✅ PASSED - All critical systems validated (v3.8.2)

Completed items from earlier phases remain documented below for context.

## Completion Summary
- **V3 Architecture**: Fully implemented, V2 removed
- **Legacy Code**: All deprecated code removed
- **Tests**: Migrated to V3-only
- **Documentation**: Updated to reflect V3
- **Quality**: All checks passing

## Completed Phases

### Phase 0: Alignment ✅
- Updated messaging to TCN+BiMamba+V3 everywhere
- Fixed W&B labels and CLI output
- Aligned documentation with implementation

### Phase 1: Soft Deprecation ✅
- Added deprecation warnings for legacy patterns
- Warned on V2 heuristic path usage
- Kept backward compatibility temporarily

### Phase 2: Test Migration ✅
- Updated all tests to V3 architecture
- Removed DynamicGraphBuilder references
- Fixed test fixtures for V3 config

### Phase 3: Complete Removal ✅
- Removed V2 code paths from SeizureDetector
- Deleted graph_builder.py
- Removed legacy config fields
- V3 is now the only architecture

## Code Simplification Achieved
- `detector.py`: Reduced from ~350 to ~250 lines
- Removed 100+ lines of V2 conditional branches
- Eliminated legacy parameter handling
- Clean single-path V3 implementation

## Environment Variable Consolidation
- Created typed helper: `src/brain_brr/utils/env.py`
- Single source of truth for all BGB_* variables
- Comprehensive documentation: `docs/03-configuration/env-vars.md`
- Removed scattered os.getenv() calls

## Numerical Stability
- Implemented 3-tier clamping system
- Fixed initialization with dependency injection
- Removed BGB_TEST_MODE anti-pattern
- Hardcoded critical safeguards

## Verification Gates Passed
- ✅ `make q` passes (ruff/format/mypy)
- ✅ `make t` passes all tests
- ✅ Integration tests pass with V3
- ✅ V3 is default everywhere
- ✅ No V2 references remain (except Modal app name kept for continuity)

## ✅ Additional Completions (October 2, 2025)

### ✅ Documentation Restructure (docs_v3)
- Adopted the Diátaxis layout (`getting-started`, `guides`, `reference`, `explanations`, `development`).
- `docs/README.md` is now the navigation hub with quick links for new users, operators, and contributors.
- Merged redundant NaN and troubleshooting docs into single canonical guides; historical investigations live under `docs/archive/`.
- Validation checks (broken links, uppercase filenames, outdated paths) are scripted as part of the restructure checklist.

### ✅ Loop.py Refactoring (COMPLETED)
**Issue**: Training loop was 958 lines with mixed concerns
**Status**: ✅ **COMPLETED** - 33% reduction achieved (958 → 640 lines)

**What Was Done:**
1. ✅ Extracted warmup utilities → `warmup.py` (43 lines)
2. ✅ Extracted sampling utilities → `sampling.py` (102 lines)
3. ✅ Extracted FocalLoss → `losses.py` (59 lines)
4. ✅ Extracted optimizer/scheduler → `optimizer_factory.py` (92 lines)
5. ✅ Extracted EarlyStopping → `early_stopping.py` (45 lines)

**Results:**
- loop.py: 958 → 640 lines (33% reduction)
- 5 new focused utility modules
- 100% test pass rate (29/29 tests)
- Full SOLID compliance
- Zero regressions

**Commit**: 36055df (2025-10-02)
**Effort**: 4 hours (estimated 5 days!)

### ✅ Evaluation pipeline fixes
- Datasets now return dictionaries with `window`, `label`, `file_id`, and `window_start_s` metadata.
- Validation uses `ValidationDataset` backed by `dev/manifest.json`; metrics stitch per-record timelines and stream memory (<5 GB peak).
- `evaluate_predictions` aggregates per recording, restoring accurate TAES and FA/24 h calculations.

### ✅ Checkpoint and utility extraction
- `train/checkpoint.py` owns save/load helpers; `train/train_utils.py` centralises seeding, memory stats, and worker init.
- Duplicate definitions were removed from `loop.py`, shrinking the module and clarifying imports.

---

## Remaining Technical Debt (October 2, 2025)

### High Priority
1. **Print Statement Migration** (387 total)
   - `src/brain_brr/train/loop.py`: Print count now reduced with refactoring
   - `deploy/modal/app.py`: 114 prints (42 with flush=True)
   - **Action**: Convert to proper logging with levels
   - **Estimated effort**: 2-3 days
   - **Status**: Planning document created, lower priority now

2. **ClampRetirementConfig Dead Code** ✅ FIXED in v3.2.1
   - Removed from `config/schemas.py` and `models/detector.py`
   - Test file renamed: `test_pr4_clamp_retirement.py` → `test_fusion_and_clamp_utils.py`

3. **Version String Updates** ✅ FIXED in v3.2.1
   - Updated all "v2" references to "V3" in module docstrings
   - Fixed in: `__init__.py` files, CLI, tests, Modal deployment

### Medium Priority
4. **File Count Brittleness** ✅ FIXED in v3.2.1
   - Changed exact counts (4667/1832) to ranges (4600-4700/1800-1900)
   - More resilient to dataset variations

5. **Unused CLI Options** ✅ FIXED in v3.2.1
   - Removed unused `--validation-split` from build-cache command
   - Cleaned up "val" split references (now only "train" and "dev")

### Low Priority
6. **PR Comment Documentation**
   - Keep PR-1 through PR-5 comments as historical documentation
   - They document what each refactor implemented
   - **Status**: Decided to KEEP

7. **Historical Planning Documents**
   - Location: `docs/10-final-refactor/`
   - Consider moving to archive subdirectory
   - **Status**: Low priority, useful for reference

### Coverage Baseline (October 3, 2025)
- Overall coverage sits at ~76% (threshold set to 75%) after utility extraction; suites remain green.
- Under-covered surfaces: `train/loop.py`, `train/train_step.py`, CLI services, and `train_utils.py` (seed/workers). Targeted unit tests there would nudge totals toward 78–79%.
- Utilities moved out of `loop.py` now have dedicated tests; we intentionally accept lower orchestration coverage instead of brittle mocks.

## Logging Migration Plan Summary

### Current State
- **Total print() statements**: 387 (247 src + 140 deploy)
- **Real-time prints with flush=True**: 147 total
- **Rich console.print() calls**: 47 in CLI (keep for user-facing output)
- **Files already using logging**: 4 (io.py, clamp_utils.py, tcn.py, mamba.py)
- **No central logging configuration exists**

### Proposed Architecture
1. Central logging configuration: `src/brain_brr/utils/logging_config.py`
2. Environment variables for control:
   - `BGB_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR`
   - `BGB_LOG_FILE=/path/to/logfile.log`
   - `BGB_LOG_FORMAT=rich|simple|json`
   - `BGB_LOG_EVERY_N_STEPS=50` (gate per-batch logs)
3. Specialized loggers for training progress and CLI output
4. Integration with existing BGB_NAN_DEBUG and other flags

### Migration Priority
1. **Critical Path** (Day 1): `train/loop.py`, `deploy/modal/app.py`
2. **Data Pipeline** (Day 2): Data loading/preprocessing files
3. **Models & Utils** (Day 2): Model files and utilities
4. **CLI & Polish** (Day 3): CLI with special handling for user output

## Code Quality Metrics

### Dead Code Detection Results (v3.2.1)
```bash
vulture src/ --min-confidence 90
# Result: 0 critical items (after cleanup)

ruff check src/ --select F401,F841
# Result: All checks passed!
```

### Current Metrics
- Total Python files: ~100
- Total lines of code: ~10,000
- DEBUG print statements: 147 in train/loop.py alone
- TODO/FIXME comments: 0 found
- NOTE comments: 3 found (acceptable)

## Future Enhancements (Optional)
- [✅] Loop.py refactoring (COMPLETED 2025-10-02)
- [ ] Complete logging migration (medium value after refactoring)
- [ ] Alternative edge models (GRU/LSTM)
- [ ] K-hop SSGConv filters
- [ ] Additional edge features (coherence)
- [ ] Pluggable metric interface
- [ ] Direct Modal upload (skip S3 intermediate)
- [ ] CI/CD integration for deployments

## Links
- [V3 Architecture](../04-model/v3-architecture.md)
- [Configuration Guide](../03-configuration/README.md)
- [NaN Prevention](../08-operations/nan-prevention-complete.md)
- [Cache Workflow](../02-data/cache-layout.md)
