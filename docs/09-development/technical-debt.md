# Technical Debt & Cleanup Status

**Last Updated**: October 2, 2025
**Status**: 100% COMPLETE ✅ (Including loop.py refactoring)

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