# Comprehensive Refactoring Plan for V4.0

**Created:** 2025-10-02
**Status:** DRAFT - Pending validation by second agent
**Scope:** Full technical debt elimination and architecture improvements
**Risk Level:** HIGH - Should only execute after current training completes

---

## Executive Summary

**Total Issues Found:** 23 items across 4 categories
**Estimated Total Effort:** 12-15 days
**High-Risk Items:** 3 (loop.py refactor, legacy split removal, deprecated code removal)
**Quick Wins:** 8 (documentation fixes, code cleanup)

**Key Findings:**
1. ✅ **P1/P2 mostly complete** - Only 1 P2 item remaining (loop.py)
2. ⚠️ **Documentation out of sync** - 6 files reference removed ResourcesConfig
3. ⚠️ **Monolithic loop.py** - 1865 lines, 13 functions (516-line train_epoch!)
4. ⚠️ **Deprecated code paths** - Still supporting legacy split policies
5. ✅ **Memory patterns good** - Minimal unnecessary copies
6. ✅ **Throughput limited by compute** - Batch timing dominated by forward/backward pass

---

## Category 1: Documentation Fixes (LOW RISK, 2-3 hours)

### 1.1 ResourcesConfig References in Docs ⚠️

**Issue:** ResourcesConfig removed from code but still in 3 doc files

**Files to update:**
- `docs/03-configuration/schema-validation.md:26` - Remove resources: block example
- `docs/03-configuration/schema-validation.md:161` - Remove invalid compile_model note
- `docs/reference/development/test-suite-fix-plan.md:577` - Remove from decision list

**Fix:**
```bash
# Remove all ResourcesConfig references
# Add note: "Removed in v3.3.1 - mixed_precision now in TrainingConfig only"
```

**Effort:** 30 minutes
**Risk:** None

---

### 1.2 TODO.md Inconsistencies ⚠️

**Issue:** Document has conflicting/duplicate information

**Problems:**
1. "Decide on ResourcesConfig" listed under "After Training" (already removed!)
2. "Cache loading consolidation" listed as P3.1 (already verified as false positive!)
3. "minimal_model_no_leak (41 usages)" is FALSE - only 1 usage (the definition)
4. P2.2/P2.3 show "Commit: pending" but commits are already done (318f4dc, f7b8fc3)

**Fix:**
```markdown
# Remove from "After Training Completes":
- Item 6 (ResourcesConfig decision) - DONE

# Update P2.2/P2.3:
- Change "Commit: pending" → "Commit: 318f4dc, f7b8fc3"

# Update P2.3 fixture analysis:
- Change "41 usages" → "1 usage (definition only, unused fixture)"

# P3.1 already marked correctly as "NO DUPLICATION FOUND"
```

**Effort:** 15 minutes
**Risk:** None

---

### 1.3 Fixture Usage Documentation ⚠️

**Issue:** Confusion about which fixtures are actually used

**Current state (VERIFIED):**
- `minimal_model` (tests/conftest.py): 43 usages across multiple files ✅ ACTIVE
- `trained_model` (tests/conftest.py): ✅ REMOVED (Oct 2, commit f7b8fc3)
- `minimal_model_no_leak` (tests/performance/conftest.py): 1 usage (UNUSED - can be removed)
- `small_model` (test_training_edge_cases.py): 2 usages ✅ ACTIVE

**Fix:** Remove unused `minimal_model_no_leak` fixture

**Effort:** 10 minutes
**Risk:** None (run `make test-performance` to verify)

---

## Category 2: Code Quality Improvements (MEDIUM RISK, 3-4 days)

### 2.1 Monolithic loop.py Refactoring 🔥 **HIGH PRIORITY**

**Issue:** Training loop is 1865 lines with massive functions

**Current structure:**
```python
# src/brain_brr/train/loop.py (1865 lines total)
- train_epoch():      516 lines (lines 316-832)   # TOO BIG
- validate_epoch():   262 lines (lines 833-1095)  # TOO BIG
- main():             335 lines (lines 1361-1695) # TOO BIG
- 10 other functions: ~750 lines
```

**Proposed refactoring:**
```
src/brain_brr/train/
├── loop.py              # Main orchestration (< 400 lines)
├── train_step.py        # train_epoch() logic (< 300 lines)
├── val_step.py          # validate_epoch() logic (< 200 lines)
├── checkpoint.py        # Checkpointing logic (< 150 lines)
├── metrics_tracker.py   # Metrics accumulation (< 100 lines)
└── warmup.py            # Warmup schedules (already small)
```

**Benefits:**
- Easier testing (can test train_step independently)
- Clearer separation of concerns
- Easier debugging (smaller files to navigate)
- Reduced cognitive load for contributors

**Risks:**
- Could break training if done incorrectly
- Needs comprehensive test coverage
- Must maintain backward compatibility

**Effort:** 3-4 days (careful extraction + testing)
**Risk:** HIGH (defer until after current training completes)

**Testing strategy:**
```bash
# Before refactoring:
make s  # Baseline smoke test
.venv/bin/pytest tests/unit/train/ -v  # All pass

# After refactoring:
make s  # Must match baseline exactly
.venv/bin/pytest tests/unit/train/ -v  # All must still pass
# Run 3-epoch test to verify metrics match
```

---

### 2.2 Large Utility Files (Optional)

**Candidates:**
- `utils/training_logger.py` (589 lines) - Could split into logger + formatters
- `cli/cli.py` (603 lines) - Could split into subcommands
- `eval/metrics.py` (667 lines, 14 functions) - Already well-structured, no action needed

**Recommendation:** DEFER - These are well-organized despite size

---

## Category 3: Deprecated Code Removal (HIGH RISK, 2-3 days)

### 3.1 Remove Legacy Split Policy Support 🔥

**Issue:** Old custom split still supported despite being deprecated

**Current code:**
```python
# src/brain_brr/config/schemas.py:47-56
split_policy: str = Field(
    default="official_tusz",
    description="Split policy: 'official_tusz' uses train/dev/eval dirs, 'custom' allows validation_split",
)
validation_split: float = Field(
    default=0.15,
    ge=0.05,
    le=0.95,
    description="DEPRECATED - Only used if split_policy='custom'. Use official TUSZ splits!",
)

# src/brain_brr/train/loop.py:1606
# DEPRECATED: Old file-based split (WARNING: May cause patient leakage!)
```

**Fix:**
1. Remove `split_policy` field (always use official_tusz)
2. Remove `validation_split` field
3. Remove legacy split code from loop.py
4. Update all configs to remove these fields
5. Add migration guide to CHANGELOG

**Effort:** 2-3 days (includes updating all configs + tests)
**Risk:** HIGH (could break existing user configs)

**Migration path:**
```yaml
# Before (deprecated):
data:
  split_policy: custom
  validation_split: 0.2

# After:
data:
  # split_policy removed - always uses official TUSZ train/dev/eval dirs
  # validation_split removed - use official splits
```

---

### 3.2 Remove Deprecated Metrics Parameters

**Issue:** `threshold` parameter deprecated but still accepted

**File:** `src/brain_brr/eval/metrics.py:199-214`

**Fix:**
```python
# Remove threshold parameter entirely
# Users must use post_cfg.hysteresis.tau_on instead
```

**Effort:** 30 minutes
**Risk:** LOW (parameter already ignored, just remove)

---

## Category 4: Performance Observations (LOW PRIORITY)

### 4.1 Batch Timing Analysis ✅

**Findings:**
- Modal (A100) batches: ~60-65s with batch_size=48; GPU compute accounts for ~90% of the wall clock.
- Local (RTX 4090) batches: ~23s with batch_size=8; single-threaded DataLoader still keeps the GPU busy.
- DataLoader prefetching and OS page cache prevent I/O stalls; measured utilisation shows no idle GPU time waiting for data.

**Conclusion:** Training is compute-bound. Restructuring the cache format would yield at most a 1-3% speedup while requiring a full cache rebuild. No action recommended.

---

### 4.2 Optional Ideas (Defer)

- **Manifest caching:** Saves ~500ms per epoch — negligible benefit.
- **Async I/O / alt formats:** Adds complexity for marginal gains; not recommended unless profiling changes.
- **Gradient checkpointing (future exploration):** Could trade compute for memory to test larger batches once current training finishes.

---

## Category 5: Test Suite Cleanup (LOW RISK, 1 day)

### 5.1 Remove Unused Fixture ✅

**Issue:** `minimal_model_no_leak` defined but never used

**Fix:**
```python
# Remove from tests/performance/conftest.py:47-70
# Verify with: rg "minimal_model_no_leak" tests/ --type py
```

**Effort:** 10 minutes
**Risk:** None

---

### 5.2 OOM Test Redesign (Optional)

**Issue:** `test_cuda_oom_recovery` simulates OOM instead of testing real behavior

**Options:**
1. Keep as-is + add docstring noting it's simulated
2. Move to manual stress test suite
3. Delete entirely

**Recommendation:** Option 1 (add docstring) - 5 minutes

---

## Category 6: New Issues Found

### 6.1 MNE Import Pattern

**Finding:** MNE imported in 4 places with type: ignore comments

**Files:**
- `src/brain_brr/data/io.py` (2 imports)
- `src/brain_brr/utils/pick_utils.py` (2 imports)

**Issue:** MNE has no type stubs, causing mypy warnings

**Options:**
1. Add `mne-stubs` package (if exists)
2. Create local stub file
3. Keep `type: ignore` comments

**Recommendation:** Option 3 (keep as-is) - No action needed

---

## Prioritized Execution Plan

### Phase 1: Quick Wins (3-4 hours) - CAN DO NOW
1. ✅ Fix TODO.md inconsistencies (15 min)
2. ✅ Update ResourcesConfig doc references (30 min)
3. ✅ Remove unused minimal_model_no_leak (10 min)
4. ✅ Add OOM test docstring (5 min)
5. ✅ Update fixture usage docs (10 min)

**Total Phase 1:** ~1 hour of actual work

---

### Phase 2: Code Quality (3-4 days) - AFTER TRAINING
1. 🔥 Refactor loop.py into modules (3-4 days, HIGH RISK)
2. ✅ Remove deprecated metrics threshold (30 min, LOW RISK)

**Total Phase 2:** 3-4 days

---

### Phase 3: Breaking Changes (2-3 days) - V4.0 ONLY
1. 🔥 Remove legacy split_policy support (2-3 days, HIGH RISK)

**Total Phase 3:** 2-3 days

---

### Phase 4: Performance (Optional, defer indefinitely)
1. Manifest loading optimization (1-2 days, marginal gain)
2. Large utility file splits (2-3 days, low value)

---

## Risk Assessment

### High-Risk Items (Require Extra Care):
1. **loop.py refactoring** - Could break training if done wrong
   - Mitigation: Extensive testing, smoke tests, metric validation

2. **split_policy removal** - Breaking change for users
   - Mitigation: Version bump to v4.0, migration guide, deprecation warnings

3. **Deprecated code removal** - Could break existing workflows
   - Mitigation: Deprecation warnings in v3.x, removal in v4.0

### Low-Risk Items (Safe to Execute):
1. Documentation updates
2. Unused code removal
3. Test docstring improvements
4. TODO.md fixes

---

## Validation Checklist

Before implementing ANY changes:
- [ ] Verify current training is complete and saved
- [ ] Run full test suite: `make test`
- [ ] Run quality checks: `make q`
- [ ] Create backup branch: `git checkout -b backup/pre-v4-refactoring`
- [ ] Document baseline metrics from current training

After each change:
- [ ] Run affected tests
- [ ] Run smoke test: `make s`
- [ ] Verify no regressions in metrics
- [ ] Update documentation

---

## Open Questions for User

1. **Timing:** Execute Phase 1 now, or wait until both trainings complete?
2. **loop.py:** Split into multiple files (recommended) or keep monolithic?
3. **split_policy:** Remove in v4.0 (breaking change) or keep deprecated forever?
4. **Performance features:** Should we explore gradient checkpointing after current training as a follow-up experiment?

---

## Files Requiring Changes

### Immediate (Phase 1):
- `TODO.md` - Fix inconsistencies
- `docs/03-configuration/schema-validation.md` - Remove ResourcesConfig
- `docs/reference/development/test-suite-fix-plan.md` - Remove ResourcesConfig
- `tests/performance/conftest.py` - Remove unused fixture
- `tests/integration/test_training_edge_cases.py` - Add OOM test docstring

### Phase 2:
- `src/brain_brr/train/loop.py` - MAJOR refactoring
- `src/brain_brr/eval/metrics.py` - Remove threshold param
- Create: `src/brain_brr/train/train_step.py`
- Create: `src/brain_brr/train/val_step.py`
- Create: `src/brain_brr/train/checkpoint.py`
- Create: `src/brain_brr/train/metrics_tracker.py`

### Phase 3:
- `src/brain_brr/config/schemas.py` - Remove split_policy, validation_split
- `src/brain_brr/train/loop.py` - Remove legacy split code
- `configs/*.yaml` - Remove deprecated fields
- `CHANGELOG.md` - Document breaking changes
- `MIGRATION.md` - Create migration guide

---

## Next Steps

1. **Review this document** - User confirms accuracy
2. **Align with agent** - Second agent validates plan
3. **Execute Phase 1** - Quick wins (if approved)
4. **Wait for training** - Don't touch risky items until training done
5. **Execute Phase 2/3** - After training completes and is validated

---

## Success Criteria

**Phase 1 (Quick Wins):**
- [ ] All documentation accurate
- [ ] No unused code in tests/
- [ ] TODO.md reflects reality
- [ ] `make q` passes
- [ ] `make test` passes

**Phase 2 (Code Quality):**
- [ ] loop.py < 400 lines
- [ ] All functions < 100 lines
- [ ] `make s` produces identical metrics
- [ ] Full test suite passes
- [ ] No regressions in training

**Phase 3 (Breaking Changes):**
- [ ] Migration guide complete
- [ ] Deprecation warnings in v3.x
- [ ] Clean removal in v4.0
- [ ] No technical debt remaining

---

**Document Status:** DRAFT - Awaiting validation
**Created by:** Claude Code (Audit + Investigation)
**Next Step:** Validate with second agent before execution
