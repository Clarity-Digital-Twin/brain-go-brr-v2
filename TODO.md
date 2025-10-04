# Technical Debt Priority List

**Last Updated:** 2025-10-03
**Status:** 🔴 1 active P1 (non-blocking) | ✅ All legacy P1 completed | ✅ **ALL P2 STRUCTURAL REFACTORING COMPLETE!** 🎉
**Refactoring Sprint:** detector.py, metrics.py, cli.py — ALL DONE (2025-10-03)
- 435 tests passing ✅
- 78% coverage (exceeds 75% threshold) ✅
- All quality checks passing ✅

---

## Priority Legend

- **P0 (BLOCKER):** Must fix before production deployment
- **P1 (HIGH):** Significant impact on maintainability/velocity
- **P2 (MEDIUM):** Should fix but not urgent
- **P3 (LOW):** Nice to have, defer indefinitely
- **✅ RESOLVED:** Completed items (kept for reference)

---

## P0: BLOCKERS (None Found! 🎉)

**No critical blockers identified.** System is production-ready with current architecture.

---

## P1: HIGH PRIORITY

### 🔴 1.0 ACTIVE P1 ITEMS

#### 1.0.1 Validation Loss Weighting Under Imbalance (OPEN)

**Issue:** Training loss uses `pos_weight` for class imbalance, validation loss doesn't.

**Evidence:**
- Training: `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` (src/brain_brr/train/train_step.py:141)
- Validation: `BCEWithLogitsLoss()` (src/brain_brr/train/val_step.py:239) — NO pos_weight

**Impact:** Train/val losses not directly comparable due to different class weighting. Affects interpretability, not correctness.

**Solutions:**
1. **Option A:** Mirror `pos_weight` in validation loss (makes losses comparable)
2. **Option B:** Report both weighted + unweighted validation loss (more informative)

**Priority:** P1 (affects model selection and interpretability during training)

**Status:** OPEN — Defer to post-training (not a blocker)

---

### ✅ 1.1-1.4 LEGACY P1 ITEMS (Completed 2025-09-30)

### ✅ 1.1 Test Suite Configuration Enforcement

**Issue:** Central configuration exists but many tests hardcoded batch sizes

**Status:** ✅ **RESOLVED** (Commits: 6cf555f, 13b50d3, 6e71240)

**What Was Done:**
1. Created `test_batch_size` fixture in `tests/conftest.py`
2. Replaced ALL 14 hardcoded batch sizes across 7 files:
   - ✅ `tests/unit/train/test_loop.py` (3 locations)
   - ✅ `tests/unit/utils/test_training_logger.py` (2 locations)
   - ✅ `tests/integration/test_tcn_integration.py` (1 location)
   - ✅ `tests/integration/test_training_edge_cases.py` (3 locations)
   - ✅ `tests/integration/test_model_assembly.py` (1 location)
   - ✅ `tests/performance/test_memory.py` (2 locations)
   - ✅ `tests/performance/test_latency.py` (2 locations)
3. Updated `sample_windows` to use `TEST_MAX_BATCH_SIZE`

**Verification:**
```bash
$ rg "batch_size\s*=\s*\d+" tests/ --type py | grep -v test_batch_size | wc -l
0  # ✅ Zero hardcoded batch sizes remaining
```

**Actual Effort:** 2 hours
**Documentation:** `tests/TEST_SUITE_CURRENT_STATE.md`

---

### ✅ 1.2 Redundant Cleanup Fixtures

**Issue:** Two fixtures performed identical GPU tensor cleanup, executing twice per test

**Status:** ✅ **RESOLVED** (Commit: 9fcb823)

**What Was Done:**
1. Simplified `cleanup_torch_resources` in conftest.py (removed duplicate `gc.get_objects()` loop)
2. Documented that `gpu_memory_guard.pytest_runtest_teardown` owns definitive cleanup
3. Made GPU memory fraction configurable via `BGB_TEST_GPU_FRACTION`

**Verification:**
```bash
$ rg "gc\.get_objects\(\)" tests/ --type py
tests/gpu_memory_guard.py:64:        for obj in gc.get_objects():  # ✅ ONLY ONE
tests/conftest.py:467:    # Note: Heavy cleanup handled by gpu_memory_guard
```

**Actual Effort:** 1 hour
**Documentation:** `tests/GPU_ADJUSTMENTS.md`

---

### ✅ 1.3 Print Statements Verification

**Issue:** Ensure no raw `print()` calls in core library code

**Status:** ✅ **VERIFIED** - No action required

**Evidence:**
- Raw `print(` only in examples: `src/brain_brr/utils/logging_patterns.py` (3 lines)
- CLI uses `console.print(...)` intentionally
- Training logger uses Rich tables intentionally

**Action:** Periodic verification (completed 2025-09-30)

---

### ✅ 1.4 GPU Memory Fraction Hardcoded

**Issue:** `gpu_memory_guard.py` hardcoded 40% memory limit

**Status:** ✅ **RESOLVED** (Commit: 9fcb823)

**What Was Done:**
```python
# BEFORE:
torch.cuda.set_per_process_memory_fraction(0.4, 0)  # Hardcoded

# AFTER:
fraction = float(os.getenv("BGB_TEST_GPU_FRACTION", "0.4"))
torch.cuda.set_per_process_memory_fraction(fraction, 0)
```

**Also Added:** `BGB_TEST_GPU_FRACTION_TRAIN` for training-safe mode (12% default)

**Actual Effort:** 5 minutes
**Documentation:** `tests/GPU_ADJUSTMENTS.md`

---

## ✅ P2: MEDIUM PRIORITY — Structural Refactors **COMPLETE!** 🎉

**Completion Date:** 2025-10-03
**Summary:** All HIGH/MEDIUM priority structural debt resolved. Codebase now follows clean code principles with improved testability and maintainability.

### ✅ 2.1 loop.py Modularisation (Completed 2025-10-02)

**Status:** ✅ COMPLETE — See `STRUCTURAL_DEBT_AUDIT_2025-10-02.md` and commit `36055df` for details.

---

### ✅ 2.2 ResourcesConfig Removal 🗑️

**Status:** ✅ COMPLETE — Dead schema removed (commit `318f4dc`).

---

### ✅ 2.3 Fixture Naming Standardisation 🏷️

**Status:** ✅ COMPLETE — Duplicate `trained_model` fixture removed (commit `f7b8fc3`).

---

### ✅ 2.4 SeizureDetector Modularisation — **COMPLETED 2025-10-03** 🎉

**Scope:** `src/brain_brr/models/detector.py`

**Status:** ✅ **COMPLETE** - See `REFACTOR_DETECTOR_PY.md` and commits `4e1ddf3`, `231c32c`, `815d712`, `040a78c`, `9f08d10`

**Execution Results:**
- ✅ Builders extracted to `models/builders/` (node_stream, edge_stream, fusion, regularization)
- ✅ Forward pipeline decomposed into helpers (`_run_node_stream`, `_run_edge_stream`, `_apply_gnn_fusion`, `_decode_and_sanitize`)
- ✅ **Line Reduction:** `from_config` -46% (199→107), `forward` -77% (187→42)
- ✅ **Verification:** 5/5 detector tests passing, backward compatibility preserved
- ✅ **Quality:** All mypy/ruff checks passing, 94% coverage

---

### ✅ 2.5 Metrics Pipeline Decomposition — **COMPLETED 2025-10-03** 🎉

**Scope:** `src/brain_brr/eval/metrics.py`

**Status:** ✅ **COMPLETE** - See `REFACTOR_METRICS_PY.md` and commits `4e1ddf3`, `231c32c`, `815d712`

**Execution Results:**
- ✅ Extracted `eval/helpers/timeline.py` (37 lines) - 100% coverage
- ✅ Extracted `eval/helpers/false_alarm.py` (58 lines) - FA sweep + sensitivity
- ✅ Extracted `eval/helpers/scalar_metrics.py` (25 lines) - TAES/AUROC/ECE reducers
- ✅ **Line Reduction:** `evaluate_predictions` -47% (185→98)
- ✅ **Verification:** 26/26 evaluation tests passing, helpers well-tested
- ✅ **Quality:** 87% eval coverage (up from 66%)

---

### ✅ 2.6 CLI Service Layer Extraction — **COMPLETED 2025-10-03** 🎉

**Scope:** `src/brain_brr/cli/cli.py`

**Status:** ✅ **COMPLETE** - See `REFACTOR_CLI_PY.md` and commit `040a78c`

**Execution Results:**
- ✅ Created `cli/services/evaluation.py` (100 lines) - core evaluation orchestration
- ✅ Thinned `evaluate` command to parse-and-delegate pattern
- ✅ **Line Reduction:** `evaluate` command -58% (224→95)
- ✅ **Verification:** CLI integration tests passing, UX unchanged
- ✅ **Quality:** 70% CLI coverage (up from 57%), service layer testable without Click

---

### ✅ 2.7 EDF IO Pipeline - Actually Clean! (Verified - LOW PRIORITY)

**Scope:** `src/brain_brr/data/io.py`

**Status:** ✅ **VERIFIED** - Plan rewritten based on actual code (2025-10-02)

**Audit Discovery:**
- ❌ Original AI-generated plan was COMPLETELY WRONG (hallucinated operations)
- ✅ Traced ACTUAL pipeline: io.py → preprocess.py → datasets.py
- ✅ Resampling/filtering happen in `preprocess.py`, NOT in `load_edf_file()`
- ✅ Function is actually well-organized (153 lines, 94% test coverage)

**What function ACTUALLY does (verified):**
1. ✅ Read EDF with MNE (with header repair fallback)
2. ✅ Normalize channel names (TUSZ-specific cleaning)
3. ✅ Apply channel synonyms (T7→T3, etc.)
4. ✅ Filter to target channels
5. ✅ Interpolate missing midline (Fz/Pz) if possible
6. ✅ Apply montage (best-effort)
7. ✅ Return RAW microvolts (NO preprocessing)

**Actual Data Pipeline (documented in new plan):**
```
1. load_edf_file() → raw µV data [io.py]
2. preprocess_recording() → filtered/resampled [preprocess.py]
3. parse_tusz_csv() → events [io.py]
4. events_to_binary_mask() → labels [io.py]
5. extract_windows() → windowed data [windows.py]
```

**Refactor Plan:** Minimal helper extraction (optional)
- Extract channel normalization helper
- Extract interpolation helper
- Keep in same file (313 lines total is fine)

**Priority:** **LOW** - Defer unless specific need arises
- Already working reliably in production
- 94% test coverage
- Clean separation of concerns across modules
- Focus on detector.py and metrics.py first (real complexity issues)

**Details:** See rewritten `REFACTOR_IO_PY.md` for complete analysis

---

## P3: LOW PRIORITY

### 3.1 Code Duplication (Minimal) 📋

**Audit Result:** ✅ **NO DUPLICATION FOUND** — prior concern resolved; no action required.

**Actual Effort:** 5 minutes (verification)

---

### 3.2 OOM Test Simulation 🧪

**Issue:** `test_cuda_oom_recovery` simulates OOM instead of exercising real behaviour.

**Options:** Document simulation or move to manual stress test. Low impact; revisit post structural refactors.

---

### 3.3 Deprecated Split Policy Warning 📢

**Issue:** Legacy split policy still supported for backwards compatibility.

**Action:** None until v4.0 roadmap finalised.

---

## ✅ RESOLVED ITEMS

- ✅ Import audit — consistent `src.brain_brr` imports.
- ✅ Race condition review — no issues found.
- ✅ Memory leak audit — clean.
- ✅ Mock usage audit — legitimate usage only.

---

**Next Review:** After consensus on structural refactor plans or completion of Modal/Local training (eta 15–42 days).
