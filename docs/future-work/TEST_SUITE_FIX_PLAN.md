# Test Suite Fix and Optimization Plan

**Created:** 2025-09-29
**Branch:** `fix/test-suite-config`
**Status:** Ready for Implementation
**Estimated Total Effort:** ~3.5 hours

---

## Executive Summary

The test suite has solid infrastructure (60%) but poor adoption of centralized configuration (40%). This plan addresses **P1 issues only** - fixing configuration enforcement, removing redundant cleanup, and documenting verification procedures. All changes are test-only and will NOT affect the currently running Modal A100 training.

**Key Metrics:**
- **Total Test Files:** 49 Python files
- **Total Tests:** 155 test functions/classes
- **Total Lines:** 10,467 lines
- **Largest Files:**
  - `test_latency.py` (594 lines)
  - `conftest.py` (498 lines)
  - `test_logging_config.py` (472 lines)
  - `test_channel_order.py` (468 lines)

---

## Test Suite Architecture

### Directory Structure
```
tests/                          (2.0M total)
├── __init__.py                 # Empty package marker
├── test_config.py              # Central config (GPU detection, batch sizes)
├── conftest.py                 # Root fixtures (17 fixtures, 498 lines)
├── gpu_memory_guard.py         # Session GPU cleanup (2 fixtures)
├── GPU_ADJUSTMENTS.md          # RTX 4090 adjustments doc
│
├── unit/                       # Fast isolated tests
│   ├── cli/                    # CLI command tests (2 files)
│   ├── config/                 # Config schema tests (empty?)
│   ├── data/                   # Dataset/cache tests (6 files)
│   ├── events/                 # Event detection tests (2 files)
│   ├── models/                 # Model component tests (14 files)
│   ├── post/                   # Post-processing tests (1 file)
│   ├── train/                  # Training loop tests (1 file, 298 lines)
│   └── utils/                  # Utility tests (2 files)
│
├── integration/                # Multi-component tests
│   ├── conftest.py             # Integration fixtures (368 lines)
│   ├── data/                   # I/O integration (1 file)
│   ├── post/                   # Hysteresis integration (1 file)
│   ├── test_evaluation.py      # End-to-end evaluation (274 lines)
│   ├── test_gnn_integration.py # GNN integration (various)
│   ├── test_model_assembly.py  # Model composition (140 lines)
│   ├── test_smoke.py           # Quick sanity check (195 lines)
│   ├── test_streaming.py       # Streaming inference (194 lines)
│   ├── test_tcn_integration.py # TCN performance (250 lines)
│   └── test_training_edge_cases.py # OOM/NaN handling (408 lines)
│
├── performance/                # Latency/memory benchmarks
│   ├── README.md               # Performance test guide
│   ├── conftest.py             # Perf fixtures (3 fixtures)
│   ├── utils.py                # Perf measurement utils (116 lines)
│   ├── test_latency.py         # Inference speed (594 lines)
│   └── test_memory.py          # VRAM usage (434 lines)
│
└── clinical/                   # Clinical validation tests
    ├── test_channel_order.py   # EEG channel mapping (468 lines)
    └── test_taes_metrics.py    # TAES scoring (423 lines)
```

### Fixture Hierarchy
```
Root Level (conftest.py - 17 fixtures):
├── sample_edf_data              # 19-channel EEG generator
├── sample_windows               # Windowed data (uses TEST_BATCH_SIZE not TEST_MAX_BATCH_SIZE!)
├── minimal_model                # 4 TCN, 1 Mamba, d_model=512, graph enabled
├── trained_model                # Same as minimal_model (DUPLICATE)
├── temp_checkpoint_dir          # Temp directory for checkpoints
├── mock_wandb                   # W&B API mock
├── cleanup_torch_resources      # AUTOUSE - GPU cleanup (REDUNDANT with gpu_memory_guard)
├── cleanup_dataloader           # DataLoader worker cleanup
└── ... (9 more fixtures)

GPU Memory Guard (gpu_memory_guard.py - 2 fixtures):
├── pytest_runtest_setup         # Pre-test GPU clear
├── pytest_runtest_teardown      # Post-test GPU cleanup (REDUNDANT with conftest)
└── gpu_memory_limit             # Session-wide 40% memory fraction (HARDCODED)

Performance (performance/conftest.py - 3 fixtures):
├── minimal_model_no_leak        # 4 TCN, 2 Mamba, d_model=256 (DIFFERENT from root minimal_model)
├── perf_data                    # Performance test data
└── gpu_stats                    # GPU memory tracking

Integration (integration/conftest.py):
└── small_detector               # Local fixture definition
```

---

## Current Problems (P1 Issues from TODO.md)

### P1.1: Test Suite Configuration Enforcement ⚠️

**Problem:** Central GPU-aware configuration exists but is ignored.

**Evidence:**
- `TEST_MAX_BATCH_SIZE` defined in `test_config.py` (lines 25-37) with RTX 4090/A100 detection
- **ZERO** external usage found (only internal definition)
- `sample_windows` fixture uses `TEST_BATCH_SIZE` (env var) instead
- **14 hardcoded batch_size assignments** across test suite:

```python
# Hardcoded Locations:
tests/unit/train/test_loop.py:              3 occurrences (batch_size=1)
tests/unit/utils/test_training_logger.py:   2 occurrences (batch_size=32)
tests/integration/test_tcn_integration.py:  1 occurrence (batch_size=4)
tests/integration/test_training_edge_cases.py: 3 occurrences (batch_size=2, 2, 4)
tests/integration/test_model_assembly.py:   1 occurrence (batch_size=4)
tests/performance/test_memory.py:           2 occurrences (batch_size=4, 8)
tests/performance/test_latency.py:          2 occurrences (batch_size=4, 8)
```

**Impact:**
- Tests fail on A100 (expect larger batches)
- Tests OOM on lower-end GPUs
- No environment variable override possible
- OOM fixes accumulated as comments ("Reduced from 16", "Further reduced")

**Root Cause:** `TEST_MAX_BATCH_SIZE` exists but no fixture exposes it for test consumption.

---

### P1.2: Redundant Cleanup Fixtures ⚠️

**Problem:** Two fixtures perform identical GPU cleanup, executing twice per test.

**Evidence:**

**Fixture 1** - `gpu_memory_guard.py` lines 21-31:
```python
def pytest_runtest_teardown(item):
    """Teardown after each test - aggressive GPU cleanup."""
    if torch.cuda.is_available():
        # Clear all GPU tensors
        for obj in gc.get_objects():  # ← Expensive iteration
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        gc.collect()
        torch.cuda.empty_cache()
```

**Fixture 2** - `conftest.py` lines 449-474:
```python
@pytest.fixture(autouse=True)  # ← Runs on EVERY test
def cleanup_torch_resources():
    """Consolidated cleanup for PyTorch resources."""
    # ... pre-test cleanup ...
    yield
    # Post-test cleanup
    for obj in gc.get_objects():  # ← IDENTICAL expensive iteration
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except (AttributeError, RuntimeError):
            pass
    gc.collect()
    torch.cuda.empty_cache()
```

**Additional Manual Cleanups Found:**
- `torch.cuda.empty_cache()`: 15 explicit calls in tests
- `gc.collect()`: 27 explicit calls in tests

**Impact:**
- 2× `gc.get_objects()` iteration per test (expensive on large test suites)
- Confusion about which fixture owns cleanup
- Developers don't trust fixtures (hence 42 manual cleanup calls)
- Test suite slowdown

---

### P1.3: Print Statements Verification 📝

**Status:** ✅ Already verified - NO ACTION REQUIRED

**Evidence:**
- Only 3 raw `print()` calls in `src/`
- All in `src/brain_brr/utils/logging_patterns.py` (example code)
- CLI uses `console.print()` intentionally (Rich library)
- Training logger uses `console.print()` for tables (intentional UX)

**Action:** Periodic verification only (already done).

---

## Implementation Plan

### Phase 1: Create `test_batch_size` Fixture (P1.1)
**Estimated Time:** 2 hours
**Files Changed:** 8 files

#### Step 1.1: Add Fixture to Root conftest.py

**File:** `tests/conftest.py`
**Location:** After line 82 (after `sample_windows` fixture)

```python
@pytest.fixture
def test_batch_size() -> int:
    """
    Get GPU-appropriate batch size for current test environment.

    Returns batch size based on available GPU:
    - RTX 4090: 4
    - A100: 8
    - Unknown GPU: 2
    - CPU: 1

    Can be overridden with TEST_BATCH_SIZE environment variable.
    """
    from tests.test_config import TEST_MAX_BATCH_SIZE
    return TEST_MAX_BATCH_SIZE
```

#### Step 1.2: Replace Hardcoded Batch Sizes (7 files, 14 locations)

**Priority Order** (address in this sequence):

1. **tests/unit/train/test_loop.py** (3 occurrences, lines TBD)
   ```python
   # BEFORE:
   loader = DataLoader(dataset, batch_size=1, ...)

   # AFTER:
   def test_training_loop_basic(test_batch_size):
       loader = DataLoader(dataset, batch_size=test_batch_size, ...)
   ```

2. **tests/unit/utils/test_training_logger.py** (2 occurrences)
   ```python
   # BEFORE:
   batch_size=32

   # AFTER (use fixture but cap at 32 for logger tests):
   def test_metric_buffer(test_batch_size):
       batch_size = min(test_batch_size, 32)  # Logger tests don't need large batches
   ```

3. **tests/integration/test_tcn_integration.py** (1 occurrence)
   ```python
   # BEFORE:
   batch_size = 4

   # AFTER:
   def test_tcn_forward(test_batch_size):
       batch_size = test_batch_size
   ```

4. **tests/integration/test_training_edge_cases.py** (3 occurrences)
   ```python
   # BEFORE:
   batch_size = 2

   # AFTER:
   def test_nan_handling(test_batch_size):
       batch_size = max(test_batch_size, 2)  # Ensure at least 2 for edge cases
   ```

5. **tests/integration/test_model_assembly.py** (1 occurrence)
   ```python
   # AFTER:
   def test_model_assembly(test_batch_size):
       batch_size = test_batch_size
   ```

6. **tests/performance/test_memory.py** (2 occurrences - parameterized)
   ```python
   # BEFORE:
   @pytest.mark.parametrize("batch_size", [4, 8])

   # AFTER (keep parameterization but scale to GPU):
   def test_memory_scaling(test_batch_size):
       # Test with current GPU's max and half of max
       test_sizes = [test_batch_size // 2, test_batch_size]
   ```

7. **tests/performance/test_latency.py** (2 occurrences)
   ```python
   # Similar approach to memory tests
   ```

#### Step 1.3: Update `sample_windows` Fixture

**File:** `tests/conftest.py` line 63-82

```python
# BEFORE:
from tests.test_config import TEST_BATCH_SIZE  # Uses env var only
n_windows = min(n_windows, TEST_BATCH_SIZE)

# AFTER:
from tests.test_config import TEST_MAX_BATCH_SIZE  # Use GPU-aware limit
n_windows = min(n_windows, TEST_MAX_BATCH_SIZE)
```

#### Step 1.4: Verification Commands

```bash
# Verify no hardcoded batch_size remains (should find 0 or only intentional ones)
rg "batch_size\s*=\s*\d+" tests/ --type py

# Run affected tests on RTX 4090
pytest tests/unit/train/test_loop.py -xvs
pytest tests/integration/test_tcn_integration.py -xvs
pytest tests/performance/ -xvs -m "not slow"

# Run with custom batch size (verify override works)
TEST_BATCH_SIZE=2 pytest tests/unit/train/test_loop.py -xvs
```

---

### Phase 2: Simplify Cleanup Fixtures (P1.2)
**Estimated Time:** 1 hour
**Files Changed:** 2 files + test verification

#### Step 2.1: Simplify `cleanup_torch_resources` in conftest.py

**File:** `tests/conftest.py` lines 449-474

```python
# BEFORE (duplicates gpu_memory_guard):
@pytest.fixture(autouse=True)
def cleanup_torch_resources():
    """Consolidated cleanup for PyTorch resources."""
    # Pre-test cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    yield

    # Post-test cleanup - DUPLICATES gpu_memory_guard
    for obj in gc.get_objects():  # ← REMOVE THIS
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except (AttributeError, RuntimeError):
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# AFTER (lightweight, no duplication):
@pytest.fixture(autouse=True)
def cleanup_torch_resources():
    """
    Lightweight GPU cleanup for tests.

    Note: Heavy cleanup (tensor deletion via gc.get_objects()) is handled
    by gpu_memory_guard.pytest_runtest_teardown() session hook to avoid
    duplicate expensive iterations.
    """
    # Pre-test: Quick cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    yield

    # Post-test: Quick cache clear only
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### Step 2.2: Document Cleanup Ownership in gpu_memory_guard.py

**File:** `tests/gpu_memory_guard.py` lines 21-35

```python
def pytest_runtest_teardown(item):
    """
    Teardown after each test - aggressive GPU cleanup.

    This session hook performs the DEFINITIVE cleanup by iterating
    gc.get_objects() and deleting CUDA tensors. This is the ONLY place
    where this expensive operation should happen.

    The conftest.py cleanup_torch_resources() fixture performs lightweight
    cache clearing only, to avoid duplicate iterations.
    """
    if torch.cuda.is_available():
        # Clear all GPU tensors (DEFINITIVE CLEANUP)
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
```

#### Step 2.3: Audit and Remove Manual Cleanups (Optional)

**Command to find manual cleanups:**
```bash
# Find manual empty_cache calls
rg "torch\.cuda\.empty_cache\(\)" tests/ --type py -n

# Find manual gc.collect calls
rg "gc\.collect\(\)" tests/ --type py -n
```

**Decision:** Remove manual cleanups ONLY if tests still pass after fixture changes. If tests fail, investigate why fixtures aren't sufficient rather than keeping manual cleanups.

**Verification:**
```bash
# Run full test suite to verify no memory leaks
pytest tests/ -x --tb=short

# Run memory-intensive tests specifically
pytest tests/performance/test_memory.py -xvs
pytest tests/integration/test_training_edge_cases.py -xvs
```

---

### Phase 3: Make GPU Memory Fraction Configurable (P2.4 - BONUS)
**Estimated Time:** 5 minutes
**Files Changed:** 1 file

**File:** `tests/gpu_memory_guard.py` line 42

```python
# BEFORE:
torch.cuda.set_per_process_memory_fraction(0.4, 0)  # Hardcoded 40%

# AFTER:
import os
fraction = float(os.getenv("BGB_TEST_GPU_FRACTION", "0.4"))
torch.cuda.set_per_process_memory_fraction(fraction, 0)
```

**Documentation Update:** `tests/GPU_ADJUSTMENTS.md`
```markdown
# Test adjustments for local GPUs (RTX 4090)
# Batch sizes reduced: 8→2 to avoid OOM (24GB VRAM limit)
# Speed thresholds relaxed: 0.5s→1.5s (architectural overhead)
# Memory thresholds adjusted: 2.5GB→4.0GB (V3 dual-stream reality)
#
# Use env vars to override for CI/A100:
# BGB_TCN_SPEED_TARGET=0.5 BGB_TCN_MEM_MAX=8.0 pytest ...
# BGB_TEST_GPU_FRACTION=0.6  # Use 60% of GPU memory (default: 0.4)
```

---

## Testing Strategy

### Pre-Implementation Baseline
```bash
# Capture current test status
pytest tests/ --co -q > /tmp/test_inventory_before.txt
pytest tests/ -x --tb=line 2>&1 | tee /tmp/test_results_before.txt
```

### Post-Implementation Verification
```bash
# 1. Quick smoke test (30 seconds)
pytest tests/unit/train/test_loop.py tests/integration/test_smoke.py -xvs

# 2. Affected integration tests (2-3 minutes)
pytest tests/integration/test_tcn_integration.py \
       tests/integration/test_model_assembly.py \
       tests/integration/test_training_edge_cases.py -xvs

# 3. Performance tests (5 minutes)
pytest tests/performance/ -xvs -m "not slow"

# 4. Full suite (10-15 minutes)
pytest tests/ -x --tb=short

# 5. Verify no hardcoded batch sizes remain
rg "batch_size\s*=\s*\d+" tests/ --type py | grep -v test_batch_size

# 6. Memory leak check (run tests twice, compare peak memory)
pytest tests/performance/test_memory.py -xvs
pytest tests/performance/test_memory.py -xvs  # Should have same peak
```

### Environment Variable Testing
```bash
# Test batch size override
TEST_BATCH_SIZE=1 pytest tests/unit/train/test_loop.py -xvs

# Test GPU memory fraction override
BGB_TEST_GPU_FRACTION=0.6 pytest tests/performance/test_memory.py -xvs

# Test on A100 (if available)
# Should automatically use batch_size=8
pytest tests/ -x --tb=short
```

---

## Risk Assessment

| Change | Risk Level | Mitigation |
|--------|-----------|------------|
| Add `test_batch_size` fixture | **LOW** | New fixture, doesn't break existing tests |
| Replace hardcoded batch sizes | **LOW** | Pure test changes, no src/ changes |
| Simplify `cleanup_torch_resources` | **MEDIUM** | Could expose hidden memory leaks |
| Remove manual cleanups | **MEDIUM** | Only if tests pass without them |
| Make GPU fraction configurable | **LOW** | Backward compatible (default 0.4) |

**Mitigation for Medium Risk Items:**
1. Test on RTX 4090 locally first
2. Run full test suite 3× to verify stability
3. Monitor peak GPU memory usage
4. Keep manual cleanups if tests fail without them

---

## Success Criteria

### Must Have (P1)
- ✅ All 14 hardcoded batch sizes replaced with `test_batch_size` fixture
- ✅ `sample_windows` uses `TEST_MAX_BATCH_SIZE` instead of `TEST_BATCH_SIZE`
- ✅ No duplicate `gc.get_objects()` iteration in cleanup
- ✅ Clear documentation of cleanup ownership
- ✅ All tests pass on RTX 4090
- ✅ No regressions in test suite

### Nice to Have (P2)
- ✅ GPU memory fraction configurable via env var
- ✅ Manual cleanups removed (if tests pass)
- ✅ Tests documented in `tests/GPU_ADJUSTMENTS.md`

### Verification Checklist
```bash
# 1. No hardcoded batch sizes (except intentional ones)
[ ] rg "batch_size\s*=\s*\d+" tests/ returns 0 hits in modified files

# 2. TEST_MAX_BATCH_SIZE is used
[ ] rg "TEST_MAX_BATCH_SIZE" tests/conftest.py returns 2+ hits

# 3. Only one gc.get_objects() loop per test
[ ] rg "gc\.get_objects\(\)" tests/ returns only gpu_memory_guard.py

# 4. All tests pass
[ ] pytest tests/ -x --tb=short exits with code 0

# 5. Memory stable
[ ] pytest tests/performance/test_memory.py -xvs shows no leaks

# 6. Documentation updated
[ ] git diff tests/GPU_ADJUSTMENTS.md shows BGB_TEST_GPU_FRACTION
```

---

## Deferred Items (NOT in this PR)

### P2: Medium Priority (After Modal Training)
- Fixture naming standardization (tiny_model, small_model, medium_model)
- Loop.py refactoring (1695 lines, defer until v4.0)
- ResourcesConfig decision (remove vs implement)

### P3: Low Priority (Defer Indefinitely)
- Code duplication in datasets.py (15 lines, 0.25% of codebase)
- OOM test simulation redesign
- Deprecated split policy removal (keep for backward compat)

---

## Rollback Plan

If tests fail or regressions occur:

```bash
# 1. Immediate rollback
git checkout development
git branch -D fix/test-suite-config

# 2. Verify rollback successful
pytest tests/ -x --tb=short

# 3. Investigate failure
git checkout -b debug/test-suite-investigation
# Add debug logging, bisect changes

# 4. Report findings
# Document failure mode in TODO.md P1 section
```

---

## File Manifest

### Files to Modify (Phase 1 & 2)
1. `tests/conftest.py` - Add fixture, simplify cleanup
2. `tests/unit/train/test_loop.py` - Replace batch_size (3 locations)
3. `tests/unit/utils/test_training_logger.py` - Replace batch_size (2 locations)
4. `tests/integration/test_tcn_integration.py` - Replace batch_size (1 location)
5. `tests/integration/test_training_edge_cases.py` - Replace batch_size (3 locations)
6. `tests/integration/test_model_assembly.py` - Replace batch_size (1 location)
7. `tests/performance/test_memory.py` - Replace batch_size (2 locations)
8. `tests/performance/test_latency.py` - Replace batch_size (2 locations)
9. `tests/gpu_memory_guard.py` - Document cleanup ownership

### Files to Modify (Phase 3 - Bonus)
10. `tests/gpu_memory_guard.py` - Make fraction configurable (same file as #9)
11. `tests/GPU_ADJUSTMENTS.md` - Document new env var

### Files to NOT Touch
- ❌ `src/brain_brr/**/*.py` - NO SOURCE CODE CHANGES
- ❌ `configs/**/*.yaml` - NO CONFIG CHANGES
- ❌ `deploy/modal/**/*.py` - NO DEPLOYMENT CHANGES

---

## Execution Timeline

**Total Estimated Time:** ~3.5 hours

| Phase | Task | Time | Cumulative |
|-------|------|------|-----------|
| 0 | Pre-implementation baseline tests | 15 min | 0:15 |
| 1.1 | Add `test_batch_size` fixture | 10 min | 0:25 |
| 1.2 | Replace batch sizes (7 files) | 1h 30min | 1:55 |
| 1.3 | Update `sample_windows` | 5 min | 2:00 |
| 1.4 | Phase 1 verification | 15 min | 2:15 |
| 2.1 | Simplify cleanup fixture | 15 min | 2:30 |
| 2.2 | Document cleanup ownership | 10 min | 2:40 |
| 2.3 | Audit manual cleanups | 20 min | 3:00 |
| 2.4 | Phase 2 verification | 15 min | 3:15 |
| 3 | GPU fraction configurable (bonus) | 5 min | 3:20 |
| Final | Full test suite run + documentation | 15 min | **3:35** |

---

## Cross-Reference Checklist for AI Agent Review

**Please verify:**
- [ ] Test count (155 tests) is accurate
- [ ] File structure matches actual directory tree
- [ ] Hardcoded batch_size locations (14) are all found
- [ ] Cleanup duplication analysis is correct
- [ ] Risk assessment is reasonable
- [ ] No source code changes proposed
- [ ] Timeline estimates are realistic
- [ ] Success criteria are measurable
- [ ] Rollback plan is complete

**Questions for reviewer:**
1. Are there any batch_size hardcoded locations we missed?
2. Is the cleanup simplification safe or too aggressive?
3. Should we remove manual cleanups in this PR or separately?
4. Are there other P1 issues we should address while we're here?

---

**Ready for Implementation:** YES ✅
**Requires Modal Training Stop:** NO ✅
**Backward Compatible:** YES ✅
**Branch:** `fix/test-suite-config` (already created)