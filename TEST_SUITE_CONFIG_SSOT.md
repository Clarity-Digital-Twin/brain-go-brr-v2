# Test Suite Configuration - Single Source of Truth (SSOT)

**Created:** 2025-09-30
**Branch:** `fix/upgrade-mamba`
**Investigation:** Deep audit of test markers, GPU usage, and Make target behavior
**Status:** ⚠️ CRITICAL ISSUES FOUND

---

## Executive Summary

**CRITICAL FINDING**: The test suite has a fundamental marker inconsistency causing GPU OOM during training despite recent OOM prevention work.

**Root Cause**: **26 tests use GPU without `@pytest.mark.gpu`**, so they:
- Run during `make test` (not excluded by `-m "not gpu"`)
- Attempt GPU allocation if `torch.cuda.is_available()`
- Can OOM during concurrent training
- Bypass the new training-detection OOM prevention

**Immediate Impact**:
- `make test` is NOT safe during training (contrary to documentation)
- `make ts` (training-safe) only covers unit tests, misses integration tests
- Performance tests need GPU but only 1/18 is marked with `@pytest.mark.gpu`

---

## Test Suite Inventory

### By Numbers

| Category | Count | Details |
|----------|-------|---------|
| **Total test functions/classes** | 504 | Across 42 test files |
| **Tests marked `@pytest.mark.performance`** | 18 | All in `tests/performance/` |
| **Tests marked `@pytest.mark.gpu`** | 16 | Scattered across integration/unit/clinical |
| **Tests with BOTH markers** | 1 | Only `test_latency.py:215` |
| **Tests using GPU WITHOUT marker** | **26+** | **test_nan_robustness.py (17), test_dynamic_pe.py (9)** |
| **Performance tests using GPU** | 18 | ALL performance tests use `torch.cuda.*` |

### Critical Unmarked GPU Usage

| File | Tests | GPU Usage | Marker Status |
|------|-------|-----------|---------------|
| `tests/unit/models/test_nan_robustness.py` | 17 | `.to(device)` throughout | ❌ NO MARKER |
| `tests/unit/models/test_dynamic_pe.py` | 9 | `.cuda()` direct calls | ❌ NO MARKER |
| `tests/performance/test_latency.py` | 9 | `torch.cuda.synchronize()`, etc. | ❌ Only 1/9 marked |
| `tests/performance/test_memory.py` | 9 | `torch.cuda.memory_stats()`, etc. | ❌ Only 1/9 marked |

**Total Unmarked**: **42+ tests** that use GPU but aren't excluded by `make test`

---

## Current Make Target Behavior

### What Actually Runs

```bash
# Makefile line 38-44
make test:
  pytest -n 1 -m "not performance and not gpu" tests/unit
  pytest -n 1 -m "not performance and not gpu" tests/integration
  pytest -n 1 tests/clinical  # NO MARKER FILTER!
```

**Excludes:**
- 18 tests marked `@pytest.mark.performance`
- 16 tests marked `@pytest.mark.gpu`

**DOES NOT EXCLUDE:**
- 26+ tests using `.cuda()` or `.to(device)` without markers
- Clinical tests (always run, may use GPU)
- Integration tests checking `torch.cuda.is_available()`

**Result**: `make test` will attempt GPU usage on ~30+ tests if GPU is available!

### Other Targets

```bash
make test-safe (line 75-78):
  pytest -n 1 -m "not performance and not gpu" tests/unit -q
  # Only unit tests, misses integration!

make test-performance (line 62-65):
  pytest -n 0 -m performance -v
  # WARNING added but tests still try to run

make test-gpu (line 67-69):
  pytest -n 1 -v -k "mamba or cuda"
  # Uses keyword matching, not markers!
```

---

## Design Confusion Analysis

### Problem 1: Opportunistic GPU Usage

**Pattern Found**: Most tests follow this antipattern:

```python
def test_something(device):  # device fixture from conftest
    model = Model().to(device)  # Uses GPU if available
    x = torch.randn(...).to(device)
    output = model(x)
    # ... no @pytest.mark.gpu decorator!
```

**Impact**: Tests "opportunistically" use GPU without declaring it via marker.

### Problem 2: Marker Inconsistency

**`@pytest.mark.performance`**:
- ✅ Applied to all performance benchmarks (18 tests)
- ❌ NOT applied to integration/unit tests that happen to be slow
- ⚠️ Performance tests also need GPU but only 1 has `@pytest.mark.gpu`

**`@pytest.mark.gpu`**:
- ✅ Applied to some GPU-heavy integration tests (16 tests)
- ❌ NOT applied to unit tests using `.cuda()` (26+ tests)
- ❌ NOT applied to most performance tests (17/18 missing)
- ⚠️ Inconsistent criteria (why mark some but not others?)

### Problem 3: Fixture Design Encourages Unmarked GPU

**From `conftest.py:147`:**
```python
@pytest.fixture
def minimal_model():
    model = SeizureDetector.from_config(config)
    model.eval()
    # Don't auto-move to CUDA - let tests do it explicitly  ← Problem!
    yield model
```

**Issue**: Fixtures don't control GPU placement, tests do it ad-hoc.

**From `conftest.py:364-371`:**
```python
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, request):
    monkeypatch.setenv("BGB_LIMIT_FILES", "2")
    # No TEST_DEVICE or GPU control!
```

**Result**: Every test independently decides to use GPU via `.cuda()` or `.to(device)`.

### Problem 4: Recent OOM Prevention Doesn't Help

**What we added (gpu_memory_guard.py:72-84)**:
```python
if training_active or available_memory < 10:
    # Training detected: use minimal allocation (3GB)
    torch.cuda.set_per_process_memory_fraction(0.12, 0)
```

**Why it doesn't help**:
1. Only limits TOTAL memory allocation (3GB vs 10GB)
2. Doesn't prevent 30+ tests from TRYING to use GPU
3. Tests can still OOM even with 3GB if they allocate large models
4. Doesn't skip GPU tests, just limits their memory

---

## Root Causes

### 1. Missing Marker Policy

**No documented policy for when to add `@pytest.mark.gpu`**.

Current usage is inconsistent:
- Some tests marked because they're "GPU-heavy" (subjective)
- Unit tests rarely marked even when using `.cuda()`
- Performance tests assume GPU but don't mark it

**Recommendation**: EVERY test that calls `.cuda()`, `.to(device)`, or `torch.cuda.*` MUST be marked.

### 2. Opportunistic GPU Usage Pattern

**Tests written during development with GPU always available**.

Pattern:
```python
# No marker, but uses GPU if present
def test_foo():
    if torch.cuda.is_available():
        model = Model().cuda()  # OOMs during training!
```

**Should be**:
```python
@pytest.mark.gpu
def test_foo():
    model = Model().cuda()
    # Or use CPU-only fixture when GPU not needed
```

### 3. Make Target Naming Confusion

**`make test` implies "full test suite" but excludes GPU+performance tests**.

Users expect:
- `make test` = everything (including GPU if available)
- `make test-cpu` = CPU only
- `make test-gpu` = GPU only
- `make test-safe` = safe during training

Reality:
- `make test` = "everything except explicitly marked GPU/perf" (but includes unmarked GPU!)
- `make test-safe` = "unit tests without markers" (incomplete)

### 4. test_config.py Not Enforced

**`TEST_USE_GPU` and `TEST_DEVICE` exist but aren't used in fixtures**.

```python
# test_config.py line 13-14
TEST_USE_GPU = torch.cuda.is_available() and os.getenv("TEST_GPU", "auto") != "false"
TEST_DEVICE = "cuda" if TEST_USE_GPU else "cpu"
```

**But conftest.py doesn't import or use these!**

Fixtures should use `TEST_DEVICE` instead of letting tests call `.cuda()` directly.

---

## Proposed Solutions

### Option A: Fix Markers (Conservative)

**Add `@pytest.mark.gpu` to all 42+ unmarked GPU tests**.

**Pros**:
- Minimal code changes
- Preserves existing test logic
- Make targets work as intended

**Cons**:
- Tedious (42+ files to edit)
- Easy to regress (future tests might not add marker)
- Doesn't fix root cause (opportunistic pattern)

**Effort**: 3-4 hours

---

### Option B: Centralized Device Control (Recommended)

**Change fixture to control device placement, not individual tests**.

**Implementation**:

1. **Add device control fixture**:
```python
# conftest.py
@pytest.fixture
def device():
    """Device for test execution - respects TEST_DEVICE from test_config."""
    from tests.test_config import TEST_DEVICE
    return torch.device(TEST_DEVICE)

@pytest.fixture
def minimal_model(device):
    """Model automatically placed on correct device."""
    model = SeizureDetector.from_config(config)
    model.eval()
    return model.to(device)  # Controlled by TEST_DEVICE
```

2. **Remove ad-hoc `.cuda()` calls**:
```python
# BEFORE (tests/unit/models/test_nan_robustness.py):
def test_tcn_with_nan_input(self, device):
    tcn = TCNEncoder(init_gain=0.2).to(device)  # Ad-hoc

# AFTER:
def test_tcn_with_nan_input(self, minimal_tcn_model):
    # Model already on correct device from fixture
    tcn = minimal_tcn_model
```

3. **Add `TEST_GPU=false` support**:
```bash
# Force CPU-only tests (overrides torch.cuda.is_available())
TEST_GPU=false make test
```

4. **Update Make targets**:
```makefile
test-cpu: ## CPU-only tests (explicit)
	TEST_GPU=false $(PYTEST) tests/ -q

test-safe: ## Safe during training (CPU + unit/integration)
	TEST_GPU=false $(PYTEST) -m "not performance" tests/unit tests/integration -q

test: ## Full test suite (uses GPU if available)
	$(PYTEST) tests/ --cov=src --cov-report=html
```

**Pros**:
- Centralized control (one place to set CPU/GPU)
- Prevents future regressions (fixtures enforce device)
- Tests remain portable (work on CPU or GPU)
- Clear semantics (`TEST_GPU=false` vs markers)

**Cons**:
- Requires refactoring many tests (~50+ files)
- Changes test behavior (some tests may break)
- More invasive than Option A

**Effort**: 1-2 days

---

### Option C: Hybrid Approach (Pragmatic)

**Combine marker fixes with gradual fixture migration**.

**Phase 1 (Immediate - 3 hours)**:
1. Add `@pytest.mark.gpu` to 42 unmarked tests
2. Update `make test-safe` to exclude GPU tests properly
3. Document marker policy in `tests/README.md`

**Phase 2 (After training completes - 1-2 days)**:
4. Refactor fixtures to use `TEST_DEVICE`
5. Migrate high-value tests to device fixtures
6. Remove ad-hoc `.cuda()` calls gradually
7. Add linter rule to catch `.cuda()` without marker

**Pros**:
- Immediate fix for training OOM issue
- Gradual migration reduces risk
- Can validate Phase 1 before Phase 2

**Cons**:
- Two-phase effort (more commits)
- Tests in inconsistent state during migration

**Effort**: 3 hours now + 1-2 days later

---

## Recommended Action Plan

**RECOMMENDATION: Option C (Hybrid)**

### Phase 1: Emergency Marker Fix (This PR)

**Goal**: Make `make test` safe during training within 3 hours.

**Tasks**:
1. Add `@pytest.mark.gpu` to:
   - `tests/unit/models/test_nan_robustness.py` (17 tests)
   - `tests/unit/models/test_dynamic_pe.py` (9 tests)
   - `tests/performance/test_latency.py` (8 missing)
   - `tests/performance/test_memory.py` (8 missing)
   - Any integration tests using `.cuda()` without marker

2. Update Makefile:
```makefile
test-safe: ## Safe during training (excludes ALL GPU tests)
	@echo "${CYAN}Running training-safe tests (CPU only)...${NC}"
	@echo "${YELLOW}Excludes all GPU and performance tests${NC}"
	$(PYTEST) -n 1 -m "not performance and not gpu" tests/unit tests/integration tests/clinical -q
```

3. Add `tests/MARKER_POLICY.md`:
```markdown
# Test Marker Policy

## When to use `@pytest.mark.gpu`

**Rule**: ANY test that uses GPU MUST be marked with `@pytest.mark.gpu`.

**Includes**:
- Tests calling `.cuda()`
- Tests calling `.to(device)` where device can be CUDA
- Tests calling `torch.cuda.*` functions
- Tests creating models that auto-select GPU

**Enforcement**:
- Run `make test-safe` during training
- Run `make test` when GPU is free
- CI runs with `TEST_GPU=false` for CPU-only validation

## Markers

- `@pytest.mark.gpu` - Requires GPU, excluded from `make test-safe`
- `@pytest.mark.performance` - Slow benchmarks, excluded from `make test`
- `@pytest.mark.serial` - Must run serially (no -n)
- `@pytest.mark.clinical` - Clinical validation tests
```

4. Verification:
```bash
# Should pass during training
make test-safe

# Should list GPU tests (42+)
pytest --collect-only -q -m "gpu"

# Should exclude GPU tests
pytest --collect-only -q -m "not gpu" | wc -l  # Should be ~460
```

### Phase 2: Fixture Refactoring (Next Sprint)

**Goal**: Eliminate ad-hoc `.cuda()` calls, centralize device control.

**Tasks** (detailed plan in separate document):
1. Add `device` fixture using `TEST_DEVICE`
2. Update model fixtures to use `device`
3. Migrate tests to use fixtures instead of `.cuda()`
4. Add pre-commit hook to catch `.cuda()` without `@pytest.mark.gpu`
5. Update documentation

---

## Immediate Next Steps

1. **Create branch**: `fix/test-suite-markers`
2. **Add markers** to 42 tests (list below)
3. **Update Makefile** `test-safe` target
4. **Add MARKER_POLICY.md**
5. **Verify** with training running in background
6. **Commit** and merge to all branches

**Files to Edit (Phase 1)**:
- `tests/unit/models/test_nan_robustness.py` - Add `@pytest.mark.gpu` to class
- `tests/unit/models/test_dynamic_pe.py` - Add `@pytest.mark.gpu` to tests
- `tests/performance/test_latency.py` - Add to 8 tests
- `tests/performance/test_memory.py` - Add to 8 tests
- `Makefile` - Update `test-safe` target
- `tests/MARKER_POLICY.md` - Create new file

**Estimated Time**: 3 hours
**Risk**: Low (adding markers doesn't change test behavior)
**Validation**: Run `make test-safe` during training (should not OOM)

---

## Appendix: Full Test Inventory

### Tests Marked `@pytest.mark.gpu` (16)

```
tests/clinical/test_taes_metrics.py:365
tests/integration/test_model_assembly.py:82
tests/integration/test_streaming.py:142
tests/integration/test_tcn_integration.py:114
tests/integration/test_tcn_integration.py:151
tests/integration/test_tcn_integration.py:191
tests/integration/test_training_edge_cases.py:51
tests/integration/test_training_edge_cases.py:130
tests/integration/test_training_edge_cases.py:184
tests/integration/test_training_edge_cases.py:256
tests/integration/test_training_edge_cases.py:310
tests/integration/test_training_edge_cases.py:333
tests/integration/test_training_edge_cases.py:369
tests/performance/test_latency.py:215
tests/unit/models/test_tcn.py:107
```

### Tests Marked `@pytest.mark.performance` (18)

```
tests/performance/test_latency.py:60
tests/performance/test_latency.py:140
tests/performance/test_latency.py:176
tests/performance/test_latency.py:214 (also has @pytest.mark.gpu)
tests/performance/test_latency.py:260
tests/performance/test_latency.py:355
tests/performance/test_latency.py:407
tests/performance/test_latency.py:458
tests/performance/test_latency.py:529
tests/performance/test_memory.py:45
tests/performance/test_memory.py:89
... (more in these files)
```

### Tests Using GPU WITHOUT Marker (26+)

**test_nan_robustness.py** (17 tests):
- All tests in `TestNaNRobustness` class use `.to(device)`
- Device fixture allows CUDA if available
- NO `@pytest.mark.gpu` on class or methods

**test_dynamic_pe.py** (9 tests):
- Multiple tests use `.cuda()` directly
- Lines 115, 122, etc.
- NO markers

**test_latency.py** (8 tests):
- 9 performance tests, only 1 marked with `@pytest.mark.gpu`
- All use `torch.cuda.synchronize()` if available

**test_memory.py** (8 tests):
- 9 performance tests, only 1 marked with `@pytest.mark.gpu`
- All use `torch.cuda.memory_stats()` and similar

---

## Questions for Review

1. **Is Option C (Hybrid) the right approach?** Or should we go full Option B immediately?

2. **Should `make test` include GPU tests by default?** Or should it match CI (CPU-only)?

3. **Should performance tests be marked with BOTH `@pytest.mark.gpu` AND `@pytest.mark.performance`?**

4. **Should we add a pre-commit hook to enforce marker policy?**

5. **Is 3 hours realistic for Phase 1?** (Add 42 markers manually)

---

**Status**: Ready for decision and implementation
**Blocker**: Training running, need safe way to run tests concurrently
**Priority**: P0 - Blocking development workflow

---

**Last Updated**: 2025-09-30
**Reviewed By**: Pending
**Approved By**: Pending