# Brain-Go-Brr v2 Test Suite Analysis & Fix Strategy

## Executive Summary
The test suite is fundamentally broken due to:
1. **Hardcoded validation constraints** in fixtures that don't match actual requirements
2. **Empty fixture directories** with no actual test data
3. **Hanging tests** due to parallel execution issues
4. **Import path confusion** between tests trying to mock non-existent modules
5. **Coverage reporting broken** - tests pass but coverage shows 0%

## Current Test Structure

```
tests/
├── conftest.py           # ROOT FIXTURE FILE - HAS CRITICAL BUGS
├── fixtures/             # EMPTY - no actual fixtures!
│   └── test_configs/     # EMPTY - no configs!
├── unit/                 # Has tests but many broken
│   ├── cli/              # Tests try to mock 'src.brain_brr.eval.evaluate' which doesn't exist
│   ├── config/           # EMPTY - no config tests!
│   ├── data/             # Has tests
│   ├── events/           # Has comprehensive tests (424 lines)
│   ├── models/           # Has tests
│   ├── post/             # Has tests
│   ├── train/            # Has tests
│   └── utils/            # EMPTY - no utils tests!
├── integration/          # Has tests
├── performance/          # Hanging/timing out
└── clinical/             # Failing due to wrong assumptions
```

## Critical Issues Found

### 1. conftest.py Fixture Bugs (ROOT CAUSE OF MANY FAILURES)

```python
# CURRENT BROKEN CODE (Line 98-103):
config = ModelConfig(
    encoder=EncoderConfig(channels=[32, 64, 128, 256], stages=4),  # WRONG!
    rescnn=ResCNNConfig(n_blocks=1, kernel_sizes=[3]),             # WRONG!
    mamba=MambaConfig(n_layers=1, d_model=256, d_state=8),         # WRONG!
    decoder=DecoderConfig(stages=4, kernel_size=4),
)
return SeizureDetector(config)  # WRONG - should be from_config()
```

**ACTUAL REQUIREMENTS** (from schema validation):
- channels MUST be `[64, 128, 256, 512]`
- n_blocks MUST be `3`
- kernel_sizes MUST be `[3, 5, 7]`
- d_model MUST be `512`
- d_state MUST be `16`
- Must use `SeizureDetector.from_config(config)` not direct init

### 2. Empty Fixtures Directory
- `tests/fixtures/` exists but contains NO actual test data
- `tests/fixtures/test_configs/` exists but is EMPTY
- Tests expect fixtures that don't exist

### 3. Test Import Errors

```python
# test_cli_commands.py Line 151:
@patch("src.brain_brr.eval.evaluate.evaluate_checkpoint")
# THIS MODULE DOESN'T EXIST!
# Actual is: src.brain_brr.eval.metrics
```

### 4. Test Timeouts & Hangs
- 300s timeout hit on multiple tests
- Parallel execution with pytest-xdist causing deadlocks
- Tests stuck in `apply_hysteresis` infinite loop
- Conv operations hanging in performance tests

### 5. Coverage Path Issues
- Coverage looking for `src/brain_brr` instead of `src.brain_brr`
- Tests pass but coverage reports 0% because modules "never imported"

### 6. Test Data Mismatches

```python
# test_channel_order.py failing because:
- Expects FP1 -> Fp1 mapping in CHANNEL_SYNONYMS (doesn't exist)
- Tests expect pick_channels to be called (it's not in new implementation)
- Channel reordering tests have wrong assumptions
```

## Test Execution Results

### Stats from Last Run
- **Total tests**: 266
- **Errors**: 28 fixture errors (all `minimal_model`)
- **Failures**: 10+ actual test logic failures
- **Timeouts**: 8+ tests hanging >300s
- **Skipped**: 1

### Broken Test Categories

1. **CLI Tests** (6 ERROR):
   - All `TestCLIEvaluateCommand` tests fail on fixture
   - Mock path wrong: `src.brain_brr.eval.evaluate` doesn't exist

2. **Memory Tests** (14 ERROR):
   - All fail on `minimal_model` fixture validation

3. **Channel Order Tests** (4 FAIL):
   - Wrong assumptions about pick_channels behavior
   - Missing synonym mappings (FP1 -> Fp1)

4. **Performance Tests** (8 TIMEOUT):
   - Hanging in conv operations
   - 300s timeout on latency tests

5. **Clinical Tests** (6 TIMEOUT + 2 FAIL):
   - TAES metrics hanging in apply_hysteresis
   - Event merging logic incorrect

## Root Cause Analysis

### Why Tests Are Broken

1. **Schema Evolution**: Model configs now have strict validation, fixtures never updated
2. **Refactoring Debt**: Module paths changed but mocks not updated
3. **Missing Test Data**: Fixture directories created but never populated
4. **Parallel Execution**: WSL2 + pytest-xdist + multiprocessing = deadlocks
5. **No CI Enforcement**: Tests have been broken for multiple commits

## Fix Strategy

### Phase 1: Emergency Fixes (1 hour)

1. **Fix conftest.py immediately**:
```python
# CORRECT FIXTURE:
config = ModelConfig(
    encoder=EncoderConfig(channels=[64, 128, 256, 512], stages=4),
    rescnn=ResCNNConfig(n_blocks=3, kernel_sizes=[3, 5, 7]),
    mamba=MambaConfig(n_layers=6, d_model=512, d_state=16, conv_kernel=5),
    decoder=DecoderConfig(stages=4, kernel_size=4),
)
return SeizureDetector.from_config(config)
```

2. **Disable parallel execution temporarily**:
```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-v --tb=short"  # Remove -n auto
timeout = 30
timeout_method = "thread"
```

3. **Fix mock paths**:
```python
# Change all:
@patch("src.brain_brr.eval.evaluate.evaluate_checkpoint")
# To:
@patch("src.brain_brr.eval.metrics.evaluate_predictions")
```

### Phase 2: Test Data Setup (2 hours)

1. **Create actual fixture files**:
```python
# tests/fixtures/sample_data.py
def create_test_edf():
    """Generate valid test EDF file"""
    ...

def create_test_config():
    """Generate valid test config"""
    ...
```

2. **Populate fixtures directory**:
- Add sample EDF files
- Add test configs
- Add mock model checkpoints

### Phase 3: Fix Test Logic (4 hours)

1. **Channel order tests**: Update to match actual implementation
2. **CLI tests**: Fix mock paths and add proper fixtures
3. **Performance tests**: Add proper device handling, reduce iterations
4. **Clinical tests**: Fix apply_hysteresis infinite loops

### Phase 4: Coverage Fix (1 hour)

1. **Fix coverage config**:
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*", "*/conftest.py"]

[tool.coverage.paths]
source = ["src", ".venv/lib/*/site-packages/src"]
```

2. **Run coverage correctly**:
```bash
python -m pytest --cov=src --cov-report=term-missing
```

## Immediate Action Items

### DO THIS NOW:
1. ✅ Fix conftest.py `minimal_model` fixture (all values)
2. ✅ Fix conftest.py to use `from_config()`
3. ✅ Disable pytest-xdist temporarily
4. ✅ Fix CLI test mock paths
5. ✅ Add timeout=30 to pytest config

### DO THIS NEXT:
1. Create actual test data in fixtures/
2. Fix channel order test assumptions
3. Fix hanging tests (apply_hysteresis)
4. Update coverage configuration

### DO THIS LATER:
1. Add property-based tests
2. Re-enable parallel execution carefully
3. Add CI/CD gates
4. Document test requirements

## Test Command Progression

```bash
# Step 1: Run without parallel, with short timeout
pytest tests/unit/events -xvs --timeout=10

# Step 2: Run all unit tests
pytest tests/unit -xvs --timeout=30

# Step 3: Run with coverage
pytest tests --cov=src --cov-report=term-missing --timeout=30

# Step 4: Only after fixing, re-enable parallel
pytest tests -n auto --timeout=30
```

## Success Criteria

- [ ] All fixtures validate correctly
- [ ] No test timeouts
- [ ] Coverage reporting works (>70%)
- [ ] Tests run in <2 minutes
- [ ] CI/CD gates pass

## Conclusion

**The test suite isn't fundamentally broken in design - it's broken in execution.**

Main issues:
1. Fixtures have hardcoded values that violate current validation
2. Mock paths point to non-existent modules
3. Parallel execution causes hangs in WSL2
4. Coverage config is misconfigured

These are all fixable with systematic changes, not a rewrite.