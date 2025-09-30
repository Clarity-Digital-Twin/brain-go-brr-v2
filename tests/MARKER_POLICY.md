# Test Marker Policy

**Created:** 2025-09-30
**Status:** ENFORCED
**Purpose:** Ensure safe concurrent testing during GPU training

---

## Overview

This document defines when and how to use pytest markers in the test suite to ensure:
1. Tests can run safely during GPU training (no OOM)
2. GPU-heavy tests are clearly identified
3. Performance benchmarks are excluded from CI
4. Test suite behavior is predictable

---

## Marker Definitions

### `@pytest.mark.gpu`

**Use when**: Test requires GPU / CUDA operations

**Includes**:
- Tests calling `.cuda()`
- Tests calling `.to(device)` where device can be CUDA
- Tests calling `torch.cuda.*` functions
- Tests using GPU memory operations (`torch.cuda.memory_allocated()`, etc.)
- Tests creating models that auto-select GPU

**Exclusion**: Tests marked with `@pytest.mark.gpu` are excluded from:
- `make test-safe` / `make ts` (safe during training)
- Any test run with `-m "not gpu"`

**Examples**:
```python
# Class-level marker (applies to all tests in class)
@pytest.mark.gpu
class TestGPUFeatures:
    def test_cuda_forward(self):
        model = Model().cuda()
        ...

# Function-level marker
@pytest.mark.gpu
def test_single_gpu_test():
    x = torch.randn(10, 10).cuda()
    ...
```

---

### `@pytest.mark.performance`

**Use when**: Test is a performance benchmark

**Includes**:
- Latency tests
- Throughput tests
- Memory usage tests
- Stress tests
- Any test taking >30 seconds

**Exclusion**: Tests marked with `@pytest.mark.performance` are excluded from:
- `make test` (default test run)
- CI/CD pipelines
- Run explicitly with `make test-performance`

**Examples**:
```python
@pytest.mark.performance
def test_inference_latency():
    # Benchmark latency over 100 runs
    ...

@pytest.mark.performance
@pytest.mark.timeout(600)
def test_memory_scaling():
    # Test memory usage with large batches
    ...
```

---

### Combined Markers

Many performance tests also require GPU:

```python
@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_gpu_throughput():
    """GPU-heavy performance benchmark."""
    ...
```

---

## Enforcement Rules

### **RULE 1: GPU Usage MUST Be Marked**

**ANY test that uses GPU MUST be marked with `@pytest.mark.gpu`.**

**Rationale**: Tests unmarked with `@pytest.mark.gpu` will run during `make test-safe`, which is intended to be safe during concurrent training. Unmarked GPU tests cause OOM.

**Detection**:
```bash
# Find tests using GPU without marker
rg "\.cuda\(\)|\.to\(device\)|torch\.cuda\." tests/ --type py | \
    grep -v "@pytest.mark.gpu"
```

---

### **RULE 2: Performance Tests MUST Be Marked**

**ANY test that is a benchmark or takes >30s MUST be marked with `@pytest.mark.performance`.**

**Rationale**: Performance tests are slow and should not run in CI or during development.

---

### **RULE 3: Class-Level Markers Preferred**

When ALL tests in a class use GPU, mark the class:

```python
# Good (concise)
@pytest.mark.gpu
class TestGPUFeatures:
    def test_a(self): ...
    def test_b(self): ...

# Bad (repetitive)
class TestGPUFeatures:
    @pytest.mark.gpu
    def test_a(self): ...
    @pytest.mark.gpu
    def test_b(self): ...
```

---

## Make Target Behavior

### `make test`
```bash
pytest tests/ -m "not performance and not gpu" --cov
```
- **Runs**: Unit, integration, clinical tests WITHOUT GPU
- **Excludes**: Performance tests, GPU tests
- **Purpose**: Fast development cycle

### `make test-safe` / `make ts`
```bash
pytest tests/ -m "not performance and not gpu" -q
```
- **Runs**: All non-GPU, non-performance tests
- **Excludes**: Performance tests, GPU tests
- **Purpose**: Safe to run during training

### `make test-performance`
```bash
pytest tests/ -m "performance" -n 0 -v
```
- **Runs**: Only performance benchmarks
- **Requires**: Stop training first (or use `BGB_SKIP_GPU_TESTS=1`)
- **Purpose**: Benchmarking

### `make test-gpu`
```bash
pytest tests/ -v -k "mamba or cuda"
```
- **Runs**: Tests with "mamba" or "cuda" in name
- **Purpose**: Quick GPU sanity check

---

## Current Test Distribution

| Category | Count | Marker Status |
|----------|-------|---------------|
| **Total tests** | ~500 | N/A |
| **GPU tests** | ~45 | ✅ All marked |
| **Performance tests** | ~18 | ✅ All marked |
| **Both** | ~16 | ✅ All marked |

---

## Environment Variable Overrides

### `BGB_SKIP_GPU_TESTS=1`
Skip all GPU tests regardless of markers:
```bash
BGB_SKIP_GPU_TESTS=1 pytest tests/
```

### `TEST_GPU=false`
Force CPU-only mode (from `test_config.py`):
```bash
TEST_GPU=false pytest tests/
```

### `BGB_TEST_GPU_FRACTION=0.6`
Adjust GPU memory limit for tests:
```bash
BGB_TEST_GPU_FRACTION=0.6 pytest tests/  # Use 60% of GPU
```

---

## Examples from Codebase

### Unit Tests

**test_nan_robustness.py**:
```python
@pytest.mark.gpu
class TestNaNRobustness:
    """17 tests using .to(device) throughout"""
    ...
```

**test_dynamic_pe.py**:
```python
@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
@pytest.mark.gpu
class TestDynamicPE:
    """9 tests using .cuda() operations"""
    ...
```

### Performance Tests

**test_latency.py**:
```python
@pytest.mark.serial
@pytest.mark.gpu
class TestInferenceLatency:
    """GPU latency benchmarks"""

    @pytest.mark.performance
    @pytest.mark.timeout(180)
    def test_single_window_latency(self): ...
```

**test_memory.py**:
```python
@pytest.mark.serial
@pytest.mark.gpu
class TestMemoryUsage:
    """GPU memory profiling"""

    @pytest.mark.performance
    def test_inference_memory_scaling(self): ...
```

---

## Pre-Commit Hook (Future)

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: check-gpu-markers
      name: Check GPU tests have @pytest.mark.gpu
      entry: python scripts/check_gpu_markers.py
      language: python
      files: ^tests/.*\.py$
```

---

## Verification Commands

### Check GPU marker coverage:
```bash
# Should return only gpu_memory_guard.py and test files with markers
pytest --collect-only -q -m "gpu" | wc -l
```

### Check performance marker coverage:
```bash
pytest --collect-only -q -m "performance" | wc -l
```

### Find unmarked GPU usage:
```bash
rg "\.cuda\(\)|\.to\(.*device.*\)|torch\.cuda\." tests/ --type py -l | \
    xargs -I {} sh -c 'grep -L "@pytest.mark.gpu" {}'
```

---

## FAQ

**Q: My test uses `device` fixture - does it need `@pytest.mark.gpu`?**
A: YES. The `device` fixture can return CUDA device, so the test must be marked.

**Q: My test only runs on GPU if available - do I mark it?**
A: YES. If there's ANY code path that uses GPU, mark it with `@pytest.mark.gpu`.

**Q: Can I use `@pytest.mark.skipif(not torch.cuda.is_available())` instead?**
A: NO. Use `@pytest.mark.gpu` AND skipif together if needed. The marker is for exclusion, skipif is for conditional execution.

**Q: Performance test takes 5 seconds - should I mark it?**
A: If it's a benchmark, YES. If it's a regular test that happens to be slow, consider optimizing first.

---

## Changelog

**2025-09-30**: Initial policy created
- Added `@pytest.mark.gpu` to 42+ tests
- Updated `make test-safe` to exclude GPU tests properly
- Documented enforcement rules

---

**Questions?** Open an issue or PR to update this policy.