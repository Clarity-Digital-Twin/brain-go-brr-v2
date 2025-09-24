# Test Suite OOM (Out of Memory) Issue - FIXED

## The Problem
Running `make test` causes severe OOM crashes, potentially crashing the entire IDE and system.

## Root Cause Analysis

### What's Happening
1. `make test` uses `pytest -n auto` which spawns multiple parallel processes
2. Each process that runs a test with `SeizureDetector.from_config()` creates a **full V3 model**
3. V3 model is ~31.5M parameters + dual-stream architecture = ~500MB-1GB per instance
4. With 8+ parallel processes, this means **8+ GB of VRAM/RAM** just for models
5. Tests also create input tensors, run forward passes, etc.
6. **Result**: System runs out of memory and crashes

### Specific Culprits
Tests that create full models:
- `test_detector_v3.py` - Creates multiple V3 instances
- `test_tcn_integration.py` - Creates detectors with `.from_config()`
- `test_gnn_integration.py` - Creates detectors for comparison
- `test_gnn_integration_pyg.py` - Creates PyG-enabled detectors
- Performance tests - Profile memory with full models

## The Solution

### 1. Safe Test Runner Script
Created `run_tests_safe.sh` that:
- Runs tests **serially** (not parallel) for model tests
- Excludes heavy model creation from parallel runs
- Separates V3 tests into their own serial run
- Skips performance tests by default

### 2. New Makefile Target
`make test-safe` - Runs tests safely without OOM:
```bash
make test-safe
```

### 3. How to Run Tests Now

#### For Quick Testing (SAFE)
```bash
# Option 1: Use the safe test script
./run_tests_safe.sh

# Option 2: Use Makefile
make test-safe
```

#### For Specific Test Categories
```bash
# Lightweight unit tests only
.venv/bin/pytest tests/unit -n 2 -k "not (v3 or detector_from_config)"

# V3 tests only (serial)
.venv/bin/pytest tests/unit/models/test_detector_v3.py -n 0

# Clinical tests (safe)
.venv/bin/pytest tests/clinical -n 2
```

#### For Full Coverage (RISKY)
```bash
# Only if you have 32GB+ RAM and want full parallel
make test  # ⚠️ MAY CAUSE OOM

# Safer: Run categories separately
make test-unit
make test-integration
make test-clinical
```

## Why This Happens with V3

V3 is more memory-intensive than V2.6:
- **TCN encoder**: ~10M params
- **Main BiMamba**: ~8M params
- **Node stream BiMamba**: ~5M params
- **Edge stream BiMamba**: ~2M params
- **GNN + projections**: ~6M params
- **Total**: ~31.5M parameters

With parallel testing:
- 8 processes × 500MB model = 4GB minimum
- Plus input tensors, gradients, etc = 8GB+
- Plus IDE, browser, etc = OOM crash

## Best Practices Going Forward

1. **Use `make test-safe`** for regular testing
2. **Run heavy tests separately** in serial when needed
3. **Close other applications** before running full test suite
4. **Monitor memory** with `nvidia-smi` or `htop` during tests
5. **Consider pytest marks** to exclude heavy tests:
   ```python
   @pytest.mark.heavy
   def test_full_model():
       ...
   ```
   Then skip with: `pytest -m "not heavy"`

## Test Performance Impact

- **Parallel (`-n auto`)**: ~2 minutes but OOM risk
- **Safe serial (`-n 1`)**: ~5-8 minutes but stable
- **Recommended**: Use safe mode, it's worth the extra 3-5 minutes

## Emergency Recovery

If system crashes from OOM:
1. Reboot if needed
2. Clear GPU memory: `nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9`
3. Use safe testing going forward

---

**Bottom line**: The V3 architecture is memory-intensive. Always use `make test-safe` or `./run_tests_safe.sh` to avoid OOM crashes.