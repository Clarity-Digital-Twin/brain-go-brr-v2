# Smoke Test Problem Analysis & Professional Solution

## Problem Statement

Our smoke tests are failing because the training pipeline has a **hard exit** when no seizures are found in the sampled data. This is correct behavior for production training (prevents wasting GPU hours on useless models) but blocks smoke testing:

```python
# src/brain_brr/train/loop.py:308
if train_sampler is None:
    print("[FATAL] No seizures found in {sample_size} windows!")
    sys.exit(1)  # <- Blocks smoke tests
```

### Why This Happens

1. Smoke tests use `BGB_LIMIT_FILES=N` to reduce data for quick validation
2. The first N files alphabetically often contain no seizures
3. The balanced sampler correctly identifies no seizures and fails
4. **Both U-Net and TCN architectures fail identically** - proving TCN is integrated correctly

## Professional Solution

### Design Principles

1. **Production safety**: Real training must still fail fast on bad data
2. **Test isolation**: Smoke tests verify pipeline mechanics, not model quality
3. **Explicit behavior**: No hidden magic - clear when in test mode
4. **Architecture agnostic**: Works for any encoder (U-Net, TCN, future architectures)

### Implementation Strategy

Add a **smoke test mode** that bypasses data quality checks while maintaining all pipeline mechanics:

```python
# Environment variable for CI/testing
BGB_SMOKE_TEST=1  # Bypasses seizure checks, allows uniform sampling
```

### Key Changes

1. **Training loop** (`src/brain_brr/train/loop.py`):
```python
is_smoke_test = os.environ.get("BGB_SMOKE_TEST", "0") == "1"

if train_sampler is None:
    if is_smoke_test:
        print("[SMOKE TEST] No seizures found - using uniform sampling")
        print("[SMOKE TEST] This is ONLY acceptable for pipeline validation")
        train_sampler = None  # DataLoader will use default sampler
    else:
        print("[FATAL] No seizures found in {sample_size} windows!")
        sys.exit(1)
```

2. **Clear logging** to prevent confusion:
```python
if is_smoke_test:
    print("=" * 60)
    print("SMOKE TEST MODE - NOT FOR REAL TRAINING")
    print("Pipeline validation only - model will not learn")
    print("=" * 60)
```

3. **Config validation** remains strict - no changes needed

### Why This Is Professional

1. **Industry standard**: Major ML frameworks (TensorFlow, PyTorch Lightning) have similar test modes
2. **Explicit opt-in**: Requires deliberate environment variable, can't happen by accident
3. **Clear warnings**: Impossible to mistake smoke test for real training
4. **CI-friendly**: Easy to set in GitHub Actions, Modal, local testing
5. **Maintains invariants**: All shapes, gradients, optimizers work exactly as production

### What Smoke Tests Validate

- ✅ Model initialization with correct shapes
- ✅ Forward pass through entire architecture (TCN/U-Net → Mamba → Head)
- ✅ Loss computation (even with random labels)
- ✅ Backward pass and gradient flow
- ✅ Optimizer step and weight updates
- ✅ Checkpoint saving/loading mechanics
- ✅ Memory allocation and CUDA operations
- ❌ Model convergence (not the goal)
- ❌ Seizure detection quality (not the goal)

### Usage

```bash
# Local smoke test
export BGB_SMOKE_TEST=1
export BGB_LIMIT_FILES=2
python -m src train configs/local/smoke_tcn.yaml

# CI/CD
BGB_SMOKE_TEST=1 make test

# Production (default - will fail on bad data)
python -m src train configs/modal/train_full.yaml
```

### Alternative Approaches Considered

1. **Synthetic data generator**: Too complex, doesn't test real data pipeline
2. **Find files with seizures**: Fragile, depends on file ordering
3. **Remove check entirely**: Unsafe - real training could silently fail
4. **Mock the sampler**: Doesn't test actual sampler logic

## Implementation Priority

1. Add `BGB_SMOKE_TEST` check to training loop
2. Update smoke test configs to document this flag
3. Add to CI/CD pipelines
4. Document in README under "Testing" section

This approach is **standard practice** in ML pipelines where data quality varies but pipeline validation is critical.