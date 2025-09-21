# CUDA Performance Investigation Findings

## Summary
We discovered and resolved a major performance regression where tests were running on CPU instead of GPU, causing 50x slower inference (338ms vs 6.3ms).

## Root Cause
The issue was caused by pytest-xdist's fork-based multiprocessing conflicting with CUDA initialization. When pytest uses fork() to create worker processes, CUDA cannot be re-initialized in the forked subprocess.

## Solution
Added spawn-based multiprocessing configuration in `tests/conftest.py`:

```python
import multiprocessing

if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # Already set
```

## Performance Results
- **Before fix**: 338-374ms per inference (CPU)
- **After fix**: 6-7ms per inference (GPU with Conv1d fallback)
- **Expected with CUDA kernels**: ~2-3ms per inference

## Remaining Issues

### 1. Mamba-SSM CUDA Kernels Not Loading
The causal_conv1d CUDA kernels fail with linking errors:
```
'NoneType' object has no attribute 'causal_conv1d_fwd'
undefined symbol: _ZN3c1021throwNullDataPtrErrorEv
```

**Likely cause**: PyTorch version mismatch between compilation and runtime
**Impact**: Using Conv1d fallback (still fast at 6ms, but could be 2-3x faster)

### 2. Test Input Device Mismatches
Some tests don't properly move inputs to GPU when model is on GPU.

## Recommendations

### Short-term
1. Accept current performance (6-7ms is still excellent for real-time)
2. Run performance tests with `-n 0` or set `CUDA_VISIBLE_DEVICES=0`
3. Fix remaining test input device issues

### Long-term
1. Rebuild mamba-ssm with matching PyTorch version
2. Consider using torch.compile() for additional speedup
3. Profile with NVIDIA Nsight for further optimization

## Key Learnings
1. Always check device placement in performance-critical code
2. pytest-xdist + CUDA requires spawn multiprocessing
3. Conv1d fallback for Mamba is surprisingly performant
4. Model achieves 32.6M parameters total (not 25M as README states)

## Test Commands
```bash
# Run performance tests correctly
make tp  # or
CUDA_VISIBLE_DEVICES=0 pytest -n 0 -m performance

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```