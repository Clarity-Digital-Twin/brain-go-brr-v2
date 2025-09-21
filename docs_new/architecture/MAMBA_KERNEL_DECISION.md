# Mamba d_conv Kernel Size Decision

## Summary
We use d_conv=5 in configs/docs but coerce to 4 for CUDA. **We could have just used 4 everywhere.**

## Why We Chose d_conv=5

### Temporal Resolution Analysis
- At 256 Hz sampling rate:
  - Kernel 5 = ~19.5ms temporal window (5/256 ≈ 0.0195 seconds)
  - Kernel 4 = ~15.6ms temporal window (4/256 ≈ 0.0156 seconds)
  - Kernel 3 = ~11.7ms temporal window (3/256 ≈ 0.0117 seconds)

### Clinical Rationale
- ~20ms window captures high-frequency epileptiform activities (20-70 Hz)
- Matches middle kernel in ResCNN stack [3, 5, 7] for multi-scale consistency
- Balances local temporal pattern capture with computational efficiency

## CUDA Kernel Limitation

### The Hard Constraint
```python
# mamba-ssm CUDA kernels only support d_conv in {2, 3, 4}
# This is a hardware optimization limitation in the CUDA implementation
```

### Our Workaround
```python
# In src/brain_brr/models/mamba.py:
self.d_conv = d_conv  # Public value = 5
self._mamba_conv_k = d_conv if d_conv in (2,3,4) else 4  # CUDA gets 4
```

## The Truth: We Should Have Used 4

### Why 4 Would Have Been Better
1. **CUDA is primary target** - We train on GPUs (A100s), not CPUs
2. **Negligible difference** - 15.6ms vs 19.5ms both capture same patterns
3. **Simpler code** - No coercion logic, no warnings, no dual paths
4. **Industry standard** - Vision Mamba, original Mamba use d_conv=4

### Why We Didn't
- Early decision picked 5 (matching ResCNN middle kernel)
- Documentation written with 5
- Tests written expecting 5
- Changing became more work than coercing

## Lessons Learned

**For future architectures:**
1. Check hardware constraints FIRST
2. Don't let early decisions become technical debt
3. The "theoretically optimal" value means nothing if hardware coerces it anyway

## Current State (DO NOT CHANGE)

Keep d_conv=5 in configs because:
- All documentation references it
- Tests expect it
- CPU fallback uses it
- Changing would require updating 50+ files

Just remember: **We're really using 4 on GPU anyway.**