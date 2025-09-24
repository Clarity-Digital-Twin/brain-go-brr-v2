# P0: BiMamba2 CUDA Kernel Incompatibility with V3 Architecture

**⚠️ UPDATE: RESOLVED - See [P0_MAMBA2_CUDA_INCOMPATIBILITY_FIXED.md](P0_MAMBA2_CUDA_INCOMPATIBILITY_FIXED.md) for the solution**

## Executive Summary (OUTDATED)
~~**CRITICAL BLOCKER**: Our V3 dual-stream architecture is fundamentally incompatible with Mamba2's CUDA kernel requirements, causing 100% fallback to Conv1d and losing all state-space modeling benefits.~~

**✅ RESOLVED**: The issue was NOT about B*19/B*171 batching but incorrect `headdim` parameter. Fixed by setting `headdim=8` for nodes and `headdim=4` for edges to satisfy `(d_model * expand) / headdim` is multiple of 8.

## The Problem

### What's Happening
1. V3 architecture batches electrodes/edges: `(B*19, D, T)` and `(B*171, D, T)`
2. Mamba2's `causal_conv1d` CUDA kernel requires strides to be multiples of 8
3. With batch_size=8: `8*19=152` which is NOT divisible by 8
4. Kernel refuses to run, throws error, falls back to Conv1d
5. **We get ZERO Mamba benefits - just expensive convolutions**

### Error Message
```
[MAMBA] Forward pass error, using fallback: causal_conv1d with channel last layout
requires strides (x.stride(0) and x.stride(2)) to be multiples of 8
```

## Why This Matters

### What We're Losing
- ❌ **State-space modeling** - The core innovation of Mamba
- ❌ **Selective scan algorithm** - Dynamic state updates based on input
- ❌ **O(N) complexity** - Falls back to O(N*K) convolution
- ❌ **Long-range dependencies** - Conv1d has limited receptive field
- ❌ **EvoBrain's key innovation** - Temporal evolution of brain dynamics

### What We're Getting Instead
- Conv1d with kernel_size=4 (depthwise separable)
- No state maintenance across time
- No selective updates
- Just a basic CNN pretending to be Mamba

## Root Cause Analysis

### Our Architecture (V3)
```python
# Node Stream
elec_feats: (B, 19, 960, 64)  # Per-electrode features
node_flat = reshape(B*19, 64, 960)  # Batch all electrodes together
# Problem: B*19 rarely divisible by 8

# Edge Stream
edge_feats: (B, 171, 960, 1)  # Per-edge features
edge_flat = reshape(B*171, 1, 960)  # Batch all edges
# Problem: B*171 rarely divisible by 8
```

### EvoBrain's Approach
```python
# Uses Mamba1 (not Mamba2)
from mamba_ssm import Mamba  # Version 1, less strict

# Different batching - processes differently
# No explicit B*N reshaping that we do
```

### Mamba2 CUDA Requirements
1. **Tensor must be contiguous** - We added `.contiguous()` but not enough
2. **Strides must be multiples of 8** - Our B*19 and B*171 break this
3. **Memory alignment** - Complex requirements for optimal kernels
4. **Channel dimensions** - Expects certain multiples for vectorization

## Failed Attempts to Fix

### What We Tried
1. ✅ Added `.contiguous()` after reshape - **Didn't work**
2. ✅ Added `.contiguous()` after transpose - **Didn't work**
3. ✅ Added `.contiguous()` after flip - **Didn't work**
4. ✅ Changed edge d_model from 1→16 - **Didn't work**
5. ✅ Added safety assertions - **Didn't prevent fallback**

### Why They Failed
The problem isn't just contiguity - it's the fundamental dimension sizes (B*19, B*171) that don't align with CUDA's requirements.

## Potential Solutions

### Option 1: Switch to Mamba1
```python
from mamba_ssm import Mamba  # Like EvoBrain
# Pros: Less strict, proven to work
# Cons: Older, potentially slower
```

### Option 2: Process Electrodes/Edges Individually
```python
# Instead of reshape(B*19, ...)
for i in range(19):
    electrode_i = elec_feats[:, i, :, :]  # (B, 960, 64)
    # Process each electrode separately
# Pros: Clean dimensions
# Cons: Slower, no batching benefits
```

### Option 3: Pad Dimensions
```python
# Pad B*19 to next multiple of 8
# If B=8: pad 152 → 160
padded = F.pad(node_flat, (0, 0, 0, 0, 0, 8))
# Pros: Maintains batching
# Cons: Wasted computation, complex bookkeeping
```

### Option 4: Different Batch Sizes
```python
# Use batch sizes where B*19 is divisible by 8
# B=8: 8*19=152 ❌
# B=16: 16*19=304 (304/8=38) ✅
# B=32: 32*19=608 (608/8=76) ✅
# Pros: Simple
# Cons: Restricts batch size choices
```

### Option 5: Redesign Architecture
- Don't batch electrodes/edges together
- Use different memory layout
- Process time-first instead of channel-first
- **Most complex but potentially best solution**

## Impact Assessment

### Current State
- **Training**: Runs but with Conv1d fallback (not real Mamba)
- **Performance**: Missing core benefits of state-space models
- **Results**: Unknown impact on accuracy - could be significant

### If Not Fixed
- V3 is essentially "TCN + Conv + GNN" not "TCN + Mamba + GNN"
- We're not implementing EvoBrain's innovation properly
- Training is misleading - we think we have Mamba but we don't

## Recommendation

### Short Term (P0)
1. **Test with batch_size=16 or 32** where dimensions align better
2. **Try Mamba1** as immediate fallback to match EvoBrain

### Long Term
1. **Research Mamba2's exact requirements** from source code
2. **Consider architectural redesign** if necessary
3. **Benchmark Conv1d vs Mamba** to quantify impact

## Decision Required

**Do we:**
1. Accept Conv1d fallback and proceed (knowing it's not real Mamba)?
2. Switch to Mamba1 immediately?
3. Redesign the batching strategy?
4. Research and fix properly before continuing?

## Code References
- Error location: `src/brain_brr/models/mamba.py:141-151`
- Node batching: `src/brain_brr/models/detector.py:191-194`
- Edge batching: `src/brain_brr/models/detector.py:204-206`
- EvoBrain reference: `reference_repos/EvoBrain-FBC5/model/EvoBrain.py:812`

## Next Steps
1. **Immediate**: Test if batch_size=16 reduces fallback
2. **Today**: Try Mamba1 as proof of concept
3. **This week**: Make architectural decision based on findings

---

**Status**: BLOCKED - Core architecture incompatible with Mamba2 CUDA kernels
**Severity**: P0 - Defeats purpose of dual-stream state-space architecture
**Owner**: Team discussion required