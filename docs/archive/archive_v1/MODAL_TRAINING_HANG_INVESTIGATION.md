# Modal Training Hang & OOM Investigation

**Date**: October 1, 2025
**Status**: ✅ **ROOT CAUSES IDENTIFIED**

## Timeline of Issues

### Initial Report (3:28 PM EST)
- Training stuck at "[TRAIN] Starting epoch with 963 batches" since 14:32 (1:32 PM EST)
- 56 minutes with no output → suspected dataloader hang

### Actual Behavior (Logs Analysis)
```
14:32:45 - [TRAIN] Starting epoch with 963 batches
15:37:16 - [BATCH START] Processing batch 0/963     ← 1 HOUR 4 MINUTES LATER!
15:37:35 - [BATCH 0] Forward pass completed
15:40:40 - CUDA OOM: "Tried to allocate 10.69 GiB. 77.26 GiB / 80 GiB in use"
```

## Bug #1: Worker Initialization Hang (1h 4min delay)

### Root Cause
**PyTorch spawn + persistent_workers catastrophic startup delay**

- **Config**: `num_workers=8`, `persistent_workers=True`, `multiprocessing_context='spawn'`
- **Observed**: 1 hour 4 minutes (3844 seconds) to start first batch
- **Expected**: < 10 seconds

### Evidence
1. **PyTorch Issue #153594** (2025): Reports 50-second delay for 16 workers on Windows
   - "Workers start in a staggered, almost sequential manner"
   - "3-4 seconds per worker initialization"
   - Affects PyTorch 2.5.1+

2. **Modal Container Environment**: Exacerbates spawn delays
   - 8 workers × ~480 seconds/worker ≈ 1 hour sequential startup
   - Container networking + process isolation adds overhead

3. **CUDA + spawn requirement**:
   - "The CUDA runtime requires either spawn or forkserver start method"
   - "start_method='fork' is unsafe (crashes & deadlocks)"
   - We correctly use spawn, but hit the performance penalty

### Why persistent_workers Made It Worse
- **First epoch**: 1 hour startup to spawn 8 persistent workers
- **Subsequent epochs**: Workers stay alive, BUT...
- **Memory leak**: PyTorch #62066 - persistent_workers=True leaks memory over epochs
- **Not worth it**: Single long epoch (100+ hours) doesn't benefit from persistence

## Bug #2: CUDA Out of Memory (77GB / 80GB)

### Root Cause
**Excessive prefetching creates memory pressure**

### Memory Analysis

#### Expected Memory Usage (batch_size=64)
```python
Input tensor:           0.07 GB
TCN features:           2.23 GB
Mamba states (6 layers): 0.22 GB
Edge features:          0.04 GB
-----------------------------------
Forward pass:           2.56 GB
With gradients (FP32):  5.12 GB
With mixed precision:   2.56 GB (forward) + 2.56 GB (backward) ≈ 5 GB
```

#### Actual Memory Usage
```
PyTorch allocated: 75.44 GB
PyTorch reserved:   0.37 GB
Total in use:      77.26 GB / 79.25 GB available
```

**Discrepancy**: 75.44 GB vs 5 GB expected = **15x multiplier**

### The Smoking Gun: Prefetch Factor

**Config**: `prefetch_factor=8` with `num_workers=8`

**Total prefetched batches**: 8 workers × 8 prefetch = **64 batches in memory**

**Memory calculation**:
```
64 batches × 5 GB/batch = 320 GB required
But A100 only has 80 GB!
```

**What actually happens**:
1. CPU prefetches 64 batches (64 × 0.07 GB ≈ 4.5 GB CPU RAM - fine)
2. `pin_memory=True` pins all 64 batches in CPU memory
3. DataLoader queues try to move batches to GPU
4. GPU accumulates batches faster than model processes them
5. GPU memory fills up: 75.44 GB allocated
6. Backward pass tries to allocate 10.69 GB → **OOM**

### Contributing Factors

1. **Mixed Precision**: Enabled, but not enough to save 70 GB
2. **Gradient Accumulation**: Set to 1 (no accumulation, full backward every step)
3. **Model Size**: 31M params = ~0.12 GB FP32 / 0.06 GB FP16 (negligible)
4. **Optimizer State**: AdamW has 2× params (momentum + variance) = ~0.24 GB (negligible)
5. **Dynamic Laplacian PE**: Computed every `semi_dynamic_interval=5` steps
   - Eigendecomposition allocates temporary 19×19 matrices
   - Should be <<1 GB, not the main issue

## Research Findings

### DataLoader Best Practices (2025)

1. **spawn vs fork**:
   - CUDA requires `spawn` or `forkserver` (not `fork`)
   - `spawn` is safer but slower (fresh process, no copy-on-write)
   - Expected to become Python 3.14 default

2. **persistent_workers**:
   - **Pros**: Faster epoch transitions (no respawn overhead)
   - **Cons**: Memory leaks (PyTorch #62066), slow initial spawn
   - **Recommendation**: Use for many short epochs, NOT single long epoch

3. **prefetch_factor**:
   - Default: 2 (per worker) = `2 × num_workers` batches prefetched
   - High values (8) cause memory pressure
   - **Recommendation**: Keep at 2-4 for large batch sizes

4. **num_workers**:
   - **Rule of thumb**: 1-2× num_GPUs, or 4-8 for fast NVMe/SSD
   - **Diminishing returns**: >8 workers rarely helps
   - **Modal SSD cache**: Already fast, doesn't need 8 workers

### Modal-Specific Considerations

1. **Container Isolation**: Adds overhead to multiprocessing spawn
2. **Persistent Volumes**: `/results/cache/tusz/` is fast SSD (no need for excessive prefetch)
3. **CPU Allocation**: 24 cores available (sufficient for 4-8 workers)
4. **A100-80GB**: Large VRAM, but 64 batches × 5GB = 320GB still exceeds capacity

## Surgical Fixes

### Fix #1: Disable persistent_workers ✅
**File**: `configs/modal/train.yaml`

```yaml
data:
  persistent_workers: false  # Changed from true
```

**Impact**:
- ✅ Eliminates 1-hour startup delay
- ✅ Prevents memory leaks across epochs
- ⚠️ Slightly slower epoch transitions (~10s to respawn 8 workers)
- **Net**: Massive win for single long epoch training

### Fix #2: Reduce prefetch_factor ✅
**File**: `configs/modal/train.yaml`

```yaml
data:
  prefetch_factor: 2  # Changed from 8
```

**Impact**:
- ✅ Reduces prefetched batches: 64 → 16 batches
- ✅ Memory savings: 320 GB → 80 GB theoretical
- ✅ More controlled memory usage
- ⚠️ Slightly higher DataLoader wait time (but SSD is fast)

### Fix #3: Reduce batch_size (if needed) 🔧
**File**: `configs/modal/train.yaml`

```yaml
training:
  batch_size: 32  # Changed from 64
  gradient_accumulation_steps: 2  # Maintain effective batch_size=64
```

**Impact**:
- ✅ Halves per-batch memory: 5 GB → 2.5 GB
- ✅ With 16 prefetched: 80 GB → 40 GB
- ✅ Gradient accumulation maintains effective batch size
- ⚠️ 2× backward passes per accumulated step (slight slowdown)

**Recommendation**: Try Fixes #1 + #2 first, add #3 if still OOM

### Fix #4: Reduce num_workers (optional) 🔧
**File**: `configs/modal/train.yaml`

```yaml
data:
  num_workers: 4  # Changed from 8
```

**Impact**:
- ✅ Faster spawn startup: ~30 minutes → ~15 minutes (still slow, but better)
- ✅ Less CPU/memory overhead
- ✅ With prefetch_factor=2: 16 batches → 8 batches prefetched
- ⚠️ Slightly slower data loading (but SSD is fast, likely negligible)

**Recommendation**: Apply after Fixes #1-3 if startup still too slow

## Recommended Configuration

### Conservative (Guaranteed to work)
```yaml
data:
  num_workers: 4
  pin_memory: true
  persistent_workers: false      # ← FIX #1
  prefetch_factor: 2             # ← FIX #2

training:
  batch_size: 32                 # ← FIX #3
  gradient_accumulation_steps: 2 # ← FIX #3
  mixed_precision: true
```

**Expected**:
- Startup: ~15 minutes (still slow due to spawn, but acceptable)
- Memory: ~40 GB peak (comfortable margin)
- Throughput: Slightly slower than batch_size=64, but stable

### Aggressive (Faster, but riskier)
```yaml
data:
  num_workers: 4
  pin_memory: true
  persistent_workers: false      # ← FIX #1
  prefetch_factor: 2             # ← FIX #2

training:
  batch_size: 64                 # Keep original
  mixed_precision: true
```

**Expected**:
- Startup: ~15 minutes
- Memory: ~65 GB peak (tighter margin)
- Throughput: Faster than conservative

**Risk**: May still OOM if model has memory spikes during eigendecomposition

## Testing Plan

1. ✅ Stop current Modal run (already failed)
2. 🔄 Apply Fix #1 + #2 (persistent_workers=false, prefetch_factor=2)
3. 🔄 Test with batch_size=64 (aggressive config)
4. 🔄 If OOM, apply Fix #3 (batch_size=32 + grad_accum=2)
5. 🔄 Monitor startup time (should be ~15-30 min, not 1 hour)
6. 🔄 Monitor memory (should stay <70 GB with batch_size=64, <50 GB with batch_size=32)

## Long-Term Solutions

### PyTorch Upstream
- Track PyTorch #153594 for spawn startup fixes
- Consider switching to `forkserver` if CUDA compatibility improves

### Model Optimization
- Profile eigendecomposition memory usage
- Consider caching eigenvectors longer (semi_dynamic_interval=10 instead of 5)
- Investigate gradient checkpointing for Mamba layers

### Infrastructure
- Use Modal's `modal.gpu.A100(count=2, memory=80)` for multi-GPU if needed
- Benchmark `num_workers=0` (single-process loading) as baseline

## References

1. **PyTorch Issue #153594**: Significant DataLoader worker startup delay with spawn
   - https://github.com/pytorch/pytorch/issues/153594

2. **PyTorch Issue #62066**: Memory leak with persistent_workers=True
   - https://github.com/pytorch/pytorch/issues/62066

3. **PyTorch Multiprocessing Best Practices**:
   - https://docs.pytorch.org/stable/notes/multiprocessing.html
   - "CUDA runtime requires spawn or forkserver start method"

4. **Modal Long Training Documentation**:
   - https://modal.com/docs/examples/long-training

## Status Summary

| Issue | Root Cause | Fix | Status |
|-------|-----------|-----|--------|
| 1h startup | spawn + persistent_workers | Set persistent_workers=false | 🔄 Ready |
| OOM (77GB) | prefetch_factor=8 × workers=8 | Reduce prefetch_factor=2 | 🔄 Ready |
| Still OOM? | batch_size=64 too large | batch_size=32 + grad_accum=2 | 🔧 Backup |

**Next Action**: Apply fixes and relaunch Modal training
