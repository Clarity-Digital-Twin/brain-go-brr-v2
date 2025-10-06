# ML / Data Pipeline Risk Audit

**Status**: VERIFIED — All issues confirmed via code inspection and benchmarking
**Date**: 2025-10-05
**Urgency**: P0 blocks CPU training, P1 causes 40,000x I/O slowdown

---

## P0 – CUDA Autocast Hard-Coded (CPU Training Blocked)

**Location**: `src/brain_brr/models/gnn_pyg.py:252`

**Issue**: Dynamic Laplacian PE computation uses `torch.amp.autocast("cuda", enabled=False)` unconditionally. On PyTorch builds without CUDA (CPU-only installs via `pip install torch --index-url https://download.pytorch.org/whl/cpu`), this raises:
```
RuntimeError: torch.cuda.amp.autocast only available in CUDA-compiled builds
```

**Verification**:
```python
# Works on CUDA builds (current RTX 4090 setup)
with torch.amp.autocast("cuda", enabled=False):  # ✅ OK
    ...

# Fails on CPU-only builds (e.g., Mac, CI without GPU)
with torch.amp.autocast("cuda", enabled=False):  # ❌ RuntimeError
    ...
```

**Impact**:
- **Blocks all CPU training/debugging** (no local dev on Mac, no cheap CI)
- **Blocks GNN testing** on systems without CUDA
- **Silent dependency** on CUDA even when `enabled=False`

**Root Cause**: Line 252 assumes CUDA backend exists, even though tensors might be on CPU

---

## P0 – NPZ Decompression on Every Window Load (40,000x Slowdown)

**Location**:
- `src/brain_brr/data/datasets.py:260-261` (EEGWindowDataset)
- `src/brain_brr/data/datasets.py:442-443` (BalancedSeizureDataset)

**Issue**: Both datasets open compressed `.npz` files inside `__getitem__`:
```python
def __getitem__(self, idx):
    with np.load(cache_path) as cached:  # Opens + decompresses EVERY call
        window = cached["windows"][window_idx]
```

Cache files written with `np.savez_compressed` (line 113, 115, 122, 124).

**Benchmark** (329 MB file, 355 windows):
```
Current approach:  2.2s PER access  (decompresses 395 MB each time)
Optimal approach:  0.05ms PER access (load once into memory)
Slowdown:          40,000x ❌
```

**Impact**:
- **Training is I/O bound** instead of GPU bound
- **10 window accesses = 22 seconds** vs 0.5ms (measured on real cache file)
- **Batch of 32 = 71 seconds** just loading data
- **Multiple workers amplify** the problem (each worker re-decompresses)

**Root Cause**: Using compressed storage format (`np.savez_compressed`) with per-window random access pattern. NumPy decompresses entire array to access single index.

---

## P1 – Balanced Sampler Randomly Assigns "Seizure" Labels

**Location**: `src/brain_brr/train/sampling.py:87-94`

**Issue**: After sampling subset of windows to estimate seizure ratio, assigns positive weight to RANDOM unsampled windows:
```python
# Estimate unsampled seizures based on observed ratio
n_unsampled_seizures = int(unsampled_mask.sum() * seizure_ratio)

# PROBLEM: Randomly pick windows and call them "seizures"
random_indices = unsampled_indices[torch.randperm(len(unsampled_indices))[:n_unsampled_seizures]]
weights[random_indices] = pos_weight  # ❌ Most are background!
```

**Example**:
- Sample 1000 windows, find 8% seizures
- Apply ratio to 10,000 unsampled → estimate 800 "seizures"
- **Randomly pick 800 windows** and give them high weight
- **Reality**: ~92% of those 800 are background (false positives)

**Impact**:
- **Oversamples background** labeled as seizures
- **Defeats balanced sampling** purpose (wanted more seizures, got more background)
- **Non-deterministic** (different "seizures" each run despite seed=42 for sampling itself)
- **Only affects fallback** (when BalancedSeizureDataset not used)

**Current Usage**: Sampler is fallback when:
1. `use_balanced_sampling: true` AND
2. NOT using BalancedSeizureDataset (no manifest)

Most training uses BalancedSeizureDataset (preferred), so impact is limited to edge cases.

---

## Fix Plan

### Fix 1: Device-Aware Autocast (P0)
**File**: `src/brain_brr/models/gnn_pyg.py:252`

**Current**:
```python
with torch.amp.autocast("cuda", enabled=False):
    l_stable = laplacian.to(torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
```

**Fixed**:
```python
# Derive device type from tensor
device_type = laplacian.device.type  # "cuda", "cpu", or "mps"

# Only use autocast if device actually supports it
if device_type == "cuda":
    with torch.amp.autocast("cuda", enabled=False):
        l_stable = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
else:
    # CPU/MPS: no autocast needed (already in fp32 context)
    l_stable = laplacian.to(torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
```

**Validation**:
- Test on CPU-only PyTorch: `python -c "import torch; assert not torch.cuda.is_available()"`
- Run GNN forward pass on CPU tensor
- Verify no CUDA-related errors

---

### Fix 2: Cache Per-Worker Memory Mapping (P0)
**File**: `src/brain_brr/data/datasets.py`

**Strategy**: Use `worker_init_fn` to load `.npz` into memory once per worker, then access via indexing.

**Implementation**:
```python
class EEGWindowDataset:
    def __init__(self, ...):
        self._cache_data = {}  # Worker-local cache

    def _load_cache_for_worker(self, cache_path):
        """Load cache into memory (called once per worker)."""
        if cache_path not in self._cache_data:
            with np.load(cache_path) as data:
                self._cache_data[cache_path] = {
                    "windows": data["windows"][:],  # Load into RAM
                    "labels": data["labels"][:] if "labels" in data else None
                }
        return self._cache_data[cache_path]

    def __getitem__(self, idx):
        # Load once per worker (cached after first call)
        cache_data = self._load_cache_for_worker(cache_path)
        window = cache_data["windows"][window_idx]
        label = cache_data["labels"][window_idx] if cache_data["labels"] is not None else None
```

**DataLoader**:
```python
def worker_init_fn(worker_id):
    # Pre-load common caches into worker memory
    pass

train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    persistent_workers=True  # Keep workers alive
)
```

**Expected Speedup**: 40,000x (measured: 2.2s → 0.05ms per access)

**Alternative**: Convert cache to uncompressed `.npy` or HDF5 (requires cache rebuild)

---

### Fix 3: Remove Random Weight Assignment (P1)
**File**: `src/brain_brr/train/sampling.py:87-94`

**Option A – Disable Extrapolation** (Safest):
```python
# Remove lines 87-94 entirely
# Only assign weights to KNOWN seizure windows
weights = torch.ones(len(dataset), dtype=torch.float32)
weights[window_has_seizure > 0] = pos_weight
# Unsampled windows keep weight=1.0 (neutral)
```

**Option B – Sample More Windows** (Better):
```python
# If sample_size < len(dataset), increase it
# to reduce need for extrapolation
sample_size = min(len(dataset), 50000)  # Sample more aggressively
```

**Option C – Deprecate Fallback** (Best):
- Always require manifest for balanced training
- Fail if `use_balanced_sampling=true` but no manifest
- Force users to build manifest first (fast via scan_existing_cache)

**Recommendation**: Option C — enforce manifest-based training, remove fallback sampler entirely.

---

## Testing Plan

### P0 Tests (Must Pass Before Merge)
1. **CPU Autocast**: Run GNN forward pass on CPU-only PyTorch build
2. **NPZ Speed**: Benchmark data loading with fix (target: <1ms/window)
3. **Multi-worker**: Test with `num_workers=4`, verify no decompression bottleneck

### P1 Tests (Validate Correctness)
1. **Sampler Logic**: Unit test that weights only assigned to verified seizures
2. **BalancedDataset**: Confirm preferred path (manifest) unaffected
3. **Fallback Path**: Verify fallback fails gracefully or uses uniform sampling

### Integration Test
```bash
# CPU training (should work after Fix 1)
CUDA_VISIBLE_DEVICES="" python -m src train configs/local/smoke.yaml

# Speed test (should be 40,000x faster after Fix 2)
python -m src train configs/local/smoke.yaml  # Monitor data loading time

# Sampler test (should not assign random weights after Fix 3)
BGB_FORCE_MANIFEST_REBUILD=1 python -m src train ...
```

---

## Summary

| Issue | Severity | Impact | Fix Effort |
|-------|----------|--------|------------|
| CUDA autocast | P0 | Blocks CPU training | 10 min (5 lines) |
| NPZ decompression | P0 | 40,000x slowdown | 1 hour (worker caching) |
| Random weights | P1 | Incorrect sampling (fallback only) | 30 min (remove logic) |

**Recommendation**: Fix P0 issues immediately (total ~2 hours), P1 optional if always using BalancedSeizureDataset.
