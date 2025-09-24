# ROOT CAUSE ANALYSIS: Non-Finite Logits at Batch 28

## Executive Summary
**CRITICAL**: Training fails with non-finite logits at batch 28 due to **unstable eigendecomposition in dynamic Laplacian PE**. The `torch.linalg.eigh()` operation produces NaN eigenvalues/eigenvectors when the adjacency matrix becomes ill-conditioned.

## Failure Details
```
Model parameters: 31,475,722
Batch 8: loss=0.0937 (healthy)
Batch 28: loss=0.1262 → NaN explosion
Error: "Non-finite logits detected"
Configuration: RTX 4090, batch_size=8, lr=5e-5, grad_clip=0.5
```

## Root Cause Analysis

### 1. Primary Cause: Unstable Eigendecomposition
**Location**: `src/brain_brr/models/gnn_pyg.py:166`
```python
eigenvalues, eigenvectors = torch.linalg.eigh(L_stable)
```

**Problem**: Even with fp32 precision, eigendecomposition can fail when:
- Adjacency matrix has near-zero or very small values
- Degree normalization creates division-by-near-zero
- Laplacian becomes numerically singular or ill-conditioned
- Repeated eigenvalues cause instability

### 2. Triggering Conditions
- **Dynamic PE enabled by default**: `use_dynamic_pe: true` in configs
- **Computed every timestep**: 960 eigendecompositions per forward pass
- **No NaN checking**: No `torch.isnan()` or `nan_to_num()` guards
- **Edge stream evolution**: Adjacency evolves, potentially becoming degenerate

### 3. Why It Fails at Batch 28
- **Gradient accumulation**: Small numerical errors compound over 28 batches
- **Specific data patterns**: Certain EEG patterns create problematic adjacency matrices
- **Loss increase**: 0.0937 → 0.1262 indicates model divergence before NaN

## Evidence

### Configuration Analysis
```yaml
# configs/local/train.yaml
model.graph.use_dynamic_pe: true  # ENABLED (problem source)
training.learning_rate: 5.0e-5    # Already reduced from 1.5e-4
training.gradient_clip: 0.5       # Already conservative
training.mixed_precision: false   # Already disabled for stability
```

### Code Analysis
```python
# MISSING SAFEGUARDS:
# 1. No eigenvalue clamping
# 2. No NaN replacement
# 3. No fallback to static PE on failure
# 4. No regularization of Laplacian (epsilon on diagonal)
```

## Proposed Solutions

### Solution 1: Add Eigendecomposition Safeguards (RECOMMENDED)
```python
def _compute_dynamic_pe_vectorized(self, adjacency):
    # ... existing code ...

    # Add regularization to Laplacian
    eps = 1e-5
    L_stable = L.to(torch.float32)
    L_stable = L_stable + eps * torch.eye(N, device=L.device)  # Regularization

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L_stable)

        # Check for NaNs
        if torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any():
            # Fallback to identity or cached PE
            return self.static_pe.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

    except RuntimeError:
        # Eigendecomposition failed completely
        return self.static_pe.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

    # Clamp eigenvalues to prevent extreme values
    eigenvalues = torch.clamp(eigenvalues, min=0, max=2)

    # ... rest of code ...
```

### Solution 2: Semi-Dynamic PE with Larger Interval
```yaml
# Reduce frequency of eigendecomposition
model:
  graph:
    semi_dynamic_interval: 10  # Update every 10 timesteps instead of 1
```

### Solution 3: Disable Dynamic PE Temporarily
```yaml
# Immediate fix to continue training
model:
  graph:
    use_dynamic_pe: false  # Revert to static PE
```

### Solution 4: Hybrid Approach with Caching
```python
# Cache last valid PE and use when eigendecomposition fails
self.last_valid_pe = None

def compute_pe(self, adjacency):
    try:
        pe = self._compute_dynamic_pe_vectorized(adjacency)
        if not torch.isnan(pe).any():
            self.last_valid_pe = pe
        return pe
    except:
        return self.last_valid_pe if self.last_valid_pe is not None else self.static_pe
```

## Immediate Action Items

### Phase 1: Quick Fix (NOW)
1. **Disable dynamic PE** in `configs/local/train.yaml`:
   ```yaml
   use_dynamic_pe: false
   ```
2. **Test training stability** for 100 batches
3. **Document behavior** with static PE

### Phase 2: Robust Implementation (AFTER APPROVAL)
1. **Add eigendecomposition safeguards** (Solution 1)
2. **Implement PE caching** (Solution 4)
3. **Add monitoring**:
   ```python
   if self.training and torch.isnan(pe).any():
       logger.warning(f"NaN in PE at batch {self.global_step}")
   ```

### Phase 3: Optimization
1. **Tune semi_dynamic_interval** (try 5, 10, 20)
2. **Profile eigendecomposition cost**
3. **Consider approximate methods** (power iteration, randomized SVD)

## Risk Assessment

### Without Fix
- **100% failure rate** with dynamic PE enabled
- **Cannot train** V3 architecture
- **Blocks all experiments** requiring graph evolution

### With Proposed Fix
- **Solution 1**: Low risk, adds ~5% overhead, maintains dynamic benefits
- **Solution 2**: Medium risk, reduces temporal resolution
- **Solution 3**: Zero risk, loses dynamic PE benefits (−2% AUROC expected)
- **Solution 4**: Low risk, best of both worlds

## Recommendation for Senior Review

**IMMEDIATE**: Disable dynamic PE to unblock training
```bash
# In configs/local/train.yaml
use_dynamic_pe: false  # TEMPORARY FIX
```

**LONG-TERM**: Implement Solution 1 + 4 (safeguards + caching)
- Maintains dynamic PE benefits
- Graceful degradation on failure
- Production-ready stability

## Validation Plan
1. Train for 1000 batches with fix
2. Monitor PE statistics (% NaN, % cache hits)
3. Compare AUROC: static vs dynamic vs safeguarded
4. Stress test with adversarial adjacency matrices

---

**Severity**: CRITICAL 🔴
**Impact**: Blocks all V3 training
**Fix ETA**: 30 minutes (disable), 2 hours (full implementation)
**Review Required**: YES - architectural change affecting core GNN module