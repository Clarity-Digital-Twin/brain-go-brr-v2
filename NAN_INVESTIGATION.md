# 🔴 CRITICAL: NaN INVESTIGATION PLAN

## THE PROBLEM
The V3 model produces NaN outputs starting at **batch 7** EVERY TIME, even with:
- Ultra-conservative learning rates (1e-4, 1e-5)
- Strong gradient clipping (0.1)
- All safeguards enabled (SAFE_CLAMP, SANITIZE_INPUTS, SANITIZE_GRADS)
- Mamba fallback to Conv1d

## SYMPTOMS OBSERVED

### Consistent Pattern
1. **Batch 1-6**: Normal training, loss ~0.1-0.2
2. **Batch 7+**: ALL outputs become NaN (61440 values = 4 batch × 15360 timesteps)
3. **Learning rate**: Proper now (6.62e-07 during warmup, not the broken 6.62e-09)
4. **Loss continues**: Even with NaN outputs, loss averages stay ~0.1 (due to sanitization)

### What We've Tried (FAILED)
- ✅ Fixed learning rate (was near-zero, now proper)
- ✅ Reduced warmup ratio (0.10 → 0.01)
- ✅ Added gradient clipping (0.1, very strong)
- ✅ Enabled all safeguards
- ✅ Used Mamba fallback (Conv1d instead of CUDA)
- ❌ STILL produces NaNs at batch 7

## ROOT CAUSE ANALYSIS

### Why Batch 7 Specifically?
This is NOT random - something DETERMINISTIC happens at batch 7:
1. **Data issue**: Specific bad sample at position 7 in shuffled dataset
2. **Accumulation**: Some value accumulates and overflows after 6 batches
3. **Architecture bug**: Component fails after processing certain amount of data

### Most Likely Culprits

#### 1. Dynamic PE (HIGHEST SUSPICION)
- Eigendecomposition can fail on certain correlation matrices
- Happens INSIDE the model forward pass
- Would explain consistent batch number
```python
# In gnn_pyg.py
eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
```

#### 2. Edge Features Calculation
- Correlation between channels can produce NaN/Inf
- Edge clamping might not be enough
```python
# In detector.py
edge_feats = F.cosine_similarity(...)  # Can produce NaN
edge_in = self.edge_in_proj(edge_feats)
edge_in = torch.clamp(edge_in, -3.0, 3.0)  # Might be too late
```

#### 3. Bad Data Window
- Specific window in dataset has extreme values
- Causes explosion in first layer (TCN)
- Propagates through entire network

## INVESTIGATION PLAN

### Phase 1: ISOLATE THE COMPONENT (30 mins)

#### Test 1: Disable GNN Completely
```yaml
# In configs/local/train.yaml
model:
  graph:
    enabled: false  # Turn off GNN entirely
```
**If this works**: Problem is in GNN/Dynamic PE

#### Test 2: Disable Dynamic PE Only
```yaml
model:
  graph:
    enabled: true
    use_dynamic_pe: false  # Static adjacency only
```
**If this works**: Problem is eigendecomposition

#### Test 3: Pure TCN+Mamba (No V3 features)
```yaml
model:
  architecture: v2  # Revert to stable V2
```
**If this works**: V3-specific issue

### Phase 2: CAPTURE THE FAILURE (15 mins)

#### Add Debug Hooks
```python
# In detector.py forward(), add after TCN:
torch.save({
    'batch_idx': batch_idx,
    'input': x.cpu(),
    'tcn_out': features.cpu(),
    'has_nan': not torch.isfinite(features).all()
}, f'debug/batch_{batch_idx}_tcn.pt')
```

#### Run with Maximum Debug
```bash
export BGB_NAN_DEBUG=1
export BGB_DEBUG_FINITE=1
export BGB_ANOMALY_DETECT=1
export TORCH_ANOMALY_DETECT=1
export CUDA_LAUNCH_BLOCKING=1

# Run for just 10 batches
timeout 300 .venv/bin/python -m src train configs/local/train.yaml
```

### Phase 3: DATA INSPECTION (15 mins)

#### Check Batch 7 Specifically
```python
# Quick script to load and inspect
import torch
import numpy as np
from src.brain_brr.data.datasets import create_datasets

train_ds, val_ds = create_datasets(config)
for i in range(10):
    windows, labels = train_ds[i]
    print(f"Sample {i}: min={windows.min():.2f}, max={windows.max():.2f}, "
          f"std={windows.std():.2f}, has_nan={torch.isnan(windows).any()}")
```

### Phase 4: NUCLEAR OPTIONS

#### Option A: Disable All Advanced Features
```yaml
model:
  architecture: tcn  # Just TCN, no Mamba, no GNN
  graph:
    enabled: false
  mamba:
    enabled: false
```

#### Option B: Try Different Hardware
- The RTX 4090 might have a specific bug
- Test on CPU only: `device: cpu`
- Or use Modal A100 directly

#### Option C: Add Aggressive Clamping
```python
# In TCN forward
x = torch.clamp(x, -10, 10)  # After EVERY layer
```

## IMMEDIATE NEXT STEPS

1. **Stop current training** (it's just wasting electricity)
   ```bash
   tmux kill-session -t local_v3_final
   ```

2. **Run Phase 1 Test 1** (Disable GNN)
   ```bash
   # Edit configs/local/train.yaml
   # Set graph.enabled: false
   # Then:
   .venv/bin/python -m src train configs/local/train.yaml 2>&1 | tee test_no_gnn.log
   ```

3. **Monitor for 200 batches**
   - If no NaNs → GNN is the problem
   - If still NaNs → Try Test 2, then Test 3

## SUCCESS CRITERIA

We'll know we've fixed it when:
- Training reaches batch 500 without any NaN warnings
- Loss decreases normally
- LR follows expected schedule
- Can re-enable features one by one

## FALLBACK PLAN

If we can't fix it locally:
1. Use Modal A100 exclusively (different hardware)
2. Revert to V2 architecture (proven stable)
3. Debug with smaller model (reduce layers/dimensions)

---

**Current Modal Status**: Cache population running (6-hour timeout)
- App: ap-rvFyQ3QyGeR0jRfS0By0a7
- Will copy 450GB (4667 train + 1832 dev files)
- Check status: `modal app list`

**Priority**: Fix NaN issue FIRST, then run Modal training