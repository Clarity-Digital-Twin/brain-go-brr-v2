# Quick Fix Verification Plan

## Test 1: Confirm Dynamic PE is the Problem
```bash
# Current config (FAILS at batch 28)
.venv/bin/python -m src train configs/local/train.yaml

# With dynamic PE disabled (SHOULD WORK)
.venv/bin/python -m src train configs/local/train.yaml \
  --model.graph.use_dynamic_pe false
```

## Test 2: Check Semi-Dynamic Interval
```bash
# Test with updates every 10 timesteps
.venv/bin/python -m src train configs/local/train.yaml \
  --model.graph.semi_dynamic_interval 10
```

## Test 3: Verify Eigendecomposition Failure
```python
# Debug script to check eigendecomposition stability
import torch

# Simulate problematic adjacency matrix
N = 19
adj = torch.randn(N, N) * 0.01  # Small values
adj = (adj + adj.T) / 2  # Symmetric
adj = torch.sigmoid(adj) * 0.1  # Very small positive values

# Degree normalization (as in code)
degree = adj.sum(dim=-1).clamp(min=1)
D_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree))
A_norm = D_sqrt_inv @ adj @ D_sqrt_inv

# Laplacian
L = torch.eye(N) - A_norm

# Test eigendecomposition
try:
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    print(f"Min eigenvalue: {eigenvalues.min():.6f}")
    print(f"Max eigenvalue: {eigenvalues.max():.6f}")
    print(f"Contains NaN: {torch.isnan(eigenvectors).any()}")
except Exception as e:
    print(f"Eigendecomposition failed: {e}")
```

## Expected Results
1. **Current config**: Fails at batch ~28 with NaN
2. **Dynamic PE disabled**: Trains successfully past batch 100
3. **Semi-dynamic interval 10**: May work, with 10x fewer eigendecompositions
4. **Debug script**: Should show numerical instability with small adjacency values