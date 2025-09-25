# V3 ARCHITECTURE: COMPLETE P0/P1/P2/P3 BUG ANALYSIS & TEST SUITE

## CRITICAL: WHY V3 IS COMPLETELY BROKEN

The V3 dual-stream architecture has **NEVER BEEN PROPERLY TESTED**. Every component in the pipeline has potential failure modes.

---

## P0 BUGS (TRAINING COMPLETELY BROKEN)

### P0.1: Edge Mamba Dimension Explosion
**Location**: `src/brain_brr/models/detector.py:209-221`
```python
# BUG: 1D → 16D projection with Xavier init on 171 parallel streams
edge_lifted = self.edge_in_proj(edge_flat)  # Random init, no bounds
edge_processed = self.edge_mamba(edge_lifted)  # 171 Mambas explode
```
**Fix Required**:
```python
# Add after line 215
edge_lifted = torch.clamp(edge_lifted, -5, 5)
edge_lifted = F.layer_norm(edge_lifted, edge_lifted.shape[-1:])
```

### P0.2: Dynamic PE Eigendecomposition on Garbage
**Location**: `src/brain_brr/models/gnn_pyg.py` (dynamic PE path)
```python
# BUG: Edge Mamba produces random adjacency at init
adj = assemble_adjacency(edge_weights)  # Garbage in
L = compute_laplacian(adj)  # Garbage Laplacian
eigenvalues, eigenvectors = torch.linalg.eigh(L)  # NaN/Inf
```
**Fix Required**:
```python
# Force valid adjacency
adj = torch.sigmoid(adj)  # Bound to [0,1]
adj = adj + 1e-5 * torch.eye(19)  # Add regularization
# Clamp eigenvalues
eigenvalues = torch.clamp(eigenvalues, min=1e-6, max=10)
```

### P0.3: Node Mamba Input Corruption
**Location**: `src/brain_brr/models/detector.py:197-199`
```python
# BUG: Complex reshape without validation
node_flat = elec_feats.permute(0, 1, 3, 2).reshape(
    batch_size * 19, 64, seq_len
).contiguous()
```
**Test Needed**:
```python
assert not torch.isnan(node_flat).any(), "NaN in node_flat"
assert node_flat.is_contiguous(), "Not contiguous"
assert node_flat.shape == (batch_size * 19, 64, seq_len)
```

### P0.4: GNN Message Passing Explosion
**Location**: `src/brain_brr/models/gnn_pyg.py:forward()`
```python
# BUG: No normalization in message passing
x = self.gnn_conv(x, edge_index, edge_weight)  # Can explode
```
**Fix Required**:
```python
# Add after each GNN layer
x = F.normalize(x, p=2, dim=-1)  # L2 normalize
x = torch.clamp(x, -10, 10)  # Hard clamp
```

---

## P1 BUGS (MAJOR FUNCTIONALITY BROKEN)

### P1.1: Edge Feature Computation Instability
**Location**: `src/brain_brr/models/edge_features.py:edge_scalar_series()`
```python
# BUG: Cosine similarity on unnormalized features
cos_sim = F.cosine_similarity(xi, xj, dim=-1)  # Can be ±∞
```
**Fix Required**:
```python
# Normalize before similarity
xi = F.normalize(xi, p=2, dim=-1)
xj = F.normalize(xj, p=2, dim=-1)
cos_sim = torch.clamp(F.cosine_similarity(xi, xj, dim=-1), -1, 1)
```

### P1.2: Adjacency Assembly Upper→Full Matrix
**Location**: `src/brain_brr/models/edge_features.py:assemble_adjacency()`
```python
# BUG: Upper triangular to full matrix can create feedback
adj[triu_indices] = edge_weights
adj = adj + adj.T  # Doubles diagonal!
```
**Fix Required**:
```python
adj = adj + adj.T - torch.diag(adj.diag())  # Fix diagonal
adj = torch.clamp(adj, 0, 1)  # Bound weights
```

### P1.3: Back-Projection Bottleneck
**Location**: `src/brain_brr/models/detector.py:246-249`
```python
# BUG: 1216→512 compression without normalization
bottleneck = self.proj_from_electrodes(flattened)  # Can explode
```
**Fix Required**:
```python
flattened = F.layer_norm(flattened, flattened.shape[-1:])
bottleneck = self.proj_from_electrodes(flattened)
bottleneck = torch.clamp(bottleneck, -20, 20)
```

### P1.4: Detection Head Input Scale
**Location**: `src/brain_brr/models/detector.py:284`
```python
# BUG: Decoder output clamped but still huge range
decoded = torch.clamp(decoded, -40.0, 40.0)  # Too wide!
```
**Fix Required**:
```python
decoded = torch.tanh(decoded) * 10  # Bound to [-10, 10]
```

---

## P2 BUGS (PERFORMANCE/STABILITY ISSUES)

### P2.1: Memory Layout Issues
**Problem**: Multiple permute→reshape→contiguous patterns
**Fix**: Use `.clone()` after complex reshapes

### P2.2: Mixed Precision Incompatibility
**Problem**: Eigendecomposition fails in FP16
**Fix**: Wrap all eigen ops with `with torch.cuda.amp.autocast(enabled=False)`

### P2.3: Batch Size Sensitivity
**Problem**: Larger batches → more NaNs
**Fix**: Add batch normalization after each major operation

### P2.4: Learning Rate Too High
**Problem**: 5e-5 might be too high for V3 complexity
**Fix**: Start with 1e-5 or even 1e-6

---

## P3 BUGS (MINOR/COSMETIC)

### P3.1: No Progress Logging
### P3.2: No Intermediate Checkpoint Saving
### P3.3: No Activation Statistics Tracking

---

## COMPLETE TEST SUITE NEEDED

```python
# tests/test_v3_components.py

def test_edge_mamba_initialization():
    """Verify edge Mamba doesn't explode on init"""
    model = create_v3_model()
    x = torch.randn(2, 19, 15360)

    # Hook to capture edge features
    edge_feats = []
    def hook(module, input, output):
        edge_feats.append(output)
    model.edge_mamba.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(x)

    assert len(edge_feats) > 0
    assert not torch.isnan(edge_feats[0]).any()
    assert edge_feats[0].abs().max() < 100

def test_dynamic_pe_stability():
    """Verify PE doesn't produce NaN on degenerate graphs"""
    # Test with:
    # - Zero adjacency
    # - Identity adjacency
    # - Random adjacency
    # - Near-singular adjacency
    pass

def test_node_mamba_forward():
    """Verify node stream is stable"""
    pass

def test_gnn_message_passing():
    """Verify GNN doesn't explode"""
    pass

def test_full_forward_pass():
    """End-to-end V3 forward pass"""
    model = create_v3_model()
    x = torch.randn(4, 19, 15360)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 15360)
    assert not torch.isnan(logits).any()
    assert logits.abs().max() < 100
```

---

## IMMEDIATE FIXES TO IMPLEMENT

### 1. Create Debug Forward Pass
```python
# src/brain_brr/models/detector_debug.py
class DebugSeizureDetector(SeizureDetector):
    def forward(self, x):
        print(f"Input: mean={x.mean():.3f}, std={x.std():.3f}, nan={x.isnan().any()}")

        # TCN
        tcn_out = self.tcn_encoder(x)
        print(f"TCN: mean={tcn_out.mean():.3f}, std={tcn_out.std():.3f}, nan={tcn_out.isnan().any()}")

        # Continue for each component...
        # This will IMMEDIATELY show where NaNs start
```

### 2. Add Guards EVERYWHERE
```python
# After EVERY operation in V3 path:
x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
x = torch.clamp(x, -threshold, threshold)
```

### 3. Initialize Properly
```python
# In detector.py:from_config()
# Initialize edge projection with small values
nn.init.uniform_(instance.edge_in_proj.weight, -0.01, 0.01)
nn.init.zeros_(instance.edge_in_proj.bias)
```

### 4. Test Each Component in Isolation
```bash
# Create test script
python -c "
import torch
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.config.schemas import ModelConfig

# Test just TCN
# Test just Node Mamba
# Test just Edge Mamba
# Test just GNN
# Find which component fails
"
```

---

## THE HARD TRUTH

**V3 was shipped without:**
- Unit tests for edge stream
- Integration tests for dual-stream
- Numerical stability analysis
- Gradient flow validation
- Initialization bounds checking
- Any safeguards whatsoever

**This is a P0 ARCHITECTURAL FAILURE requiring complete rewrite of the forward pass with guards at every step.**

## RECOMMENDATION

1. **IMMEDIATE**: Add test suite above
2. **TODAY**: Implement all P0 fixes
3. **THIS WEEK**: Run component-wise debugging
4. **IF STILL BROKEN**: V3 needs fundamental redesign

The architecture CAN work, but needs EXTENSIVE numerical hardening.