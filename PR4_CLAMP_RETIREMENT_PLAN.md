# PR-4: Clamp Retirement & Gated Fusion - Detailed Planning Document

## Problem Statement

After implementing PR-1, PR-2, and PR-3, many of the 47 manual stability interventions become redundant. This PR:
1. Removes redundant clamps systematically
2. Adds gated fusion for node/edge streams
3. Keeps only essential safety guards
4. Validates stability without band-aids

## Current State Analysis

### The 43 Interventions Breakdown (Corrected Count)

**27 Clamps** across:
- TCN: 4 clamps
- Mamba: 9 clamps
- Edge features: 7 clamps
- Detector: 6 clamps
- GNN: 1 clamp

**9 nan_to_num calls**
**6 epsilon additions**
**2 gradient sanitization paths**

After PR-1, PR-2, PR-3, we can retire ~70% of these.

## Theoretical Foundation

### Literature Support

1. **"Gated Linear Units" (Dauphin et al., 2017)**
   - Gating mechanisms prevent feature dominance
   - Learnable gates adapt to data

2. **"Highway Networks" (Srivastava et al., 2015)**
   - Gated shortcuts improve gradient flow
   - Transform gates control information flow

3. **"On the Importance of Initialization and Momentum" (Sutskever et al., 2013)**
   - Proper initialization removes need for clamps
   - Momentum helps stability

4. **"Batch Normalization" (Ioffe & Szegedy, 2015)**
   - With proper normalization, clamps become redundant
   - Only output clamps needed for loss stability

## Clamp Retirement Analysis

### Category 1: SAFE TO REMOVE (After PR-1,2,3)

These become redundant with boundary norms and bounded activations:

```python
# TCN (with PR-1 boundary norms)
✓ tcn.py:248: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 1
✓ tcn.py:255: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 2
✓ tcn.py:262: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 3

# Mamba (with PR-1 boundary norms)
✓ mamba.py:173: x = torch.clamp(x, min=-10.0, max=10.0)  # Input
✓ mamba.py:242: x_output = torch.clamp(x_output, min=-5.0, max=5.0)  # Internal
✓ mamba.py:248: output = torch.clamp(output, min=-10.0, max=10.0)  # Output
✓ mamba.py:324: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba intermediate

# Edge features (with PR-2 bounded activation)
✓ edge_features.py:77: x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)
✓ detector.py:250: edge_feats = torch.clamp(edge_feats, -0.99, 0.99)
✓ detector.py:258: edge_in = torch.clamp(edge_in, -3.0, 3.0)

# Detector intermediate (with PR-1 norms)
✓ detector.py:211: features = torch.clamp(features, safe_clamp_min(), safe_clamp_max())
✓ detector.py:299: temporal = torch.clamp(temporal, safe_clamp_min(), safe_clamp_max())

# GNN (with PR-3 adjacency conditioning)
✓ gnn_pyg.py:220: eigenvalues = torch.clamp(eigenvalues, min=1e-6, max=2.0)
```

**Total: 13 clamps can be removed**

### Category 2: KEEP FOR SAFETY

These provide essential bounds for numerical stability:

```python
# Input validation (keep for data quality)
✗ tcn.py:241: x = torch.clamp(x, min=-10.0, max=10.0)  # Initial input

# Loss computation guards (standard practice)
✗ detector.py:307: decoded = torch.clamp(decoded, -50.0, 50.0)  # Pre-loss
✗ detector.py:314: output = torch.clamp(output, -100.0, 100.0)  # Final output

# Similarity bounds (mathematical requirement)
✗ edge_features.py:81: sim = torch.clamp(sim, min=-1.0, max=1.0)  # Cosine
✗ edge_features.py:91: sim = torch.clamp(sim, min=-1.0, max=1.0)  # Correlation

# Division safety (numerical requirement)
✗ edge_features.py:73: norms = torch.clamp(norms, min=1e-6)
✗ edge_features.py:87: denom = torch.clamp(denom, min=1e-6)

# Mamba boundaries (architecture-specific)
✗ mamba.py:314: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba input
✗ mamba.py:327: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba output
```

**Total: 9 clamps to keep**

### Category 3: SIMPLIFY nan_to_num

With proper bounds, most nan_to_num become unnecessary:

```python
# Can remove (redundant with norms)
✓ tcn.py:238: x = torch.nan_to_num(...)
✓ mamba.py:170: x = torch.nan_to_num(...)
✓ mamba.py:313: x = torch.nan_to_num(...)
✓ detector.py:210: features = torch.nan_to_num(...)
✓ detector.py:298: temporal = torch.nan_to_num(...)
✓ gnn_pyg.py:349: x_batch = torch.nan_to_num(...)

# Keep for PE safety
✗ gnn_pyg.py:240: pe = torch.nan_to_num(...)  # Eigendecomp safety

# Keep for final output
✗ detector.py:306: decoded = torch.nan_to_num(...)
✗ detector.py:313: output = torch.nan_to_num(...)
```

**Remove 6, Keep 3**

## New Addition: Gated Fusion

### Motivation

Current node/edge fusion is additive:
```python
fused = node_features + edge_features  # Can cause interference
```

### Proposed Gated Fusion

```python
class GatedFusion(nn.Module):
    """Learnable gating for node/edge stream fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node: Tensor, edge: Tensor) -> Tensor:
        # Compute gate from concatenated features
        combined = torch.cat([node, edge], dim=-1)
        gate = self.sigmoid(self.gate_proj(combined))

        # Gated combination
        output = node + gate * edge
        return output
```

**Benefits**:
- Prevents edge noise from dominating
- Learnable importance weighting
- Smooth gradient flow

### Alternative: Multi-Head Gating

```python
class MultiHeadGatedFusion(nn.Module):
    """Multi-head attention-style gating."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, node: Tensor, edge: Tensor) -> Tensor:
        B, N, T, D = node.shape

        # Multi-head attention between streams
        Q = self.q_proj(node).reshape(B, N, T, self.num_heads, self.head_dim)
        K = self.k_proj(edge).reshape(B, N, T, self.num_heads, self.head_dim)
        V = self.v_proj(edge).reshape(B, N, T, self.num_heads, self.head_dim)

        # Attention weights
        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)

        # Weighted combination
        out = (attn @ V).reshape(B, N, T, D)
        return node + out
```

## Configuration Schema

```yaml
model:
  clamp_retirement:
    # Control which clamps to remove
    remove_intermediate_clamps: bool  # true (after PR-1,2,3)
    remove_nan_to_num: bool  # true (keep only critical)

    # Safety guards to always keep
    keep_input_clamp: bool  # true
    keep_output_clamp: bool  # true
    keep_loss_clamps: bool  # true

  fusion:
    # Node/edge fusion strategy
    fusion_type: str  # "add" | "gated" | "multihead"
    fusion_heads: int  # 4 (if multihead)
    fusion_dropout: float  # 0.1

  debug:
    # Validation during transition
    validate_finite: bool  # true (during rollout)
    log_clamp_hits: bool  # true (monitor what would clamp)
```

## Implementation Strategy

### Step 1: Add Monitoring (Don't Remove Yet)

```python
def monitored_clamp(x: Tensor, min_val: float, max_val: float, name: str) -> Tensor:
    """Clamp with monitoring to see if it's needed."""
    would_clamp = (x < min_val).any() or (x > max_val).any()

    if would_clamp and _env.log_clamp_hits():
        num_low = (x < min_val).sum().item()
        num_high = (x > max_val).sum().item()
        logger.info(f"Clamp {name} would affect {num_low} low, {num_high} high")

    if _env.remove_intermediate_clamps() and name in SAFE_TO_REMOVE:
        return x  # Don't actually clamp
    else:
        return torch.clamp(x, min_val, max_val)
```

### Step 2: Gradual Removal

```python
# Phase 1: Remove with fallback
if not remove_clamps or not torch.isfinite(x).all():
    x = torch.clamp(x, min_val, max_val)

# Phase 2: Remove completely
# Just delete the clamp line
```

### Step 3: Add Gated Fusion

```python
# In detector.py forward:
if self.fusion_type == "gated":
    fused = self.gated_fusion(node_out, edge_transformed)
elif self.fusion_type == "multihead":
    fused = self.multihead_fusion(node_out, edge_transformed)
else:
    fused = node_out + edge_transformed  # Original
```

## Testing Strategy

### Unit Tests

```python
def test_clamp_removal_safety():
    """Verify model stable without intermediate clamps."""
    config = {
        "norms": {"boundary_norm": "rmsnorm"},  # PR-1
        "graph": {"edge_lift_activation": "tanh"},  # PR-2
        "clamp_retirement": {"remove_intermediate_clamps": True}
    }

    model = SeizureDetector(config)
    x = torch.randn(2, 19, 15360) * 10  # Moderate input

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()
    assert output.abs().max() < 1000

def test_gated_fusion():
    """Test gated fusion mechanism."""
    fusion = GatedFusion(64)

    node = torch.randn(8, 19, 960, 64)
    edge = torch.randn(8, 19, 960, 64) * 10  # Large edge

    fused = fusion(node, edge)

    # Should be closer to node than edge initially
    node_dist = (fused - node).norm()
    edge_dist = (fused - edge).norm()
    assert node_dist < edge_dist

def test_monitor_clamp_hits():
    """Test clamp monitoring system."""
    x = torch.randn(100) * 100
    x[0] = 1000  # Would trigger clamp

    with log_capture() as logs:
        y = monitored_clamp(x, -50, 50, "test")

    assert "would affect" in logs[0]
```

### Integration Tests

```python
def test_full_model_no_clamps():
    """Test full forward pass with clamps removed."""
    # All 4 PRs enabled
    # No intermediate clamps
    # Should still be stable

def test_training_step_no_clamps():
    """Test training step with clamps removed."""
    # Forward + backward + optimizer step
    # Verify gradients finite
```

## Rollout Plan

### Phase 1: Monitor (Week 1)
1. Add monitoring to all clamps
2. Run training with monitoring
3. Identify which clamps never trigger

### Phase 2: Remove Safe Clamps (Week 1-2)
1. Remove clamps that never trigger
2. Keep safety guards
3. Validate stability

### Phase 3: Add Gated Fusion (Week 2)
1. Implement gated fusion module
2. Compare with additive fusion
3. Choose best approach

### Phase 4: Final Cleanup (Week 2-3)
1. Remove monitoring code
2. Clean up environment variables
3. Update documentation

## Success Metrics

### Must Have
- [ ] Model stable with 12+ clamps removed
- [ ] Zero NaN/Inf in 10,000 batches
- [ ] Performance maintained or improved
- [ ] No reliance on BGB_SAFE_CLAMP environment variable

### Nice to Have
- [ ] Remove all intermediate clamps (15+ total)
- [ ] Simplify to only essential math/output guards (~11 clamps)
- [ ] Improved gradient flow metrics

## Risk Assessment

### Risk 1: Premature Removal
- **Impact**: NaN returns
- **Mitigation**: Gradual removal with monitoring

### Risk 2: Edge Case Instability
- **Impact**: Rare NaN on unusual inputs
- **Mitigation**: Keep input/output guards

### Risk 3: Gradient Issues
- **Impact**: Training instability
- **Mitigation**: Keep gradient clipping in optimizer

## Questions for Cross-Validation

1. Should we remove clamps one at a time or in groups?
2. Is gated fusion necessary or just nice to have?
3. Should monitoring be permanent or temporary?
4. Keep any nan_to_num besides PE and output?
5. What's the minimum set of safety guards?

## Expected Code Reduction

### Lines Removed
- 13 clamp lines
- 6 nan_to_num lines
- ~50 lines of conditional clamping logic

### Lines Added
- 20 lines for gated fusion
- 10 lines for monitoring (temporary)

**Net reduction: ~40 lines**

## Performance Impact

### Speed Improvement
- Each clamp: ~0.1% overhead
- 13 clamps removed: ~1.3% speedup
- Gated fusion: ~0.5% overhead
- **Net: ~0.8% faster**

### Memory Impact
- Negligible change
- Gated fusion adds small parameters

## Final State

After all 4 PRs, the architecture will have:

### Essential Guards Only
```python
# Input validation
x = torch.clamp(input, -10, 10)  # Data quality

# Math requirements
sim = torch.clamp(sim, -1, 1)  # Cosine bounds
denom = torch.clamp(denom, min=1e-6)  # Division safety

# Output guards
output = torch.clamp(output, -100, 100)  # Loss stability
```

### Clean Architecture
- Boundary normalization (PR-1)
- Bounded edge stream (PR-2)
- Conditioned adjacency (PR-3)
- Gated fusion (PR-4)
- **No band-aids, stable by construction**

---

**STATUS**: Ready for cross-validation with senior AI agent before implementation