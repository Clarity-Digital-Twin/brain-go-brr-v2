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

### Category 1: CANDIDATES TO RETIRE (after PR‑1/2/3 validation)

These are expected to become redundant with boundary norms, bounded edge lift, and adjacency conditioning. Keep during rollout; retire only after monitoring shows zero clamp hits for 10k+ batches.

```python
# TCN (internal feature tier clamps guarded by env)
# Candidates: internal [-50,50] clamps around TCN feature maps

# Detector (with PR‑1 norms in place)
# Candidates: post‑TCN features and post‑temporal modeling safe_clamp conditionals

# Edge stream (with PR‑2 bounded lift)
# Candidate: detector.py edge_in clamp [-3,3]

# GNN (with PR‑3 conditioning)
# Candidate: eigenvalue clamp — keep initially as guard; consider reducing reliance later
```

### Category 2: KEEP FOR SAFETY (baseline guards)

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

# Mamba boundaries (keep until long‑run stability proven)
✗ mamba.py: input/intermediate/output clamps (e.g., [-10,10], [-5,5])
```

**Total: 13 clamps to keep (27 total - 14 removable = 13)**

### Category 3: SIMPLIFY nan_to_num (defer until metrics confirm)

With proper bounds, most nan_to_num become unnecessary:

```python
# Candidates to remove later (if zero NaN/Inf observed for 10k+ batches):
# tcn.py: input/feature nan_to_num
# mamba.py: input nan_to_num
# detector.py: feature/temporal nan_to_num
# gnn_pyg.py: x_batch nan_to_num before reshape

# Keep for PE safety (always)
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

### Step 3: Add Gated Fusion (node vs GNN‑enhanced)

```python
# In detector.py forward (both tensors: B, 19, T, 64):
node = node_feats
edge = elec_enhanced  # post‑GNN output carries edge information
if self.fusion_type == "gated":
    fused = self.gated_fusion(node, edge)
elif self.fusion_type == "multihead":
    fused = self.multihead_fusion(node, edge)
else:
    fused = node + edge  # Original additive fusion
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
    """Test gated fusion mechanism on per‑electrode features (64‑dim)."""
    fusion = GatedFusion(64)

    node = torch.randn(8, 19, 960, 64)
    edge = torch.randn(8, 19, 960, 64) * 0.1  # Small edge initially

    fused = fusion(node, edge)

    # Gate ∈ (0,1); fused lies between node and node+edge
    diff = (fused - node)
    assert torch.isfinite(diff).all()

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
- [ ] Model stable with 14 clamps removed
- [ ] Zero NaN/Inf in 10,000 batches
- [ ] Performance maintained or improved
- [ ] No reliance on BGB_SAFE_CLAMP environment variable

### Nice to Have
- [ ] Remove all intermediate clamps (14 total)
- [ ] Simplify to only essential math/output guards (~13 clamps)
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
- 14 clamp lines
- 6 nan_to_num lines
- ~50 lines of conditional clamping logic

### Lines Added
- 20 lines for gated fusion
- 10 lines for monitoring (temporary)

**Net reduction: ~40 lines**

## Performance Impact

### Speed Improvement
- Each clamp: ~0.1% overhead
- 14 clamps removed: ~1.4% speedup
- Gated fusion: ~0.5% overhead
- **Net: ~0.9% faster**

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
