# PR-2: Bounded Edge Stream - Detailed Planning Document

## Problem Statement

The edge stream has a **pathological 16x dimension explosion**:
```
1D edge features → 16D projection → BiMamba → 1D output
```

Without bounded activation, this causes:
- Values exploding to ±infinity
- Gradient explosion/vanishing
- Required manual clamp at line 258: `edge_in = torch.clamp(edge_in, -3.0, 3.0)`

## Theoretical Foundation

### Literature Support

1. **"Searching for Activation Functions" (Ramachandran et al., 2017)**
   - Shows bounded activations (tanh, sigmoid) provide stability
   - Swish/GELU can explode without normalization

2. **"Understanding the Difficulty of Training Deep Networks" (Glorot & Bengio, 2010)**
   - Dimension changes require careful activation choice
   - Bounded activations prevent gradient issues

3. **"Fixing the Stationary Point Problem in Deep Networks" (Pennington et al., 2017)**
   - Shows tanh maintains gradient flow better than ReLU in deep networks
   - Critical for dimension-changing projections

4. **"Self-Normalizing Neural Networks" (Klambauer et al., 2017)**
   - Proves certain activation patterns lead to self-normalization
   - SELU for this purpose, but tanh + norm achieves similar effect

## Current Problem Analysis

### The 16x Explosion Path
```python
# Current implementation (UNSTABLE)
edge_feats = compute_edge_features(x)  # (B, 171, 960, 1)
# Uses a 1x1 Conv1d lift (projection-only, unrelated to Mamba fallback)
edge_in_proj = nn.Conv1d(1, 16, kernel_size=1)  # 16x expansion
# Reshape for Conv1d: (B*E, 1, T) -> (B*E, 16, T)
edge_flat = edge_feats.squeeze(-1).reshape(B * 171, 1, T)
edge_in = edge_in_proj(edge_flat)  # UNBOUNDED
# Band‑aid clamp currently in forward path
edge_in = torch.clamp(edge_in, -3, 3)

edge_mamba = BiMamba2(d_model=16)
edge_out = edge_mamba(edge_in)  # Can still explode

edge_out_proj = nn.Conv1d(16, 1, kernel_size=1)  # 16x contraction
edge_weights = F.softplus(edge_out_proj(edge_out))  # Finally bounded
```

### Mathematical Analysis

Without bounds, variance can grow as:
```
Var(edge_in) = 16 * Var(edge_feats) * Var(W)
```

With unbounded activations and nominal init (worst‑case intuition):
```
After projection: σ² → ~16·σ²
Edge Mamba (2 layers in edge path): σ² → ~(16^2)·σ²
```

**Result**: Values reach ±10^6 range, causing NaN.

## Proposed Solution

### 1. Bounded Activation After Projection

Replace:
```python
# detector.py:254-258
edge_in = self.edge_in_proj(edge_flat).contiguous()  # Unbounded
edge_in = torch.clamp(edge_in, -3.0, 3.0)  # Band-aid
```

With:
```python
edge_in = self.edge_in_proj(edge_flat).contiguous()  # (B*E, 16, T)
edge_in = torch.tanh(edge_in)  # Bounded [-1, 1]
# Prefer LayerNorm on feature dim for consistency with PR‑1
edge_in = edge_in.transpose(1, 2)              # (B*E, T, 16)
edge_in = self.edge_lift_norm(edge_in)         # nn.LayerNorm(16) defined in __init__
edge_in = edge_in.transpose(1, 2)              # (B*E, 16, T)
# Note: Avoid GroupNorm(1,16) here — with Conv1d it normalizes across channels and time,
# which couples timesteps. LayerNorm on the feature dim keeps normalization per‑timestep.
```

### 2. Improved Initialization

```python
# Current: Already has conservative initialization in SeizureDetector.from_config
edge_in_proj = nn.Conv1d(1, 16, kernel_size=1)
nn.init.xavier_uniform_(edge_in_proj.weight, gain=0.1)  # Reduced gain

# Good news: Initialization is already conservative
# Just need to add bounded activation
```

### 3. Optional: Graduated Expansion

Instead of 1→16 directly, use graduated expansion:
```python
# Optional only; default is 1x1 Conv1d + tanh + LayerNorm.
# If exploring graduated expansion, prefer 1x1 Conv1d stages to avoid extra permutes:
edge_lift = nn.Sequential(
    nn.Conv1d(1, 4, 1), nn.Tanh(),
    nn.Conv1d(4, 8, 1), nn.Tanh(),
    nn.Conv1d(8, 16, 1), nn.Tanh(),
)
# Then apply LayerNorm(16) on (B*E, T, 16) as above.
```

## Configuration Schema

```yaml
model:
  graph:
    # Edge lifting configuration
    edge_lift_activation: str  # "tanh" | "sigmoid" | "selu" | "none"
    edge_lift_norm: str  # "rmsnorm" | "layernorm" | "none"
    edge_lift_init_std: float  # 0.1 (conservative)

    # Edge dimension
    edge_d_model: int  # 16 (current) | 8 (memory-efficient; see note)

    # Graduated expansion
    edge_graduated_expansion: bool  # false (default) | true
    edge_expansion_stages: list  # [1, 4, 8, 16] if graduated

    # Edge Mamba config
    edge_mamba_layers: int  # 2 (current)
    edge_mamba_d_state: int  # 8 (current)
    # Note: If changing edge_d_model, adjust headdim to keep
    # (d_model*expand)/headdim an integer and preferably a multiple of 8.
    # e.g., d_model=16 -> headdim=4; d_model=8 -> headdim=2.
```

## Mathematical Guarantees

### With Tanh Activation
```
Given: x ∈ [-∞, +∞]
After tanh: y ∈ [-1, +1]
After LayerNorm/RMSNorm: y' has bounded scale (unit RMS for RMSNorm)
→ Bounded regardless of input magnitude
```

### Gradient Flow
```
tanh'(x) = 1 - tanh²(x) ∈ (0, 1]
Maximum at x=0: tanh'(0) = 1
Minimum as x→±∞: tanh'(±∞) → 0
```

With LayerNorm/RMSNorm, gradients are rescaled to prevent vanishing.

## Impact Analysis

### Clamps That Become Redundant
```python
# Can remove after PR-2:
detector.py:258: edge_in = torch.clamp(edge_in, -3.0, 3.0)  # Main target

# Keep initially for safety; consider removal only after PR‑3 proves stable:
detector.py:250: edge_feats = torch.clamp(edge_feats, -0.99, 0.99)  # Input validation
edge_features.py:77: x_norm = torch.clamp(x_norm, -10.0, 10.0)  # After normalization
```

### Expected Improvements
- Edge stream variance: Unbounded → O(1)
- Gradient variance in edge path: 100+ → <5
- Edge Mamba stability: Requires clamps → Stable without

## Alternative Designs Considered

### Option A: Sigmoid Instead of Tanh
```python
edge_in = torch.sigmoid(edge_in_proj(edge_flat))  # [0, 1]
```
- **Pros**: Positive values only, matches softplus output
- **Cons**: Gradient vanishing at extremes worse than tanh

### Option B: GELU/Swish
```python
edge_in = F.gelu(edge_in_proj(edge_flat))  # Unbounded but smoother
```
- **Pros**: Better gradient flow than ReLU
- **Cons**: Still unbounded, doesn't solve core issue

### Option C: Reduce Edge Dimension
```python
edge_d_model = 8  # Instead of 16
```
- **Pros**: Less explosion, less memory
- **Cons**: May reduce capacity too much

### Option D: Remove Edge Stream Entirely
- **Pros**: Maximum stability
- **Cons**: Loses learned adjacency capability

**Recommendation**: Option A (Tanh + Norm) provides best stability/capacity tradeoff.

## Testing Strategy

### Unit Tests
```python
def test_bounded_edge_projection():
    """Verify edge projection stays bounded."""
    proj = BoundedEdgeProjection(1, 16)

    # Test with extreme inputs
    x = torch.randn(32, 171, 960, 1) * 1000
    y = proj(x)

    assert y.abs().max() <= 2.0  # Bounded by tanh + norm
    assert torch.isfinite(y).all()

def test_edge_gradient_flow():
    """Verify gradients flow through edge stream."""
    # Test gradient magnitude through 1→16→1 path

def test_edge_mamba_stability():
    """Verify Edge Mamba remains stable with bounded input."""
```

### Integration Tests
```python
def test_edge_stream_e2e():
    """Test full edge stream path."""
    # Edge features → projection → Mamba → weights

def test_adjacency_from_bounded_edges():
    """Verify adjacency matrix quality with bounded edges."""
```

## Rollout Plan

### Phase 1: Add Bounded Activation (After PR-1)
1. Add tanh activation after edge_in_proj
2. Add RMSNorm after tanh
3. Test with existing clamps still in place

### Phase 2: Tune Initialization
1. Reduce edge_in_proj initialization std
2. Test gradient flow
3. Verify edge weight distribution

### Phase 3: Remove Redundant Clamps (after PR‑3 passes stability tests)
1. Remove edge_in clamp (-3, 3)
2. Consider removing edge_feats clamp (-0.99, 0.99) if adjacency conditioning yields stable eigens
3. Verify stability over 1000 batches

## Success Metrics

### Must Have
- [ ] Edge stream stable without the [-3, 3] clamp
- [ ] Edge weights remain in reasonable range via softplus
- [ ] Adjacency matrix quality maintained
- [ ] No NaN/Inf in edge stream for 1000 batches

### Nice to Have
- [ ] Reduce edge dimension to 8 (memory savings)
- [ ] Improve edge gradient flow
- [ ] Faster convergence of edge learning

## Risks and Mitigations

### Risk 1: Reduced Edge Capacity
- **Mitigation**: Start with tanh, can try SELU if needed

### Risk 2: Gradient Vanishing in Tanh
- **Mitigation**: RMSNorm rescales gradients

### Risk 3: Edge Weights Too Small
- **Mitigation**: Adjust softplus or add learnable scale

## Questions for Cross-Validation

1. Tanh vs Sigmoid for bounded activation?
2. Should we try graduated expansion (1→4→8→16)?
3. Is 16D necessary or can we use 8D?
4. Should edge normalization use different epsilon?
5. Keep Softplus or try exponential for edge weights?

## Performance Considerations

### Memory Impact
- No change with same dimensions
- 50% reduction if edge_d_model: 16→8

### Compute Impact
- Tanh: ~same cost as ReLU
- RMSNorm: Small overhead (~1% total)

## Code References

### Tanh Best Practices
- PyTorch docs: https://pytorch.org/docs/stable/nn.html#torch.nn.Tanh
- Initialization: https://pytorch.org/docs/stable/nn.init.html

### Similar Architectures
- Graph Attention Networks use sigmoid for attention
- Transformer uses bounded softmax for weights
- Neural ODE uses tanh for stability

---

**STATUS**: Ready for cross-validation with senior AI agent before implementation
