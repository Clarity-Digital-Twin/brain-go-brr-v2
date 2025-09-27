# PR-1: Boundary Normalization - Detailed Planning Document

## Problem Statement

The V3 architecture has **unbounded information flow** between major components:
- TCN → Projection: No normalization
- Projection → Mamba streams: No normalization
- Mamba → GNN: No normalization
- GNN → Decoder: No normalization

This causes activation magnitudes to grow exponentially, requiring 27 manual clamps (23 in-model + 4 in loss).

## Theoretical Foundation

### Literature Support

1. **"Layer Normalization" (Ba et al., 2016)**
   - Shows normalization at module boundaries prevents internal covariate shift
   - Critical for deep networks with multiple processing stages

2. **"Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)**
   - RMSNorm is more stable than LayerNorm with lower computational cost
   - Particularly effective for sequence models like ours

3. **"Going Deeper with Image Transformers" (Touvron et al., 2021)**
   - LayerScale prevents feature collapse in deep residual networks
   - Initializing residual branches with small weights (0.1) improves stability

4. **"On the Stability of Transformers" (Liu et al., 2020)**
   - Proves that architectures need normalization at component boundaries
   - Shows gradient explosion without proper normalization

## Proposed Solution

### 1. Add LayerNorm/RMSNorm at Component Boundaries

**Location Points** (5 critical boundaries):
```
1. After proj_to_electrodes (TCN → dual streams)
   - Shape: (B, 19, 960, 64)
   - Normalize over dim=64

2. After node_mamba (before GNN)
   - Shape: (B, 19, 960, 64)
   - Normalize over dim=64

3. After edge_mamba (before adjacency assembly)
   - Shape: (B, 171, 960, 16) -> projected to (B, 171, 960, 1)
   - Normalize over dim=16 before projection (recommended)

4. After GNN (before back-projection)
   - Shape: (B, 19, 960, 64)
   - Normalize over dim=64

5. Before ProjectionHead.proj (512→19)
   - Shape: (B, 512, 960)
   - Normalize over channel dim=512 (requires permute to make 512 the last dim)
```

### 2. Add LayerScale for Residual Connections

**Location Points** (2 residual merges):
```
1. Node stream residual (inside BiMamba2Layer): y = x + LayerScale(α) * F(x)
   - Place LayerScale on the residual branch inside `BiMamba2Layer` after the output projection and before dropout/addition.
   - Initial α = 0.1

2. GNN residual: y = x + LayerScale(α) * gnn(x)
   - Initial α = 0.1

Practical wiring notes

- LayerNorm application by shape:
  - (B, 19, 960, 64): last dim is 64 → apply `nn.LayerNorm(64)` directly, no permute.
  - Edge stream after Mamba: (B*E, D, T) → permute to (B*E, T, D), apply `nn.LayerNorm(D)`, permute back before `edge_out_proj`.
  - Temporal before ProjectionHead.proj: (B, 512, 960) → permute to (B, 960, 512), apply `nn.LayerNorm(512)`, permute back.

- Suggested modules in `SeizureDetector.__init__` (behind config flags):
  - `self.norm_after_proj_to_electrodes = nn.LayerNorm(64)`
  - `self.norm_after_node_mamba = nn.LayerNorm(64)`
  - `self.norm_after_edge_mamba = nn.LayerNorm(edge_d_model)`
  - `self.norm_after_gnn = nn.LayerNorm(64)`
  - `self.norm_before_projection = nn.LayerNorm(512)`
```

## Mathematical Analysis

### Before PR-1
```
Let x₀ be input with ||x₀|| = O(1)
After TCN: ||x₁|| = O(10)
After projection: ||x₂|| = O(100)
After Mamba: ||x₃|| = O(1000)
After GNN: ||x₄|| = O(10000)
→ Exponential growth requiring clamps
```

### After PR-1
```
Let x₀ be input with ||x₀|| = O(1)
After TCN: ||x₁|| = O(10)
After norm: ||x₁'|| = O(1)
After Mamba: ||x₂|| = O(10)
After norm: ||x₂'|| = O(1)
→ Bounded growth without clamps
```

## Implementation Details

### LayerNorm vs RMSNorm Choice
```python
# Preferred: Use existing LayerNorm for consistency
nn.LayerNorm(dim, eps=1e-5)

# Alternative: RMSNorm if lower overhead needed
RMSNorm(x) = γ * (x / RMS(x) + ε)
where RMS(x) = sqrt(mean(x²))

# Note: Mamba reference uses RMSNorm internally
# reference_repos/mamba/mamba_ssm/modules/block.py uses RMSNorm
```

### LayerScale Formula
```python
LayerScale(x) = α * x
where α starts at 0.1 and is learnable
```

## Configuration Schema

```yaml
model:
  norms:
    boundary_norm: str  # "layernorm" | "rmsnorm" | "none" (default: layernorm)
    boundary_eps: float  # 1e-5 (default)
    layerscale_alpha: float  # 0.1 (default)
    norm_locations:  # Fine-grained control
      after_tcn_proj: bool  # true
      after_node_mamba: bool  # true
      after_edge_mamba: bool  # true
      after_gnn: bool  # true
      before_decoder: bool  # true
```

## Expected Impact

### Stability Improvements
- **Gradient variance**: 100+ → <10
- **Activation magnitude**: Unbounded → O(1) at boundaries
- **NaN occurrences**: Every 10-20 batches → Zero

### Clamps That Can Be Retired After PR-1 (candidates)
```python
# Candidates that become redundant with boundary norms; keep initially and retire only after PR‑1/2 pass stability checks:
- tcn.py:248: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 1 output
- tcn.py:255: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 2 output
- tcn.py:262: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 3 output
- detector.py:211: features = torch.clamp(features, safe_clamp_min(), safe_clamp_max())  # TCN features
- detector.py:299: temporal = torch.clamp(temporal, safe_clamp_min(), safe_clamp_max())  # Mamba output
# Keep internal Mamba clamps initially (mamba.py:242, 248, 324, 327); consider removing only after full-stack stability proven.
```

## Validation Criteria

### Correctness
1. All norms should maintain tensor shapes
2. Gradient flow should remain unimpeded
3. No introduction of new NaN/Inf

### Performance
1. Training loss convergence rate unchanged
2. Inference latency increase < 2%
3. Memory overhead < 5%

## Risk Assessment

### Low Risk
- Fully backward compatible (can be disabled)
- Well-established techniques from literature
- Additive changes only

### Potential Issues
- Small initial performance dip while network adapts
- Need to retune learning rate (possibly lower)

## Testing Strategy

### Unit Tests
```python
test_rmsnorm_stability()  # Extreme inputs remain bounded
test_rmsnorm_gradient_flow()  # Gradients flow properly
test_layerscale_initialization()  # Correct init values
test_shape_preservation()  # Output shapes match input
```

### Integration Tests
```python
test_detector_with_norms()  # Full model stability
test_forward_backward_pass()  # E2E gradient flow
test_convergence_rate()  # Training dynamics unchanged
```

### Ablation Tests
```python
test_each_norm_location()  # Test each boundary independently
test_rmsnorm_vs_layernorm()  # Compare norm types
test_layerscale_values()  # Test different α values
```

## Rollout Plan

### Phase 1: Implementation (Day 1)
- Create `norms.py` module
- Add norm layers to detector
- Update config schema

### Phase 2: Validation (Day 2-3)
- Run unit tests
- Run 1000-batch stability test
- Compare with baseline

### Phase 3: Integration (Day 4)
- Enable in smoke.yaml first
- Gradual rollout to train.yaml
- Monitor metrics

## Success Metrics

### Must Have
- [ ] Zero NaN/Inf in 1000 consecutive batches
- [ ] Gradient norm P95 < empirically set threshold
- [ ] Maintain current accuracy

### Nice to Have
- [ ] Remove 6 manual clamps (TCN layers + detector conditionals)
- [ ] Improve convergence speed
- [ ] Reduce gradient variance by 90%

## Questions for Cross-Validation

1. Should we use RMSNorm or LayerNorm? (LayerNorm for consistency with existing architecture)
2. Should LayerScale start at 0.1 or smaller (1e-4)?
3. Should we normalize before or after residual connections?
4. Do we need normalization inside Mamba blocks too?
5. Should edge stream use different norm epsilon?

## Dependencies

- No external dependencies
- PyTorch native operations only
- Compatible with existing codebase

## Alternative Approaches Considered

1. **BatchNorm instead of RMSNorm**
   - Rejected: Batch statistics unstable with small batches

2. **No LayerScale**
   - Rejected: Residuals can still explode without scaling

3. **Normalization everywhere**
   - Rejected: Too much overhead, key boundaries sufficient

## References for Implementation

- Mamba's own use of RMSNorm: `reference_repos/mamba/mamba_ssm/modules/block.py`
- LayerScale in timm: https://github.com/rwightman/pytorch-image-models
- Transformer implementations: https://github.com/facebookresearch/deit

---

**STATUS**: Cross-validated and ready for implementation

## Cross-Validation Results

### ✅ Literature References Verified
1. **LayerNorm (Ba et al., 2016)**: Confirmed - prevents internal covariate shift, works at boundaries
2. **RMSNorm (Zhang & Sennrich, 2019)**: Confirmed - simpler than LayerNorm, used in Mamba itself
3. **LayerScale (Touvron et al., 2021)**: Confirmed - uses 0.1 initialization, controls residuals
4. **Transformer stability (Liu et al., 2020)**: Conceptually aligned

### ✅ Architecture Alignment
- **Mamba reference** (`reference_repos/mamba/mamba_ssm/modules/block.py`) uses pre-norm pattern
- **EVOBRAIN** literature shows normalization helps dual-stream architectures
- Shapes verified against actual code flow in `detector.py`

### ✅ Specific Implementation Notes
1. **Prefer LayerNorm** for consistency with existing V3 components
2. **Edge stream normalization**: Can normalize either before (dim=16) or after (dim=1) projection
3. **LayerScale init**: Start with 0.1 as per paper, tune if needed
4. **Residual connections**: Only node stream and GNN have explicit residuals currently

### ⚠️ CRITICAL MEMORY FIX DISCOVERED (Sep 27, 2025)
**Problem**: OOM errors with normalization enabled due to non-contiguous tensors after transpose
**Root Cause**: Transpose operations for normalization created memory fragmentation
**Solution**: Add `.contiguous()` after transpose operations:
```python
# Fixed implementation:
edge_processed = edge_processed.transpose(1, 2).contiguous()  # Ensures contiguous memory
edge_processed = self.norm_after_edge_mamba(edge_processed)
edge_processed = edge_processed.transpose(1, 2).contiguous()  # Maintains contiguity
```
**Impact**: Resolved OOM issues, allows batch_size=8 on RTX 4090 with full V3 architecture
