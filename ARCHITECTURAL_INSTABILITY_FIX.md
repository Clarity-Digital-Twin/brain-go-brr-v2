# V3 Architectural Instability: Root Cause Analysis & Surgical Fix Plan

## Executive Summary

The V3 dual-stream architecture contains **43 numerical stability interventions** (27 clamps + 9 nan_to_num + 6 epsilon additions + 2 gradient sanitization paths) across the forward pass, indicating fundamental architectural instability. This document provides:
1. Complete inventory of all stability band-aids
2. Root cause analysis based on literature
3. Surgical fix plan to make V3 "stable by construction"
4. Implementation roadmap with 4 targeted PRs

**Key Finding**: The instability is NOT a bug - it's an architectural problem from unbounded information flow between components.

## Part 1: Complete Inventory of Stability Band-Aids

### 1.1 Manual Clamps (27 instances total: 23 in-model + 4 in loss)

#### TCN Module (`src/brain_brr/models/tcn.py`)
```python
Line 241: x = torch.clamp(x, min=-10.0, max=10.0)  # Input clamping
Line 248: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 1 output (conditional)
Line 255: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 2 output (conditional)
Line 262: x = torch.clamp(x, min=-50.0, max=50.0)  # Layer 3 output (conditional)
```

#### Mamba Module (`src/brain_brr/models/mamba.py`)
```python
Line 173: x = torch.clamp(x, min=-10.0, max=10.0)  # Input to Mamba block
Line 242: x_output = torch.clamp(x_output, min=-5.0, max=5.0)  # Internal output
Line 248: output = torch.clamp(output, min=-10.0, max=10.0)  # Final output
Line 314: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba input
Line 324: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba intermediate
Line 327: x = torch.clamp(x, min=-10.0, max=10.0)  # BiMamba final
```

#### Edge Features Module (`src/brain_brr/models/edge_features.py`)
```python
Line 73: norms = torch.clamp(norms, min=1e-6)  # Division safety
Line 77: x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)  # Normalized features
Line 81: sim = torch.clamp(sim, min=-1.0, max=1.0)  # Cosine similarity
Line 87: denom = torch.clamp(denom, min=1e-6)  # Division safety
Line 91: sim = torch.clamp(sim, min=-1.0, max=1.0)  # Final similarity
```

#### GNN Module (`src/brain_brr/models/gnn_pyg.py`)
```python
Line 220: eigenvalues = torch.clamp(eigenvalues, min=1e-6, max=2.0)  # Laplacian eigenvalues
Line 350: x_batch = torch.clamp(x_batch, safe_clamp_min(), safe_clamp_max())  # Output
```

#### Detector Module (`src/brain_brr/models/detector.py`)
```python
Line 211: features = torch.clamp(features, safe_clamp_min(), safe_clamp_max())  # TCN features
Line 250: edge_feats = torch.clamp(edge_feats, -0.99, 0.99)  # Edge features
Line 258: edge_in = torch.clamp(edge_in, -3.0, 3.0)  # Edge projection
Line 299: temporal = torch.clamp(temporal, safe_clamp_min(), safe_clamp_max())  # Mamba output
Line 307: decoded = torch.clamp(decoded, -50.0, 50.0)  # Decoder output
Line 314: output = torch.clamp(output, -100.0, 100.0)  # Final logits
```

### 1.2 NaN/Inf Replacements (9 instances in-model)

```python
# TCN
tcn.py:238: x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# Mamba
mamba.py:170: x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
mamba.py:313: x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# GNN
gnn_pyg.py:240: pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)
gnn_pyg.py:349: x_batch = torch.nan_to_num(x_batch, nan=0.0, posinf=0.0, neginf=0.0)

# Detector
detector.py:210: features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
detector.py:298: temporal = torch.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)
detector.py:306: decoded = torch.nan_to_num(decoded, nan=0.0, posinf=50.0, neginf=-50.0)
detector.py:313: output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
```

### 1.3 Loss-Level Clamps (4 instances in training loop)

```python
# Training loop (src/brain_brr/train/loop.py)
Line 205: logits_clamped = logits.clamp(min=-100, max=100)  # Pre-loss logits
Line 212: p = p.clamp(min=1e-6, max=1 - 1e-6)  # Probability bounds
Line 218: p_t_stable = p_t.clamp(min=1e-7, max=1 - 1e-7)  # Focal term stability
Line 223: focal_loss = focal_loss.clamp(max=100.0)  # Prevent loss explosion
```

### 1.4 Gradient Sanitization (2 paths)

```python
# Training loop (src/brain_brr/train/loop.py)
Lines 694-709: Mixed precision gradient sanitization
Lines 728-745: Standard gradient sanitization
# Mechanism: param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
```

### 1.5 Epsilon Additions (6 instances)

```python
gnn_pyg.py:159: degrees = a_flat.sum(dim=-1).clamp_min(1e-6)
gnn_pyg.py:176-177: l_stable = l_stable + eps * torch.eye(...)  # eps=1e-4
gnn_pyg.py:184-185: eps = 1e-3  # Fallback regularization

edge_features.py:73: min=1e-6  # Norm safety
edge_features.py:86: + 1e-6  # Sqrt safety
edge_features.py:87: min=1e-6  # Denominator safety
```

### 1.6 Environment Variable Controls

```python
BGB_SANITIZE_GRADS=1      # Gradient sanitization
BGB_SAFE_CLAMP=1          # Activation clamping
BGB_SAFE_CLAMP_MIN=-10    # Clamp lower bound (default: -10.0)
BGB_SAFE_CLAMP_MAX=10     # Clamp upper bound (default: 10.0)
BGB_NAN_DEBUG=1           # NaN monitoring
BGB_SKIP_OPT_STEP_ON_NAN=1  # Skip corrupted updates
```

## Part 2: Root Cause Analysis

### 2.1 Primary Issue: Unbounded Information Flow

The V3 architecture has **NO inherent bounds** between components:

```
TCN (unbounded ReLU) →
  Projection (linear, no norm) →
    Node Mamba (19 parallel, no cross-regulation) +
    Edge Mamba (171 parallel, 1→16→1 explosion) →
      Dynamic PE (eigendecomp on learned matrix) →
        GNN (message passing) →
          Decoder (linear projection)
```

**Literature Reference**: "On the Stability of Transformers" (Liu et al., 2020) shows that architectures need normalization at component boundaries to prevent gradient explosion.

### 2.2 Dimension Explosion in Edge Stream

The edge stream has a pathological expansion:
```
1D edge features → 16D projection (16x expansion!) → BiMamba → 1D output
```

Without bounded activation or normalization, this causes exponential growth.

**Literature Reference**: "Understanding the Difficulty of Training Deep Networks" (Glorot & Bengio, 2010) - dimension changes require careful initialization and normalization.

### 2.3 Dynamic Eigendecomposition Instability

Computing eigendecomposition on a **changing adjacency matrix** every forward pass:
- Condition numbers can explode
- Eigenvectors can flip signs randomly
- Small input changes → large eigenvalue changes

**Literature Reference**: "Numerical Linear Algebra" (Trefethen & Bau, 1997) - eigendecomposition of learned matrices is inherently unstable without regularization.

### 2.4 Missing Architectural Regularization

Comparison with stable architectures:

| Architecture | Normalization | Bounded Activations | Residual Scaling | Clamps Needed |
|-------------|--------------|-------------------|-----------------|---------------|
| Transformer | LayerNorm everywhere | Softmax attention | LayerScale | 0 |
| ResNet | BatchNorm every block | ReLU with BN | Identity shortcuts | 0 |
| EfficientNet | BatchNorm + SE blocks | Swish (bounded) | Weighted residuals | 0 |
| **V3 (current)** | Sparse/missing | Unbounded ReLU | No scaling | **22+** |

## Part 3: Surgical Fix Plan - "Stable by Construction"

Based on literature and successful architectures, here's how to fix V3 **without removing features**:

### 3.1 Theoretical Foundation

From "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015) and "Layer Normalization" (Ba et al., 2016):
- Normalization at component boundaries prevents internal covariate shift
- Bounded activations prevent gradient explosion
- Residual scaling prevents feature collapse

### 3.2 The 4-PR Fix Strategy

#### PR-1: Boundary Normalization (Seams & Scales)

**Theory**: Add RMSNorm (from "Root Mean Square Layer Normalization", Zhang & Sennrich 2019) at component boundaries.

**Implementation**:
```python
# Add RMSNorm at these seams:
1. After proj_to_electrodes (TCN → streams)
2. After node_mamba (before fusion)
3. After edge_mamba (before adjacency)
4. After GNN (before decoder)
5. Before decoder head

# Add LayerScale (from "Going Deeper with Image Transformers", Touvron et al. 2021):
y = x + α * f(x) where α = 0.1 initially
```

**Config**:
```yaml
model:
  norms:
    boundary_norm: "rmsnorm"  # rmsnorm|layernorm|none
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
```

#### PR-2: Bounded Edge Stream

**Theory**: Replace unbounded expansion with bounded activation (from "Searching for Activation Functions", Ramachandran et al. 2017).

**Implementation**:
```python
# Replace:
edge_in = edge_in_proj(edge_flat)  # Unbounded
edge_in = torch.clamp(edge_in, -3, 3)  # Band-aid

# With:
edge_in = torch.tanh(edge_in_proj(edge_flat))  # Bounded [-1, 1]
edge_in = RMSNorm(D, eps=1e-5)(edge_in)  # Normalized
```

**Config**:
```yaml
model.graph:
  edge_lift_activation: "tanh"  # tanh|sigmoid|none
  edge_lift_norm: "rmsnorm"  # rmsnorm|layernorm|none
```

#### PR-3: Adjacency Matrix Conditioning

**Theory**: Well-conditioned graph matrices from "Spectral Networks and Deep Locally Connected Networks" (Bruna et al., 2014).

**Implementation**:
```python
# Row-wise softmax (from "Graph Attention Networks", Veličković et al. 2018):
A = softmax(A / τ, dim=-1)  # τ = temperature

# EMA smoothing (from "Momentum Contrast", He et al. 2020):
A_smooth = β * A_prev + (1-β) * A  # β = 0.9

# Strict symmetrization:
A = (A + A.T) / 2
```

**Config**:
```yaml
model.graph:
  adj_row_softmax: true
  adj_softmax_tau: 1.0
  adj_ema_beta: 0.9
  adj_force_symmetric: true
```

#### PR-4: Gating & Clamp Retirement

**Theory**: Gated fusion from "Gated Linear Units" (Dauphin et al., 2017).

**Implementation**:
```python
# Gated node/edge fusion:
gate = sigmoid(MLP([node_feats || edge_feats]))
fused = node_feats + gate * edge_feats

# Remove these clamps (now redundant):
- edge [-3,3] clamp → covered by tanh
- internal [-50,50] clamps → covered by RMSNorm
- Keep only: decoder [-40,40] and focal loss [1e-6, 1-1e-6]
```

## Part 4: Implementation Roadmap

### Phase 1: Merge PR-1 & PR-2 (Week 1)
- Add boundary norms and bounded edge lift
- Keep existing clamps temporarily
- Verify stability for 1000 batches
- Expected: 70% reduction in NaN occurrences

### Phase 2: Enable PR-3 Features (Week 1-2)
- Turn on adjacency conditioning
- Monitor condition numbers
- Expected: Dynamic PE stability without fallbacks

### Phase 3: Merge PR-4 (Week 2)
- Remove redundant clamps
- Add gated fusion
- Expected: Clean training without environment variables

### Phase 4: Performance Validation (Week 2-3)
- Compare metrics with/without fixes
- Measure latency impact (expected: <5% overhead)
- Verify reproducibility across seeds

## Part 5: Success Metrics

### Stability Metrics
- [ ] Zero NaN/Inf in 10,000 consecutive batches (forward + backward)
- [ ] Gradient norm P95 < K (set empirically per stack)
- [ ] No reliance on BGB_SAFE_CLAMP or gradient sanitization in steady state
- [ ] Dynamic LPE fallback rate < 0.1% (PE computation succeeds without cached fallback)

### Performance Metrics
- [ ] Maintain or improve seizure detection sensitivity
- [ ] Inference latency increase < 5%
- [ ] Memory usage increase < 10%
- [ ] Training convergence in same number of epochs

## Part 6: Literature References

1. **State-Space Models & Normalization**:
   - Gu & Dao (2023/2024): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - Gu et al. (2022): "Efficiently Modeling Long Sequences with Structured State Spaces (S4)"
   - Zhang & Sennrich (2019): "Root Mean Square Layer Normalization"
   - Reference implementation: `reference_repos/mamba` uses RMSNorm before gates

2. **Graph Neural Networks & Positional Encoding**:
   - Dwivedi et al. (2020): "Benchmarking Graph Neural Networks" (Laplacian PE)
   - Veličković et al. (2018): "Graph Attention Networks" (row-softmax normalization)
   - Klicpera et al. (2019): "APPNP: Predict then Propagate" (α-mixing, SSGConv)
   - Local evidence: `literature/markdown/EVOBRAIN.md` - two-stream Mamba + LapPE

3. **Numerical Stability**:
   - Trefethen & Bau (1997): "Numerical Linear Algebra" (conditioning, eigendecomposition)
   - Touvron et al. (2021): "Going Deeper with Image Transformers" (LayerScale)
   - He et al. (2020): "Momentum Contrast" (EMA for stability)

4. **EEG-Specific Architecture**:
   - Local research: `literature/markdown/EVOBRAIN.md` - dual-stream architecture
   - Local research: `literature/markdown/EEMG2` - Mamba-2 blocks with LayerNorm
   - Implementation: Our V3 architecture implements similar concepts

## Part 7: Code Implementation Examples

### 7.1 Boundary RMSNorm Implementation

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        x = x / (rms + self.eps)
        return self.scale * x
```

### 7.2 LayerScale Implementation

```python
class LayerScale(nn.Module):
    """Learnable scaling of residual branches"""
    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x
```

### 7.3 Bounded Edge Projection

```python
class BoundedEdgeProjection(nn.Module):
    """Edge projection with bounded activation and normalization"""
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super().__init__()
        # Use Conv1d to match current shape handling
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # GroupNorm works on channel dimension without permutes
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 171, 960, 1) -> reshape for Conv1d
        B, E, T, C = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * E, C, T)  # (B*E, 1, T)

        x = self.proj(x)  # (B*E, 16, T)
        x = torch.tanh(x)  # Bounded [-1, 1]
        x = self.norm(x)    # Normalized

        # Reshape back
        x = x.reshape(B, E, -1, T).permute(0, 1, 3, 2)  # (B, 171, 960, 16)
        return x
```

### 7.4 Well-Conditioned Adjacency

```python
def condition_adjacency(
    A: torch.Tensor,
    top_k: int = 5,
    tau: float = 1.0,
    ema_beta: Optional[float] = None,
    prev_A: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Condition adjacency matrix for stability with masked row-softmax."""
    B, T, N, _ = A.shape

    # Apply top-k mask first
    topk_vals, topk_idx = torch.topk(A, k=min(top_k, N), dim=-1)
    mask = torch.zeros_like(A)
    mask.scatter_(-1, topk_idx, 1.0)
    A_masked = A * mask

    # Masked row-wise softmax (only over kept neighbors)
    # Add large negative value to masked out elements
    A_for_softmax = A_masked / tau
    A_for_softmax = A_for_softmax.masked_fill(mask == 0, -1e9)
    A_norm = F.softmax(A_for_softmax, dim=-1)

    # EMA smoothing if enabled (for PE computation)
    if ema_beta is not None and prev_A is not None:
        A_norm = ema_beta * prev_A + (1 - ema_beta) * A_norm

    # Force symmetry (important for Laplacian)
    A_sym = (A_norm + A_norm.transpose(-2, -1)) / 2

    # Add small identity to prevent singular Laplacian
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A_final = A_sym + 1e-4 * I.unsqueeze(0).unsqueeze(0)

    return A_final
```

## Part 8: Testing Strategy

### Unit Tests for Each Fix
```python
def test_boundary_norm_stability():
    """Verify RMSNorm prevents explosion"""
    x = torch.randn(32, 19, 960, 64) * 100  # Large input
    norm = RMSNorm(64)
    y = norm(x)
    assert torch.isfinite(y).all()
    assert y.std() < 10  # Bounded variance

def test_edge_projection_bounds():
    """Verify edge projection stays bounded"""
    proj = BoundedEdgeProjection(1, 16)
    x = torch.randn(32, 171, 960, 1) * 1000  # Huge input
    y = proj(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 2.0  # Bounded by tanh + norm

def test_adjacency_conditioning():
    """Verify adjacency normalization and stability"""
    A = torch.randn(2, 10, 19, 19) * 10  # Random adjacency
    A_cond = condition_adjacency(A, top_k=5)

    # Test row sums approximately 1 (for non-zero rows)
    row_sums = A_cond.sum(dim=-1)
    assert torch.allclose(row_sums[row_sums > 0.1], torch.ones_like(row_sums[row_sums > 0.1]), atol=0.1)

    # Test symmetry
    sym_error = (A_cond - A_cond.transpose(-2, -1)).abs().max()
    assert sym_error < 1e-6

    # Test finite values
    assert torch.isfinite(A_cond).all()

    # Test Laplacian can be computed without NaN
    L = torch.eye(19).unsqueeze(0).unsqueeze(0) - A_cond
    eigenvalues = torch.linalg.eigvalsh(L.to(torch.float32))
    assert torch.isfinite(eigenvalues).all()
```

## Part 9: Rollout Checklist

### Week 1
- [ ] Create feature branch: `fix/architectural-stability`
- [ ] Implement PR-1: Boundary normalization
- [ ] Implement PR-2: Bounded edge stream
- [ ] Run smoke tests with existing clamps
- [ ] Verify 1000 batches without NaN

### Week 2
- [ ] Enable PR-3: Adjacency conditioning
- [ ] Implement PR-4: Gating and clamp removal
- [ ] Run full epoch test
- [ ] Compare metrics with baseline

### Week 3
- [ ] Performance benchmarking
- [ ] Documentation update
- [ ] Code review
- [ ] Merge to main

## Conclusion

The V3 architecture's 43 stability interventions (27 clamps + 9 nan_to_num + 6 epsilon additions + 2 gradient sanitization paths) are symptoms of missing architectural regularization. By adding principled bounds at component seams (RMSNorm/LayerNorm), bounded activations (tanh), conditioned adjacency matrices (masked row-softmax + EMA), and gated fusion, we can retire ~60% of manual clamps while improving stability.

This is not about removing features or reducing capacity - it's about making the existing architecture **stable by construction** using proven techniques from literature.

**The path forward is clear**: Surgical fixes at component boundaries, not band-aids throughout the forward pass.

---

*"An architecture that needs 43 manual interventions is telling you something. Listen to it."*

## Appendix: Complete Intervention Inventory with File:Line References

### Clamps (27 total)
**In-Model (23):**
- `src/brain_brr/models/tcn.py:241` - Input clamp [-10, 10]
- `src/brain_brr/models/tcn.py:248` - Layer 1 conditional clamp [-50, 50]
- `src/brain_brr/models/tcn.py:255` - Layer 2 conditional clamp [-50, 50]
- `src/brain_brr/models/tcn.py:262` - Layer 3 conditional clamp [-50, 50]
- `src/brain_brr/models/mamba.py:173` - Mamba input clamp [-10, 10]
- `src/brain_brr/models/mamba.py:242` - Internal output clamp [-5, 5]
- `src/brain_brr/models/mamba.py:248` - Final output clamp [-10, 10]
- `src/brain_brr/models/mamba.py:314` - BiMamba input clamp [-10, 10]
- `src/brain_brr/models/mamba.py:324` - BiMamba intermediate clamp [-10, 10]
- `src/brain_brr/models/mamba.py:327` - BiMamba final clamp [-10, 10]
- `src/brain_brr/models/edge_features.py:73` - Norm safety clamp [1e-6, inf]
- `src/brain_brr/models/edge_features.py:77` - Normalized features clamp [-10, 10]
- `src/brain_brr/models/edge_features.py:81` - Cosine similarity clamp [-1, 1]
- `src/brain_brr/models/edge_features.py:87` - Denominator safety clamp [1e-6, inf]
- `src/brain_brr/models/edge_features.py:91` - Final similarity clamp [-1, 1]
- `src/brain_brr/models/gnn_pyg.py:220` - Eigenvalues clamp [1e-6, 2.0]
- `src/brain_brr/models/gnn_pyg.py:350` - Output conditional clamp (env-based)
- `src/brain_brr/models/detector.py:211` - TCN features conditional clamp
- `src/brain_brr/models/detector.py:250` - Edge features clamp [-0.99, 0.99]
- `src/brain_brr/models/detector.py:258` - Edge projection clamp [-3, 3]
- `src/brain_brr/models/detector.py:299` - Mamba output conditional clamp
- `src/brain_brr/models/detector.py:307` - Decoder output clamp [-50, 50]
- `src/brain_brr/models/detector.py:314` - Final logits clamp [-100, 100]

**Loss-Level (4):**
- `src/brain_brr/train/loop.py:205` - Logits pre-loss clamp [-100, 100]
- `src/brain_brr/train/loop.py:212` - Probability bounds [1e-6, 1-1e-6]
- `src/brain_brr/train/loop.py:218` - Focal term stability [1e-7, 1-1e-7]
- `src/brain_brr/train/loop.py:223` - Focal loss max clamp [0, 100]

### NaN/Inf Replacements (9 in-model)
- `src/brain_brr/models/tcn.py:238`
- `src/brain_brr/models/mamba.py:170`
- `src/brain_brr/models/mamba.py:313`
- `src/brain_brr/models/gnn_pyg.py:240`
- `src/brain_brr/models/gnn_pyg.py:349`
- `src/brain_brr/models/detector.py:210`
- `src/brain_brr/models/detector.py:298`
- `src/brain_brr/models/detector.py:306`
- `src/brain_brr/models/detector.py:313`

### Epsilon Additions (6)
- `src/brain_brr/models/gnn_pyg.py:159` - Degree clamping
- `src/brain_brr/models/gnn_pyg.py:176-177` - Laplacian regularization
- `src/brain_brr/models/gnn_pyg.py:184-185` - Fallback regularization
- `src/brain_brr/models/edge_features.py:73` - Norm safety
- `src/brain_brr/models/edge_features.py:86` - Sqrt safety
- `src/brain_brr/models/edge_features.py:87` - Denominator safety

### Gradient Sanitization (2 paths)
- `src/brain_brr/train/loop.py:694-709` - Mixed precision path
- `src/brain_brr/train/loop.py:728-745` - Standard path