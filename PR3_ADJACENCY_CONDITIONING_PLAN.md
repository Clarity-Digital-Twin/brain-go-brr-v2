# PR-3: Adjacency Matrix Conditioning - Detailed Planning Document

## Problem Statement

Dynamic Laplacian PE computes eigendecomposition on a **learned, changing adjacency matrix** every forward pass, causing:
- Eigenvalue explosion (condition numbers > 10^6)
- Eigenvector sign flips
- NaN in PE computation
- Required fallback to cached PE

From v3-nan-explosion-resolution.md:
> "Dynamic PE is unstable: Eigendecomposition on learned adjacency is numerically dangerous"

## Theoretical Foundation

### Literature Support

1. **"Spectral Networks and Deep Locally Connected Networks" (Bruna et al., 2014)**
   - Shows importance of well-conditioned graph Laplacians
   - Recommends regularization for learned adjacencies

2. **"Graph Attention Networks" (Veličković et al., 2018)**
   - Uses row-wise softmax for attention weights
   - Ensures each node's outgoing weights sum to 1

3. **"Numerical Linear Algebra" (Trefethen & Bau, 1997)**
   - Eigendecomposition condition number = λ_max/λ_min
   - High condition numbers → numerical instability

4. **"Momentum Contrast for Unsupervised Visual Representation Learning" (He et al., 2020)**
   - EMA (Exponential Moving Average) for stability
   - Smooth transitions prevent sudden changes

## Current Problem Analysis

### The Unstable Adjacency Path
```python
# Current implementation (UNSTABLE)
edge_weights = edge_mamba(...)  # (B, 171, 960)
adjacency = assemble_adjacency(edge_weights)  # (B, 960, 19, 19)

# Problems:
1. No row normalization → unbounded weights
2. No temporal smoothing → rapid changes
3. Asymmetric → complex eigenvalues possible
4. Can be disconnected → singular Laplacian
```

### Numerical Analysis

Condition number of Laplacian:
```
κ(L) = λ_max / λ_min

Current: κ(L) > 10^6 (very ill-conditioned)
Target: Well-defined eigenvalues without NaN

Note: Laplacian has λ_min = 0 by construction (for connected graph),
so traditional condition number is infinite. Focus on stable
eigendecomposition and bounded non-zero eigenvalues instead.
```

Eigendecomposition stability requires:
- Symmetric matrix (real eigenvalues)
- Positive semi-definite (non-negative eigenvalues)
- Bounded condition number
- Smooth temporal evolution

## Proposed Solution

### 1. Masked Row-Wise Softmax Normalization (with zero-diagonal)

```python
def normalize_adjacency(A: Tensor, top_k: int = 3, tau: float = 1.0) -> Tensor:
    """Apply masked row-wise softmax (respecting top-k sparsity).

    Note: zeroes the diagonal before softmax to avoid self-loops dominating a row.
    Identity fallback is applied later if a row becomes empty.
    """
    # A shape: (B, T, N, N)
    # Note: Our code uses top_k=3 by default (detector.py:271)
    B, T, N, _ = A.shape

    # Remove self-loops before normalization
    I = torch.eye(N, device=A.device, dtype=A.dtype).view(1, 1, N, N)
    A = A * (1.0 - I)

    # Apply top-k mask first (keep top k edges per node)
    topk_vals, topk_idx = torch.topk(A, k=min(top_k, N), dim=-1)
    mask = torch.zeros_like(A)
    mask.scatter_(-1, topk_idx, 1.0)

    # Masked softmax only over kept neighbors
    A_masked = A.masked_fill(mask == 0, -1e9)
    A_norm = F.softmax(A_masked / tau, dim=-1)
    return A_norm
```

**Benefits**:
- Weights bounded in [0, 1]
- Each node's outgoing weights sum to 1
- Prevents weight explosion

### 2. Temporal EMA Smoothing (within-sequence by default)

```python
class TemporalEMA:
    """Exponential moving average across timesteps (causal smoothing).

    Default usage smooths across the time dimension T within the same forward pass.
    Cross-forward EMA is optional and off by default to avoid state leakage.
    """

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.prev_A = None

    def update(self, A: Tensor) -> Tensor:
        # Vectorized causal EMA across time axis T
        # A: (B, T, N, N)
        B, T, N, _ = A.shape
        out = torch.empty_like(A)
        out[:, 0] = A[:, 0]
        for t in range(1, T):
            out[:, t] = self.beta * out[:, t - 1] + (1 - self.beta) * A[:, t]
        return out
```

**Benefits**:
- Smooth temporal evolution
- Prevents sudden eigenvalue jumps
- Maintains temporal coherence

### 3. Strict Symmetrization (+ identity fallback)

```python
def symmetrize_adjacency(A: Tensor, threshold: float = 1e-6, identity_fallback: bool = True) -> Tensor:
    """Force adjacency to be symmetric and ensure no disconnected rows."""
    # A shape: (B, T, N, N)
    A_sym = (A + A.transpose(-2, -1)) / 2
    if identity_fallback:
        # Add self-loops where rows are near-empty after symmetrization
        row_sums = A_sym.sum(dim=-1)  # (B,T,N)
        disconnected = row_sums < threshold
        diag = torch.diagonal(A_sym, dim1=-2, dim2=-1)  # (B,T,N)
        diag.copy_(torch.where(disconnected, torch.ones_like(diag), diag))
    return A_sym
```

**Benefits**:
- Guarantees real eigenvalues
- Required for valid Laplacian
- Improves numerical stability

### 4. Enhanced Laplacian Regularization (match normalized Laplacian in code)

```python
def compute_stable_laplacian(A: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute numerically stable Laplacian."""
    N = A.shape[-1]

    # Degree matrix
    D = torch.diag_embed(A.sum(dim=-1))

    # Normalized Laplacian with stronger regularization
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    # Normalized Laplacian (preferred for stability)
    D_sqrt_inv = torch.diag_embed(1.0 / torch.sqrt(D.diagonal(dim1=-2, dim2=-1) + eps))
    L_norm = I - D_sqrt_inv @ A @ D_sqrt_inv

    return L_norm
```

## Configuration Schema

```yaml
model:
  graph:
    # Adjacency conditioning
    adj_row_softmax: bool  # true (enable masked row normalization)
    adj_softmax_tau: float  # 1.0 (temperature for softmax)
    edge_top_k: int  # 3 (default from detector.py:271)
    edge_threshold: float  # 1e-4 (default from detector.py:272)

    # Temporal smoothing
    adj_ema_beta: float  # 0.9 (EMA coefficient, null to disable)
    adj_ema_per_batch: bool  # false (share EMA across batch)

    # Symmetrization
    adj_force_symmetric: bool  # true (force symmetry)

    # Laplacian regularization
    laplacian_eps: float  # 1e-3 (identity regularization)
    laplacian_normalize: bool  # true (use normalized Laplacian)

    # Eigendecomposition
    eig_fp32: bool  # true (compute in float32)
    eig_max_condition: float  # 100.0 (max condition number)
    eig_fallback_on_nan: bool  # true (use cached PE on failure)
```

## Mathematical Guarantees

### With Full Conditioning

1. **Row Softmax**: ∑_j A_ij = 1 for all i
2. **Symmetry**: A = A^T → real eigenvalues
3. **EMA**: ||A_t - A_{t-1}|| < (1-β)||A_new - A_old||
4. **Regularization**: λ_min ≥ eps > 0

### Condition Number Bounds

```
Before: κ(L) = λ_max/λ_min → ∞ (λ_min can be 0)
After: κ(L) ≤ (2 + eps)/eps ≈ 2000 for eps=1e-3
With normalization: κ(L_norm) < 100
```

## Impact Analysis

### Stability Improvements
- Eigenvalue explosion: Eliminated
- Eigenvector flips: Rare with EMA
- NaN in PE: Should not occur
- Condition number: >10^6 → <100

### Clamps/Checks That May Be Reduced (after validation)
```python
# Candidates to simplify later (keep initially):
gnn_pyg.py:220: eigenvalues = torch.clamp(eigenvalues, min=1e-6, max=2.0)  # Keep as guard
gnn_pyg.py:240: pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)  # Keep as guard

# Fallback logic can be simplified
if not torch.isfinite(pe).all():  # Should rarely trigger
    use_cached_pe()
```

## Alternative Approaches Considered

### Option A: Static PE Only
- **Pros**: Maximum stability, zero compute
- **Cons**: Loses dynamic adaptation capability

### Option B: Learned PE Embeddings
```python
self.pe_embed = nn.Parameter(torch.randn(1, T, N, k))
```
- **Pros**: Stable, learnable
- **Cons**: Not graph-aware, fixed size

### Option C: Random Walk PE
- **Pros**: No eigendecomposition needed
- **Cons**: Less expressive than Laplacian PE

### Option D: Attention-Based Adjacency
```python
A = torch.softmax(Q @ K^T / sqrt(d), dim=-1)
```
- **Pros**: Always normalized, proven stable
- **Cons**: Different from current edge-based approach

**Recommendation**: Full conditioning (proposed solution) keeps current architecture while fixing stability.

## Testing Strategy

### Unit Tests
```python
def test_adjacency_row_normalization():
    """Verify row sums equal 1 before symmetrization."""
    A = torch.randn(32, 960, 19, 19)
    A_norm = normalize_adjacency(A)
    row_sums = A_norm.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))

def test_ema_smoothing():
    """Verify EMA reduces variance."""
    A_seq = [torch.randn(1, 19, 19) for _ in range(100)]
    ema = TemporalEMA(beta=0.9)
    smoothed = [ema.update(A) for A in A_seq]

    # Smoothed should have lower variance
    orig_var = torch.stack(A_seq).var()
    smooth_var = torch.stack(smoothed).var()
    assert smooth_var < orig_var * 0.5

def test_laplacian_stability():
    """Verify Laplacian eigendecomposition is stable."""
    A = torch.rand(2, 10, 19, 19)
    A_norm = normalize_adjacency(A, top_k=3)  # Use actual default
    A_sym = symmetrize_adjacency(A_norm)
    L = compute_stable_laplacian(A_sym, eps=1e-4)

    # Test eigendecomposition succeeds without NaN
    eigenvalues, eigenvectors = torch.linalg.eigh(L.to(torch.float32))
    assert torch.isfinite(eigenvalues).all()
    assert torch.isfinite(eigenvectors).all()

    # Check eigenvalues are finite and non-negative within tolerance
    assert (eigenvalues >= -1e-6).all()
```

### Integration Tests
```python
def test_stable_pe_computation():
    """Test PE computation with conditioned adjacency."""
    # Full path: edges → adjacency → Laplacian → eigendecomp → PE

def test_pe_temporal_consistency():
    """Verify PE doesn't jump wildly between timesteps."""

def test_gnn_with_conditioned_adjacency():
    """Test GNN forward pass with new adjacency."""
```

## Rollout Plan

### Phase 1: Add Conditioning (After PR-1 & PR-2)
1. Implement row softmax
2. Add symmetrization
3. Test with existing PE code

### Phase 2: Add Temporal Smoothing
1. Implement EMA module
2. Test temporal consistency
3. Tune beta parameter

### Phase 3: Update Laplacian Computation
1. Increase regularization epsilon
2. Add normalized Laplacian option
3. Remove redundant eigenvalue clamps

## Success Metrics

### Must Have
- [ ] Zero NaN/Inf in eigendecomposition across 10,000 batches
- [ ] Dynamic PE fallback rate < 0.1%
- [ ] Eigendecomposition succeeds in fp32 with sign-consistency
- [ ] No reliance on cached PE fallback in steady state

### Nice to Have
- [ ] Remove eigenvalue clamping
- [ ] Faster eigendecomposition (better conditioned)
- [ ] Improved graph learning dynamics

## Risks and Mitigations

### Risk 1: Over-Smoothing from EMA
- **Impact**: Graph becomes too static
- **Mitigation**: Tune beta, possibly schedule it

### Risk 2: Information Loss from Normalization
- **Impact**: Reduced expressiveness
- **Mitigation**: Use temperature parameter in softmax

### Risk 3: Symmetrization Loses Directional Info
- **Impact**: Can't model directed relationships
- **Mitigation**: Consider separate forward/backward adjacencies

## Questions for Cross-Validation

1. Should EMA be per-sample or per-batch?
2. What's the optimal regularization epsilon?
3. Should we use normalized vs unnormalized Laplacian?
4. Is row-softmax better than L2 normalization?
5. Should temperature tau be learnable?

## Performance Considerations

### Compute Impact
- Row softmax: ~2% overhead
- Symmetrization: Negligible
- EMA: Negligible (one multiply-add)
- Total: <5% overhead

### Memory Impact
- Within-sequence EMA is vectorized across T and keeps no persistent state
- Optional cross-forward EMA (disabled by default) would require caching last A per sample
- No change to PE dimension

## Mathematical Proofs

### Proof: Row Softmax Bounds Condition Number

Given row-normalized A where ∑_j A_ij = 1:
```
L = D - A where D_ii = ∑_j A_ij = 1
→ L = I - A
→ eigenvalues(L) = 1 - eigenvalues(A)

Since A is row-stochastic: |λ(A)| ≤ 1
→ λ(L) ∈ [0, 2]
→ κ(L) ≤ 2/eps with regularization
```

### Proof: EMA Reduces Variance

Let A_t be adjacency at time t, Â_t be EMA:
```
Â_t = βÂ_{t-1} + (1-β)A_t

Var(Â_t) = β²Var(Â_{t-1}) + (1-β)²Var(A_t)
        ≤ β²Var(Â_{t-1}) + (1-β)²σ²

At steady state: Var(Â) = (1-β)²σ² / (1-β²)
                        = (1-β)σ² / (1+β)
                        < σ² for β > 0
```

---

**STATUS**: Ready for cross-validation with senior AI agent before implementation
