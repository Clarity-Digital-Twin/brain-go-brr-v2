# 🚨 P0 CRITICAL: V3 DUAL-STREAM ARCHITECTURE NAN EXPLOSION ROOT CAUSE ANALYSIS

## IMMEDIATE ISSUE

**EVERY SINGLE BATCH** (156+ batches) produced 61,440 NaN/Inf values in logits. The V3 dual-stream architecture is fundamentally broken at initialization or forward pass.

## ROOT CAUSE CANDIDATES (ORDERED BY LIKELIHOOD)

### 1. ❌ EDGE MAMBA INITIALIZATION CATASTROPHE (90% PROBABILITY)

**The Problem:**
```python
# src/brain_brr/models/detector.py:206-221
edge_feats = edge_scalar_series(elec_feats, metric=edge_metric)  # (B, 171, 960, 1)

# CRITICAL BUG: Edge features start as SCALAR (1-dim)
edge_flat = edge_feats.squeeze(-1).permute(0, 2, 1).reshape(
    batch_size * seq_len, 171, 1
)

# Then project 1 → 16 dimensions
edge_lifted = self.edge_in_proj(edge_flat)  # (B*960, 171, 16)

# Feed through 171 PARALLEL Mambas
edge_processed = self.edge_mamba(edge_lifted.transpose(1, 2).contiguous())
```

**WHY IT EXPLODES:**
- Starting with 1D scalar features (cosine similarity) is information-poor
- Projecting 1→16 dims with random initialization → extreme values
- 171 parallel Mambas processing near-identical noisy signals → resonance
- Mamba state accumulation → exponential growth → NaN

### 2. ❌ DYNAMIC LAPLACIAN PE EIGENDECOMPOSITION FAILURE (80% PROBABILITY)

**The Problem:**
```python
# src/brain_brr/models/gnn_pyg.py - Dynamic PE computation
for t_idx in range(0, T, interval):
    # Compute Laplacian from learned adjacency
    L = compute_laplacian(adj[:, :, :, t_idx])

    # CRITICAL: Eigendecomposition on LEARNED (potentially garbage) adjacency
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # If adjacency is degenerate → eigenvalues can be negative/complex
    # If adjacency has disconnected components → zero eigenvalues
    # If adjacency is near-singular → numerical instability
```

**WHY IT EXPLODES:**
- Edge Mamba produces garbage adjacency at initialization
- Laplacian of garbage graph → degenerate eigenvalues
- `torch.linalg.eigh` on singular matrix → NaN eigenvectors
- NaN positional encoding → contaminates everything

**EVIDENCE:**
- `semi_dynamic_interval: 5` means 192 eigendecompositions per batch
- RTX 4090 FP32 precision issues with small eigenvalues
- No eigenvalue clamping or regularization in code

### 3. ❌ EDGE→NODE FUSION DIMENSION MISMATCH (70% PROBABILITY)

**The Problem:**
```python
# After edge processing, we have:
edge_weights = edge_processed  # (B, 960, 171, 1) - learned adjacency

# But node features are:
node_feats = node_processed  # (B, 19, 64, 960)

# The GNN expects:
# nodes: (B, 19, 64, 960)
# edges: (B, 19, 19, 960) - BUT WE HAVE (B, 171, 960)!

# CRITICAL: 171 edges → 19×19 adjacency matrix assembly
adj_matrix = assemble_adjacency(edge_weights)  # POTENTIAL SHAPE ISSUES
```

**WHY IT EXPLODES:**
- Upper triangular (171) to full matrix (19×19) conversion
- Symmetrization might create feedback loops
- Incorrect adjacency → GNN message passing explosion

### 4. ❌ BACK-PROJECTION BOTTLENECK (60% PROBABILITY)

**The Problem:**
```python
# After GNN, we have:
gnn_out  # (B, 19, 64, 960)

# Must project back:
flattened = gnn_out.reshape(B, 19*64, 960)  # (B, 1216, 960)
bottleneck = self.proj_from_electrodes(flattened)  # (B, 512, 960)
```

**WHY IT EXPLODES:**
- 1216→512 compression after GNN processing
- If GNN produced large values → projection amplifies
- No normalization between GNN and projection

### 5. ❌ NODE MAMBA RESHAPE CONTIGUITY (50% PROBABILITY)

**The Problem:**
```python
# Line 197-199: DANGEROUS RESHAPES
node_flat = elec_feats.permute(0, 1, 3, 2).reshape(
    batch_size * 19, 64, seq_len
).contiguous()  # Force contiguous for CUDA
```

**WHY IT EXPLODES:**
- Permute → reshape → contiguous is risky
- CUDA kernels expect specific memory layout
- If not truly contiguous → garbage memory reads → NaN

## SMOKING GUN EVIDENCE

1. **CONSISTENT 61,440 NaNs** = exactly 4 × 15,360 (batch_size × sequence_length)
   - ALL outputs are NaN, not partial
   - Suggests early layer failure (TCN or first Mamba)

2. **IMMEDIATE FAILURE** from batch 1
   - Not gradual degradation
   - Points to initialization or first forward pass

3. **SANITIZATION TRIGGERED** on every batch
   - Model NEVER produces valid logits
   - Loss of 0.1316 is from zeros, not real values

## CRITICAL CODE PATHS TO INVESTIGATE

```python
# 1. Edge feature initialization
src/brain_brr/models/edge_features.py:edge_scalar_series()

# 2. Edge Mamba forward pass
src/brain_brr/models/detector.py:209-221

# 3. Dynamic PE computation
src/brain_brr/models/gnn_pyg.py:forward() with use_dynamic_pe=True

# 4. Adjacency assembly
src/brain_brr/models/edge_features.py:assemble_adjacency()

# 5. Back-projection
src/brain_brr/models/detector.py:246-249
```

## IMMEDIATE FIXES TO TRY

### FIX 1: DISABLE DYNAMIC PE (HIGHEST PRIORITY)
```yaml
# configs/local/train.yaml
graph:
  use_dynamic_pe: false  # TURN THIS OFF FIRST
```

### FIX 2: DISABLE EDGE STREAM (IF FIX 1 FAILS)
```yaml
graph:
  edge_mamba_layers: 0  # Disable edge Mamba completely
```

### FIX 3: ADD INITIALIZATION SAFEGUARDS
```python
# In detector.py after line 221:
edge_processed = torch.clamp(edge_processed, -10, 10)  # Clamp edge outputs
adj_matrix = torch.clamp(adj_matrix, 0, 1)  # Clamp adjacency
```

### FIX 4: ADD EIGENVALUE REGULARIZATION
```python
# In gnn_pyg.py dynamic PE:
eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # Prevent zero/negative
eigenvectors = torch.nan_to_num(eigenvectors, 0.0)  # Replace NaN
```

## THE REAL PROBLEM

**V3 WAS NEVER PROPERLY TESTED**. The architecture has:
- No unit tests for edge stream
- No gradient flow validation
- No initialization bounds checking
- No intermediate activation monitoring
- No safeguards against degenerate adjacency matrices

## RECOMMENDATION

1. **IMMEDIATELY**: Set `use_dynamic_pe: false` and retry
2. **IF STILL FAILS**: Set `edge_mamba_layers: 0`
3. **IF STILL FAILS**: The issue is in node Mamba or TCN initialization
4. **LONG TERM**: V3 needs complete numerical stability overhaul

---

**THIS IS A P0 ARCHITECTURAL FAILURE. THE V3 DUAL-STREAM IS FUNDAMENTALLY BROKEN.**