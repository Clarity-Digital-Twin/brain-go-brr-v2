# GraphChannelMixer (PyG)

**Last Updated**: October 1, 2025 (v3.4.1)
**File**: `src/brain_brr/models/gnn_pyg.py`

## Overview

- **Vectorized** over all timesteps (B×T graphs processed simultaneously)
- **Dynamic Laplacian PE**: ALWAYS ENABLED (k=16 eigenvectors), computed from learned adjacency per timestep
- **GNN architecture**: 2× SSGConv (α=0.05) with residual + LayerNorm + Dropout
- **Edge transform bypass**: True (edge weights already Softplus'ed upstream in V3)

## Vectorized Path (V3)

1. Flatten `(B,19,T,D)` → `(B*T,19,D)` and `(B,T,19,19)` → `(B*T,19,19)`
2. Build disjoint batch for PyG; construct `edge_index`/`edge_weight` from adjacency
3. Concatenate Laplacian PE (k=16) to node features on first GNN layer only
4. Apply SSGConv → LayerNorm → Dropout; residuals from layer 2 onward
5. Reshape back to `(B,19,T,D)` from batched result

## Dynamic Laplacian PE (v3.3.1+)

### Current Behavior (ALWAYS ENABLED)

- **Configuration**: `graph.use_dynamic_pe: true` (default, recommended)
- **Implementation**: Vectorized across B×T with sign-consistency
- **Update interval**: `graph.semi_dynamic_interval: 5` (optimal - update every 5 timesteps)
- **Gradient behavior**: **EIGENVECTORS DETACHED** (v3.3.1 critical fix, `gnn_pyg.py:205`)

### Critical Fix (v3.3.1 - September 30, 2025)

**Line**: `src/brain_brr/models/gnn_pyg.py:205`
```python
eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)

# CRITICAL: Detach eigenvectors to prevent gradient explosion
eigenvectors = eigenvectors.detach()
```

**Why This Matters**:
- PyTorch eigendecomposition backward: `∂L/∂A ∝ 1/(λᵢ - λⱼ)`
- Near-degenerate eigenvalues (from row-softmax + EMA + symmetry) → gradient explosion
- **Solution**: Eigenvectors are FIXED positional coordinates (2025 best practice)
- Learning happens in GNN layers that PROCESS PE, not in PE itself
- **Result**: Zero gradient explosions, architecture ROCK SOLID

### Static PE (Legacy - Not Recommended)

- Available when `use_dynamic_pe: false`
- Computed once from structural 10-20 montage
- Not recommended - dynamic PE with detached eigenvectors is stable and more effective

Adjacency conditioning (PR‑3)

- Optional row‑wise softmax normalization with temperature: `graph.adj_row_softmax`, `graph.adj_softmax_tau`.
- Optional temporal smoothing via EMA: `graph.adj_ema_beta`.
- Optional symmetry enforcement for Laplacian stability: `graph.adj_force_symmetric`.
- Laplacian regularization: `graph.laplacian_eps` (default `1e‑4`) and `graph.laplacian_normalize`.

Bypass edge transform

- In V3, `bypass_edge_transform=True` because edge weights are already transformed by `Linear+Softplus` in the edge stream.

## Laplacian PE Implementation Details

### Computation
1. **Normalized Laplacian**: `L = I - D^{-1/2} A D^{-1/2}` for each graph (A = learned adjacency)
2. **Vectorized eigendecomposition**: Across `(B×T)` graphs using `torch.linalg.eigh` in float32
3. **AMP handling**: Disabled around eigendecomp for numerical stability
4. **Eigenvector selection**: k=16 smallest eigenvectors
5. **Sign consistency**: Enforced per eigenvector (non-negative sum heuristic)
6. **Detachment (v3.3.1)**: `eigenvectors = eigenvectors.detach()` at line 205

### Semi-Dynamic Updates
- **Purpose**: Reduce computation by updating PE every N timesteps
- **Configuration**: `graph.semi_dynamic_interval: 5` (optimal)
- **Benefit**: 5× fewer eigendecompositions with negligible accuracy impact
- **Overhead**: Small (tens of MB, milliseconds per batch for N=19 electrodes)

Memory notes (RTX 4090)

- Full dynamic (interval=1) drives many eigendecompositions (960 per window). To reduce memory:
  - Set `semi_dynamic_interval: 5–10` to cut eigendecomps 5–10× with negligible accuracy impact.
  - Use a moderate batch size (e.g., 4 on 24GB VRAM).
  - A100‑80GB can run full dynamic with large batches; keep `mixed_precision: true` on A100.

## Quick Tips

- **Gradient policy (v3.3.1+)**: Eigenvectors detached for stability; adjacency still learns through GNN output
- **Memory optimization**: Increase `semi_dynamic_interval` (5-10) before disabling dynamic PE
- **Always use dynamic PE**: v3.3.1+ fixes make it stable on all hardware

Stability safeguards (implemented)

- Degree clamping before normalization prevents divide‑by‑zero.
- Diagonal regularization `L += εI` (ε=1e‑4; increase to 1e‑3 when ill‑conditioned) avoids singular Laplacians.
- NaN/Inf detection with graceful fallback:
  - Use last valid PE when available; else small random PE as a last resort.
  - Final `torch.nan_to_num` to ensure finite tensors.
- Cached PE buffer to reuse the last valid dynamic PE on rare failures.
- These guards eliminate non‑finite logits stemming from ill‑conditioned adjacencies.

Edge similarity margin

- Similarity is clamped at source with a configurable margin: `graph.edge_similarity_margin` (default 0.01), keeping values within `[-1+margin, 1-margin]` to prevent downstream explosions in the edge Mamba path.
