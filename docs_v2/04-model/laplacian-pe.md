# Laplacian Positional Encoding (LPE)

Goal
- Provide node positional features via graph Laplacian eigenvectors; support dynamic updates over time for V3.

Dynamic LPE (vectorized)
- Compute per‑timestep normalized Laplacian L and take the k smallest eigenvectors.
- Implementation uses a vectorized path across all timesteps; eigendecomposition runs with AMP disabled for stability.
- **v3.3.1 CRITICAL**: Eigenvectors are **DETACHED** after computation (gnn_pyg.py:205) to prevent gradient explosion through eigendecomposition.

Gradient Flow (v3.3.1+)
- ✅ **Adjacency learns**: Gradients flow through GNN output → adjacency matrix
- ✅ **Eigenvectors detached**: NO gradients through unstable eigendecomposition backward pass
- ✅ **Positional encodings are FIXED coordinates**: Like Transformer sinusoidal PE (2025 best practice)
- ✅ **Learning happens in GNN layers**: That PROCESS the PE, not in PE itself

Why Detachment Is Required
- PyTorch's `torch.linalg.eigh()` backward uses `∂L/∂A ∝ 1/(λᵢ - λⱼ)` for i ≠ j
- PR-3 adjacency conditioning (row-softmax + EMA + symmetry) creates **near-degenerate eigenvalues**
- Near-zero denominators → **gradient explosion** (observed: 7.03 spikes on Modal A100)
- **Fix**: Detach eigenvectors after computation → stable gradients (<1.0 P95)

Numerical stability
- Disable autocast for eigendecomposition; compute in float32 (or float64 if needed).
- Clamp degrees and add small diagonal regularization upstream to avoid singularities.
- Apply `nan_to_num` and cached‑PE fallback if needed (see `gnn_pyg.py`).
- Sign consistency: make each eigenvector's sum non‑negative (or align to previous timestep if using temporal alignment).
- **v3.3.1**: Eigenvectors detached to prevent gradient explosion (see ARCHITECTURAL_STABILITY_INVESTIGATION.md).

Semi‑dynamic mode
- `semi_dynamic_interval: N` computes PE every N timesteps and repeats in between.
- Greatly reduces memory while preserving accuracy (interval 5 works well on RTX 4090).

Config knobs
```yaml
model:
  graph:
    use_dynamic_pe: true
    k_eigenvectors: 16
    semi_dynamic_interval: 1   # 1 = fully dynamic; 5–10 for memory relief
```

Code anchors
- Vectorized dynamic PE: `src/brain_brr/models/gnn_pyg.py`
- V3 architecture: `docs/04-model/v3-architecture.md`
