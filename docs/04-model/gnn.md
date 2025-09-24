# GraphChannelMixer (PyG)

File: `src/brain_brr/models/gnn_pyg.py`

- Vectorized over all timesteps (B×T graphs)
- Dynamic Laplacian PE by default (k=16), computed from the learned adjacency per timestep
- 2× SSGConv (α=0.05) with residual + LayerNorm + Dropout
- Bypass edge transform when edge weights already Softplus’ed upstream

Vectorized path (V3)

- Flattens `(B,19,T,D)` → `(B*T,19,D)` and `(B,T,19,19)` → `(B*T,19,19)`.
- Builds a disjoint batch for PyG; constructs `edge_index`/`edge_weight` from the adjacency.
- Concatenates Laplacian PE (k=16) to node features only on the first GNN layer.
- Applies SSGConv → LayerNorm → Dropout; residuals from layer 2 onward.
- Returns `(B,19,T,D)` reshaped back from the batched result.

Dynamic vs static PE

- Dynamic PE is configurable via `graph.use_dynamic_pe` (recommended true for V3).
- Dynamic PE implementation is vectorized across B×T with sign-consistency and optional semi-dynamic interval (`graph.semi_dynamic_interval`).
- Static PE remains available when `use_dynamic_pe: false` and is computed once from the structural 10–20 montage.

Bypass edge transform

- In V3, `bypass_edge_transform=True` because edge weights are already transformed by `Linear+Softplus` in the edge stream.

Laplacian PE details (dynamic)

- Normalized Laplacian: `L = I - D^{-1/2} A D^{-1/2}` computed for each graph where `A` is the learned adjacency.
- Vectorized eigendecomposition across `(B×T)` graphs using `torch.linalg.eigh` in float32 with AMP disabled for stability.
- Takes the k smallest eigenvectors (k=16) and enforces sign consistency per eigenvector (non‑negative sum heuristic).
- Optional `semi_dynamic_interval`: compute PE every N timesteps and repeat between updates (reduces compute further).
- Typical overhead is small (tens of MB, milliseconds per batch) given `N=19`.

Memory notes (RTX 4090)

- Full dynamic (interval=1) drives many eigendecompositions (960 per window). To reduce memory:
  - Set `semi_dynamic_interval: 5–10` to cut eigendecomps 5–10× with negligible accuracy impact.
  - Use a moderate batch size (e.g., 4 on 24GB VRAM).
  - A100‑80GB can run full dynamic with large batches; keep `mixed_precision: true` on A100.

Stability safeguards (implemented)

- Degree clamping before normalization prevents divide‑by‑zero.
- Diagonal regularization `L += εI` (ε=1e‑5) avoids singular Laplacians.
- NaN/Inf detection with graceful fallback:
  - Use last valid PE when available; else small random PE as a last resort.
  - Final `torch.nan_to_num` to ensure finite tensors.
- Cached PE buffer to reuse the last valid dynamic PE on rare failures.
- These guards eliminate non‑finite logits stemming from ill‑conditioned adjacencies.
