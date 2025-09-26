# Incident: Non‑Finite Logits (Dynamic PE Eigendecomposition)

Summary

- Symptom: Training aborted with “Non‑finite logits detected” (batch ~28).
- Root cause: Unstable eigendecomposition of the normalized Laplacian during dynamic PE when adjacency became ill‑conditioned.
- Fix: Add regularization, NaN/Inf detection, cached PE fallback, and final sanitization.

Technical details

- Operation: `torch.linalg.eigh(L)` where `L = I − D^{-1/2} A D^{-1/2}` on `(B×T)` graphs.
- Failure modes: near‑zero degrees, rank‑deficient adjacencies, repeated eigenvalues.

Implemented safeguards

- Clamp degrees: `degrees.clamp_min(1e-6)`.
- Regularize Laplacian: `L = L + 1e-4 * I` in float32 with AMP disabled (increase to `1e-3` when ill‑conditioned).
- Guard eigendecomp: try/except with NaN/Inf checks; use cached PE or small random PE as fallback.
- Final sanity: `torch.nan_to_num` on PE; cache last valid PE for reuse.

Status

- Integrated into `src/brain_brr/models/gnn_pyg.py` (vectorized dynamic PE path).
- Verified with synthetic stress tests (zero/near‑zero and rank‑1 adjacencies) and full V3 forward: no NaNs.

Operator guidance

- Dynamic PE remains enabled by default in configs.
- Occasional warning logs indicate fallback engaged; training continues safely.
- For debugging, set `BGB_NAN_DEBUG=1` to surface non‑finite inputs early.

Links

- Component: `docs/04-model/gnn.md` (stability safeguards)
- Config: `docs/03-configuration/config-schema.md` (graph PE flags)
