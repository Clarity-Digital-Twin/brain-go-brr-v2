# Archived: Non‑Finite Logits at Batch 28 (Dynamic PE)

This document has been integrated into the canonical docs:

- See `docs/08-operations/troubleshooting.md#dynamic-pe-nans` for current guidance.
- Stability details: `docs/04-model/gnn.md` (Stability safeguards)

Context

- Original issue: NaNs from `torch.linalg.eigh` during dynamic Laplacian PE when the learned adjacency becomes ill‑conditioned.
- Fix is implemented: degree clamping, diagonal regularization, NaN/Inf checks, cached PE fallback, and final `nan_to_num`.

Refer to the links above for the current, maintained guidance.
