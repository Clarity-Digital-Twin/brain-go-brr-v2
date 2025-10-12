# Versioning and Roadmap

Current

- v4.0.0 (current): Dual SSM stacks (BiMamba2 baseline + Flash Linear Attention variant) with deterministic resume, ext4 cache requirement on WSL2, Modal + local training validated.
- v3.x (archived baseline): learned adjacency (Edge Mamba) + vectorized GNN + Dynamic LPE (BiMamba2-only).

Historical

- v2.6 (archived): heuristic adjacency (top‑k cosine)

Planned

- Modal A/B analysis: compare BiMamba2 vs FLA sensitivity/FA curves, publish delta.
- Optional post-v4.0.0 ideas: Hybrid SWA experiment (`docs/flash-linear-attention/FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md`), gradient sanitisation filter, logging polish.

Historical notes

- PR planning and implementation summaries: `docs/10-final-refactor/`
