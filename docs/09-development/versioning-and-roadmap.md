# Versioning and Roadmap

Current

- v4.0.0 (current): Dual SSM stacks (BiMamba2 baseline + Flash Linear Attention variant) with deterministic resume, ext4 cache requirement on WSL2, Modal + local training validated.
- v3.x (archived baseline): learned adjacency (Edge Mamba) + vectorized GNN + Dynamic LPE (BiMamba2-only).

Historical

- v2.6 (archived): heuristic adjacency (top‑k cosine)

Planned

- Local FLA completion: Complete 100-epoch FLA training on RTX 4090 (Epoch 7/100 in progress, ~40 days total)
- Optional Modal restart: Resume BiMamba2 if budget permits (PAUSED at Epoch 6, $18,600 projected cost)
- A/B analysis: Compare BiMamba2 vs FLA sensitivity/FA curves if both stacks complete
- Optional post-v4.0.0 ideas: Hybrid SWA experiment (`docs/flash-linear-attention/FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md`), gradient sanitisation filter, logging polish.

Historical notes

- PR planning and implementation summaries: `docs/10-final-refactor/`
