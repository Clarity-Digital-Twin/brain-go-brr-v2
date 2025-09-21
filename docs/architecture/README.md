Architecture Docs (Index)

Use these to understand and validate the end-to-end model and pipeline:

- CANONICAL_ARCHITECTURE_SPEC.md — Source of truth for the full network and dataflow
- ASCII_PIPELINE_PLAN.md — Readable pipeline diagram and stages
- ARCHITECTURE_COMPARISON.md — Alternatives and rationale
- FINAL_STACK_ANALYSIS.md — Final chosen stack and trade-offs
- MAMBA_KERNEL_DECISION.md — CUDA kernel vs fallback trade-offs

Related operational notes
- Sampling strategy (manifest + balanced): see components/caching_and_sampling.md and TUSZ/CACHE_AND_SAMPLING.md
- Mamba CUDA fallback: set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d path on CPU/CI

Tip: Keep the canonical spec aligned with code changes; update the ASCII plan when stages or interfaces change.
