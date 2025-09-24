Docs Migration Notes (Sept 2025)

We consolidated one‑off documents at the root of docs/ into the canonical structure.
Active docs reflect the current codebase (TCN → Bi‑Mamba → PyG GNN + LPE). The Edge temporal stream and learned adjacency are implemented in V3; this archive captures pre‑V3 planning/history.

Moved to archive:
- ARCHITECTURE_EVOLUTION.md — history and strategy; superseded by current‑state and canonical‑spec.
- EXPERIMENT_TRACKING.md — stale branches/tags; see operations/training.md and W&B dashboard for live tracking.
- GNN_CRITICAL_FIX_PLAN.md — consolidated into 04‑research/future/v2_6_dynamic_gnn_lpe_plan.md.
- v2_6_dynamic_gnn_lpe_CORRECTED.md — merged into the v2.6 plan; kept here for reference.
- MODAL_DEPLOYMENT_GUIDE.md — content integrated into modal/deploy.md.
- MODAL_PERFORMANCE_FIX.md — content integrated into modal/performance_optimization.md and troubleshooting.md.
- HISTORY.md — outdated structure map; superseded by docs/README and archive/README.
