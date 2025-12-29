# FLA Quick Reference - 2025-12-20

**TL;DR**: FLA Exp4 is complete and benchmarked on held-out TUSZ eval; BiMamba2 remains paused (Modal cost).

---

## Current Status (v4.3.0)

| Component | Status | Evidence |
|-----------|--------|----------|
| **BiMamba2 Baseline** | ⏸️ Paused at epoch 6 (Modal A100) | Cost control |
| **FLA Exp4 (BiGatedDeltaNet)** | ✅ COMPLETE | SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json` |
| **Held-out TUSZ eval** | **35.9% @ 10 FA/24h** (AUROC 0.8654) | `results/local_fla_exp4_cyclic/eval_v2.log` |

---

## Key Artifacts

- Config: `configs/local/train_fla_exp4_cyclic.yaml`
- Checkpoint: `results/local_fla_exp4_cyclic/checkpoints/best.pt`
- Eval SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- Training log: `results/local_fla_exp4_cyclic/training_resumed.log`

---

## Next Actions

1. Add a 4 FA/24h operating point for Exp4 evaluation (current export: 10/5/2.5/1)
2. Run official NEDC v6.0.0 scoring for Exp4 outputs (OVERLAP/TAES/SzCORE) for publication-ready reporting

---

See also:
- `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md`
- `docs/06-evaluation/TAES_DISAMBIGUATION.md`
