# Brain-Go-Brr Exp4 (FLA / Gated DeltaNet) - Final Results & SGDR Notes

**Last Updated**: 2025-12-20  
**Config**: `configs/local/train_fla_exp4_cyclic.yaml`  
**Checkpoint**: `results/local_fla_exp4_cyclic/checkpoints/best.pt`  
**Eval SSOT**: `results/local_fla_exp4_cyclic/eval_results_v2.json`

## Executive Summary

Experiment 4 (SGDR / cosine restarts) completed 78 epochs (early stopped) with best epoch 63, and produced a held-out TUSZ eval benchmark that beats our SeizureTransformer baseline at the tuned OVERLAP operating points.

### Verified Held-Out Results (TUSZ Eval)

| Metric | Value |
|--------|-------|
| **AUROC** | **0.8654** |
| **PR-AUC** | 0.5409 |
| **Sensitivity @ 10 FA/24h (primary)** | **35.9%** |
| **Sensitivity @ 5 FA/24h** | 27.1% |
| **Sensitivity @ 2.5 FA/24h** | 18.6% |
| **Sensitivity @ 1 FA/24h** | 5.8% |
| **ECE** | 0.029 |
| **Val Loss** | 0.090 |

### Head-to-Head vs SeizureTransformer (Same TUSZ Eval Split, OVERLAP Scoring)

| FA Rate | FLA Exp4 | SeizureTransformer | Delta |
|--------:|---------:|------------------:|------:|
| 10 FA/24h | **35.9%** | 33.90% | **+2.0%** |
| 2.5 FA/24h | **18.6%** | 14.50% | **+4.1%** |

SeizureTransformer numbers are from our run in `reference_repos/SeizureTransformer/docs/results/FINAL_COMPREHENSIVE_RESULTS_TABLE.md` (Python OVERLAP = NEDC OVERLAP).

---

## Dataset Accounting (Avoiding “865 vs 836” Confusion)

- TUSZ eval split contains 865 EDF/label pairs (confirmed in `results/local_fla_exp4_cyclic/eval_v2.log`).
- Under our 60s windowing, 29 recordings produce 0 windows and are excluded from scoring.
- Metrics in `results/local_fla_exp4_cyclic/eval_results_v2.json` are computed on 836 scored recordings totaling 127.8 hours.

---

## Training Details (Exp4)

- Total epochs: 78 (early stopped)
- Best epoch: 63
- Best dev sensitivity@10FA during training: 0.2904 (29.0%)
- Training log: `results/local_fla_exp4_cyclic/training_resumed.log`
- Eval log: `results/local_fla_exp4_cyclic/eval_v2.log`

---

## Scoring Methodology (What These Numbers Mean)

- Reported sensitivity@FA targets uses OVERLAP-style event scoring (binary any-overlap TP counting), not NEDC TAES fractional scoring.
- See `docs/06-evaluation/TAES_DISAMBIGUATION.md` for the “TAES” naming collision and the exact overlap logic.

---

## SGDR Notes (High-Level)

- Exp4 uses cosine restarts with `t_initial=10`, `t_mult=2`, and `eta_min=1e-6`.
- Best dev checkpoint was reached late (epoch 63), consistent with longer training + restarts being useful in this setting.
- We still do not report a 4 FA/24h operating point from this eval export (current eval JSON includes 10/5/2.5/1).
