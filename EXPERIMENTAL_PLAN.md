# 🧪 Experimental Plan - FLA Hyperparameter Study

**Version**: 1.3
**Date**: 2025-12-20
**Status**: ✅ Exp4 Complete (benchmark documented)
**Target**: Optimize FLA (Gated DeltaNet) for clinical EEG seizure detection

---

## 📌 Current Status (v4.2.0)

**Current Best (Held-Out TUSZ Eval)**:
- ✅ **FLA Exp4 (Cyclic LR / SGDR)**: **35.9% sensitivity @ 10 FA/24h** (AUROC 0.8654)
- SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json` (checkpoint: `results/local_fla_exp4_cyclic/checkpoints/best.pt`)
- Training: 78 epochs (early stopped), best epoch 63

**Completed**:
- ✅ **Exp1 (Regularization)**: Negative result (-2.7% vs baseline), model NOT overfitting
- ⏸️ **BiMamba2 Stack**: PAUSED at epoch 6 (Modal A100, $1.1k spent) - focusing on local training due to cost

**Key Findings**:
- SGDR restarts produced a new best dev checkpoint at epoch 63 (dev sensitivity@10FA = 0.2904)
- Held-out TUSZ eval beat our SeizureTransformer baseline at tuned operating points (+2.0% @ 10 FA, +4.1% @ 2.5 FA)

---

## 🎯 Exp4 Summary: Cyclic LR (SGDR)

**Config**: `configs/local/train_fla_exp4_cyclic.yaml`
**Status**: ✅ Complete
**Hypothesis**: Cyclic LR restarts help escape local minima and improve sensitivity@FA targets

**Key Config Changes**:
```yaml
scheduler:
  type: cosine_restarts        # SGDR with warm restarts
  t_initial: 10                # First cycle: 10 epochs
  t_mult: 2                    # Double each cycle: 10→20→40
  eta_min: 1e-6                # Min LR before restart to 1e-4

early_stopping:
  patience: 15                 # Faster verdict (vs baseline 20)
```

**Artifacts**:
- Checkpoint: `results/local_fla_exp4_cyclic/checkpoints/best.pt`
- Eval SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- Eval log: `results/local_fla_exp4_cyclic/eval_v2.log`

---

## 📊 Experimental History

### Baseline (Control)
**Best**: 0.284 @ epoch 9 (Oct 16)
**Post-resume plateau (snapshot)**: 0.257 @ epochs 17-29 (13-epoch plateau)
**Status**: Historical baseline run notes (superseded by Exp4)

### Exp1: Stronger Regularization
**Result**: ❌ NEGATIVE (-2.7%)
**Best**: 0.272 @ epoch 6
**Conclusion**: Model NOT overfitting

### Exp2: Lower LR (5e-5)
**Status**: ⏸️ Available but not prioritized (Exp4 more promising)

### Exp3: Smaller Model
**Status**: ❌ REJECTED (capacity reduction not justified)

### Exp4: Cyclic LR (SGDR)
**Status**: ✅ COMPLETE
**Result**: TUSZ eval 35.9% @ 10 FA/24h (AUROC 0.8654)

---

## 📏 Success Metrics

**Primary**: Sensitivity @ 10 FA/24h (held-out TUSZ eval, OVERLAP)
- ✅ **Achieved**: 35.9% (Exp4)
- **Next target**: Move toward Temple SOTA (≈50% @ 4 FA/24h)

**Secondary**: AUROC, PR-AUC, calibration (ECE), and stability

---

## 🔬 Key Hypotheses

1. ✅ **DISPROVEN**: Model overfitting (Exp1 showed regularization hurts)
2. ✅ **SUPPORTED**: Cyclic LR restarts improve dev sensitivity@10FA vs baseline plateau
3. ⏸️ **Alternative**: LR too high throughout (Exp2 available if Exp4 fails)

---

## 📖 References

**Configs**:
- Baseline: `configs/local/train_fla.yaml`
- Exp4 (ready): `configs/local/train_fla_exp4_cyclic.yaml`

**Trackers**:
- Baseline metrics: `BASELINE_METRICS.md`
- Exp1 analysis: `docs/archive/METRICS_EXP1_REGULARIZATION.md`

**WandB**:
- Baseline run: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90
- Current (partial): https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/c7eb044ceee34392ad1c793e783f4bc4

---

**Next Action**: Add a 4 FA/24h operating point + run official NEDC scoring for Exp4 outputs

**Last Updated**: 2025-12-20
