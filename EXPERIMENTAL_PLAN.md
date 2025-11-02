# 🧪 Experimental Plan - FLA Hyperparameter Study

**Version**: 1.2
**Date**: 2025-11-01 (Updated after Exp4 cyclic LR implementation)
**Status**: Active - Baseline Running, Exp4 Ready
**Target**: Optimize FLA (Gated DeltaNet) for clinical EEG seizure detection

---

## 📌 Current Status (v4.2.0)

**Active Training**:
- 🔄 **FLA Baseline**: Epoch 30+ running, plateaued at 0.257 for 13 epochs (17-29), patience 13/20, early stop expected epoch 36
- ✅ **Exp4 (Cyclic LR)**: Ready to launch after baseline completes - escape 0.257 local minimum via SGDR restarts

**Completed**:
- ✅ **Exp1 (Regularization)**: Negative result (-2.7% vs baseline), model NOT overfitting
- ⏸️ **BiMamba2 Stack**: PAUSED at epoch 6 (Modal A100, $1.1k spent) - focusing on local training due to cost

**Key Findings**:
- Best performance: 0.284 @ epoch 9 (lost after resume chaos at epoch 13)
- Current: 0.257 (plateau 13 epochs)
- Hypothesis: Stuck in local minimum, need LR restarts to escape

---

## 🎯 Next Experiment: Exp4 - Cyclic LR (SGDR)

**Config**: `configs/local/train_fla_exp4_cyclic.yaml`
**Status**: ✅ Validated, ready to launch
**Hypothesis**: Stuck in local minimum at 0.257, cyclic LR restarts will escape

**Key Changes**:
```yaml
scheduler:
  type: cosine_restarts        # SGDR with warm restarts
  t_initial: 10                # First cycle: 10 epochs
  t_mult: 2                    # Double each cycle: 10→20→40
  eta_min: 1e-6                # Min LR before restart to 1e-4

early_stopping:
  patience: 15                 # Faster verdict (vs baseline 20)
```

**How SGDR Works**:
- **Cycle 1** (epochs 3-13): LR 1e-4 → 1e-6, then restart
- **Cycle 2** (epochs 13-33): LR 1e-4 → 1e-6, then restart
- **Cycle 3** (epochs 33-73): LR 1e-4 → 1e-6, then restart
- **Goal**: LR spikes help escape 0.257 plateau, rediscover 0.28+ region

**Launch After Baseline**:
```bash
tmux new -s exp4
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp4_cyclic.yaml
```

---

## 📊 Experimental History

### Baseline (Control)
**Best**: 0.284 @ epoch 9 (Oct 16)
**Current**: 0.257 @ epochs 17-29 (13-epoch plateau)
**Status**: Running, will auto-stop ~epoch 36 (patience exhausted)

### Exp1: Stronger Regularization
**Result**: ❌ NEGATIVE (-2.7%)
**Best**: 0.272 @ epoch 6
**Conclusion**: Model NOT overfitting

### Exp2: Lower LR (5e-5)
**Status**: ⏸️ Available but not prioritized (Exp4 more promising)

### Exp3: Smaller Model
**Status**: ❌ REJECTED (capacity reduction not justified)

### Exp4: Cyclic LR (SGDR)
**Status**: ✅ Ready to launch after baseline completes
**Expected**: Break 0.257 plateau, potentially recover 0.28+

---

## 📏 Success Metrics

**Primary**: Sensitivity @ 10 FA/24h
- 🎯 **Target**: ≥0.28 (match original peak)
- 🚀 **Excellent**: ≥0.30 (+7% vs current plateau)

**Secondary**: AUROC, stability, training efficiency

---

## 🔬 Key Hypotheses

1. ✅ **DISPROVEN**: Model overfitting (Exp1 showed regularization hurts)
2. 🔄 **TESTING**: Stuck in local minimum (Exp4 will test via cyclic LR)
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

**Next Action**: Wait for baseline early stop (~epoch 36), then launch Exp4

**Last Updated**: 2025-11-01 21:30 EDT
