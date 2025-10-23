# 📊 Training Metrics Tracker - Exp1 Regularization

**Experiment**: Stronger Regularization (dropout 0.1→0.2, weight_decay 0.01→0.05)
**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Run Start**: 2025-10-18 14:52 EDT
**Status**: ✅ Running (Epoch 8 in progress)

---

## 🎯 Performance Tracking

### Primary Metric: Sensitivity @ 10 FA/24h
**Target**: ≥0.28 (baseline) | **Stretch**: ≥0.30

| Epoch | Sens@10FA | Global Step | Status | Completed |
|-------|-----------|-------------|--------|-----------|
| 1 | 0.2045 | 7,702 | | Oct 19, 01:00 |
| 2 | 0.2530 | 15,404 | ↗️ +0.0485 | Oct 19, 11:08 |
| 3 | 0.2708 | 23,106 | ↗️ +0.0178 | Oct 19, 21:21 |
| 4 | 0.2708 | 30,808 | → Plateau | Oct 20, 07:28 |
| 5 | 0.2026 | 38,510 | ⚠️ -0.0682 | Oct 21, 03:57 |
| **6** | **0.2726** | **46,212** | **⭐ BEST** | **Oct 22, 03:18** |
| 7 | 0.2726 | 53,914 | → Plateau | Oct 22, 18:55 |
| 8 | _running_ | ~61,616 | 🔄 1% done | Oct 22, ~09:00 (est) |

---

## 📈 Performance Analysis

### Progression Summary
- **Epochs 1-3**: Steady improvement (0.2045 → 0.2708, +32%)
- **Epoch 4**: Plateau (same as epoch 3)
- **Epoch 5**: Sharp drop (-25% to 0.2026) ⚠️
- **Epoch 6**: Recovery to peak (0.2726) ⭐
- **Epoch 7**: Maintained peak (no improvement, patience 1/5)

### Current Status
- **Best**: Epoch 6 @ 0.2726 sensitivity
- **Patience**: 1/5 (epoch 7 didn't improve)
- **Early Stop ETA**: Will trigger at epoch 11 if no improvement
- **Next Milestone**: Epoch 9 validation (~Oct 23, 05:00 EST)

---

## 🎓 Comparison to Baseline

### Baseline FLA (train_fla.yaml)
- **Best**: Epoch 9 @ 0.284 sensitivity (stopped at epoch 13)
- **Final**: 0.284 @ 10 FA/24h

### Exp1 Regularization (this run)
- **Best so far**: Epoch 6 @ 0.2726 sensitivity
- **Delta**: -0.0114 (-4% vs baseline) ❌

### Hypothesis
Stronger regularization (dropout 0.2, weight_decay 0.05) appears to be **slightly hurting** performance compared to baseline (dropout 0.1, weight_decay 0.01).

---

## 🔮 Predictions

### Scenario 1: No Further Improvement (Most Likely)
- Epochs 8-11: Plateau at 0.2726
- Early stop triggers at epoch 11 (patience=5)
- **Final result**: 0.2726 @ 10 FA/24h (-4% vs baseline)
- **Conclusion**: Regularization too strong

### Scenario 2: Late Improvement (Possible)
- Epochs 8-9: One more improvement to ~0.28
- Continue to epoch 13-14
- **Final result**: ~0.28 @ 10 FA/24h (matches baseline)
- **Conclusion**: Regularization neutral

### Scenario 3: Continued Degradation (Unlikely)
- Epochs 8+: Drop below 0.27
- Early stop at lower performance
- **Conclusion**: Regularization harmful

---

## 📊 Training Dynamics

### Epoch Durations
- **Training**: ~4 hours (batch processing)
- **Validation**: ~5.5 hours (full dataset eval)
- **Total/Epoch**: ~9.5 hours
- **100 Epochs**: ~40 days (if no early stop)

### Resource Usage
- **GPU**: RTX 4090 (0.28GB alloc / 17.26GB total)
- **RAM**: 3-6GB used / 27GB available
- **Batch Size**: 8
- **Speed**: ~2.1-2.3s/batch

---

## 🔬 Technical Notes

### Regularization Changes (vs Baseline)
```yaml
model:
  tcn:
    dropout: 0.20        # ✨ +0.05 from baseline (0.15)
  mamba:
    dropout: 0.2         # ✨ +0.1 from baseline (0.1)
  graph:
    dropout: 0.2         # ✨ +0.1 from baseline (0.1)

training:
  weight_decay: 0.05     # ✨ +0.04 from baseline (0.01)
```

### Early Stopping Config
```yaml
early_stopping:
  patience: 5           # Same as baseline (for fair comparison)
  min_epochs: 0         # Same as baseline
  metric: sensitivity_at_10fa
```

### Gradient Health (Epoch 8, batch 52)
```
P50: 0.52 | IQR: 0.96 | P95: 4.43 | Max: 5.65
```
✅ Healthy - no gradient explosions

---

## 🎯 Next Steps

### If Early Stops at Epoch 11-12:
1. ✅ Accept result (~0.2726, -4% vs baseline)
2. ✅ Conclude stronger regularization hurts performance
3. ✅ **Recommendation**: Use baseline config for future runs
4. ✅ Resume baseline training (patience=20, target 100 epochs)

### If Matches Baseline (~0.28):
1. ✅ Regularization neutral
2. ⚠️ Consider: is training time worth the neutrality?
3. ✅ Still prefer baseline (faster to similar result)

### If Beats Baseline (>0.284):
1. 🎉 Regularization helps!
2. ✅ Update configs to use stronger regularization
3. ✅ Test on other experiments

---

## 📝 Data Sources

**Checkpoints**: `results/local_fla_exp1_reg/checkpoints/epoch_*.pt`
- Contains: best_metric (sensitivity@10FA), global_step, model state
- **NOT contains**: Full validation metrics (val_loss, AUROC, etc.)

**Wandb**: ❌ Data not syncing (502 error since 12:35 EDT Oct 22)
- Local: 5.8MB `.wandb` file (epochs 7-8 buffered)
- Cloud: Only epochs 0-3 from old run

**Training Logs**: Limited (tmux buffer)
- Can monitor live via: `tmux attach -t exp1-reg-resume`

---

## 🔔 Alerts

### Current
- ⚠️ **Wandb offline** - use this file + checkpoints for monitoring
- ⏱️ **Epoch 8 in progress** - validation ETA ~09:00 Oct 23

### Upcoming
- 📅 **Epoch 9 validation**: ~Oct 23, 05:00 EST (will update metrics)
- 🛑 **Early stop likely**: Epoch 11 (if no improvement from epoch 6 best)

---

**Last Updated**: 2025-10-22 19:34 EDT (Epoch 8, batch 52)
**Next Update**: After epoch 8 validation completes
**Auto-refresh**: Manual (check after each epoch)

---

*Generated from checkpoint data. Wandb unavailable due to 502 server error.*
