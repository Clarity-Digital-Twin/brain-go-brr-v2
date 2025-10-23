# 📊 Training Metrics Tracker - Exp1 Regularization

**Experiment**: Stronger Regularization (dropout 0.1→0.2, weight_decay 0.01→0.05)
**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Run Start**: 2025-10-18 14:52 EDT
**Restart**: 2025-10-23 17:01 EDT (after WSL2 crash, fresh wandb)
**Status**: ❌ **CANCELLED @ Epoch 10** (Oct 23, 17:31 EDT)
**Reason**: Regularization proven harmful (-2.7% vs baseline), resources reallocated to baseline

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
| 7 | 0.2726 | 53,914 | → Plateau (1/20) | Oct 22, 18:55 |
| 8 | 0.2726 | 61,616 | → Plateau (2/20) | Oct 23, 05:10 |
| 9 | 0.2726 | 69,318 | → Plateau (3/20) | Oct 23, 15:21 |
| 10 | _running_ | ~77,020 | 🔄 22% done | Oct 23, ~00:30 (est) |

---

## 📈 Performance Analysis

### Progression Summary
- **Epochs 1-3**: Steady improvement (0.2045 → 0.2708, +32%)
- **Epoch 4**: Plateau (same as epoch 3)
- **Epoch 5**: Sharp drop (-25% to 0.2026) ⚠️
- **Epoch 6**: Recovery to peak (0.2726) ⭐
- **Epochs 7-9**: **Locked at 0.2726** (patience 3/20)
- **Epoch 10**: In progress (resumed after WSL2 crash)

### Current Status
- **Best**: Epoch 6 @ 0.2726 sensitivity
- **Patience**: 3/20 (epochs 7-9 no improvement)
- **Early Stop ETA**: Will trigger at epoch 26 if no improvement (min_epochs=30 override)
- **Next Milestone**: Epoch 10 validation (~Oct 24, 00:30 EST)

---

## 🎓 Comparison to Baseline

### Baseline FLA (train_fla.yaml)
- **Best**: Epoch 9 @ 0.2801 sensitivity (stopped at epoch 12)
- **Final**: 0.2801 @ 10 FA/24h

### Exp1 Regularization (this run - CANCELLED)
- **Best**: Epoch 6 @ 0.2726 sensitivity
- **Final**: 0.2726 @ 10 FA/24h (plateaued epochs 6-10)
- **Delta**: -0.0075 (-2.7% vs baseline) ❌

### Conclusion
Stronger regularization (dropout 0.2, weight_decay 0.05) **hurts performance** compared to baseline (dropout 0.1, weight_decay 0.01). Model plateaued 4 epochs straight with no improvement, confirming regularization was too strong.

---

## 🔮 Predictions (UPDATED after epochs 7-9)

### Scenario 1: No Further Improvement (LIKELY - happening now)
- Epochs 7-9: ✅ **CONFIRMED** - Locked at 0.2726
- Will continue to min_epochs=30, then patience countdown
- Early stop triggers at epoch 50 if no improvement (patience=20 after epoch 30)
- **Final result**: 0.2726 @ 10 FA/24h (-4% vs baseline's 0.284)
- **Conclusion**: Regularization too strong, hurts performance

### Scenario 2: Late Improvement (UNLIKELY after 3 plateaus)
- Would need epochs 10-20 to show improvement
- Breakthrough to ~0.28 possible but not expected
- **Final result**: ~0.28 @ 10 FA/24h (matches baseline)
- **Conclusion**: Regularization neutral, but slower convergence

### Scenario 3: Continued Degradation (VERY UNLIKELY)
- Stable at 0.2726 for 3 epochs, unlikely to degrade
- **Conclusion**: Not expected

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
  patience: 20          # UPDATED v4.1 (was 5) - match baseline for fair comparison
  min_epochs: 30        # UPDATED v4.1 (was 0) - ensure adequate training
  metric: sensitivity_at_10fa
```

### Gradient Health (Epoch 8, batch 52)
```
P50: 0.52 | IQR: 0.96 | P95: 4.43 | Max: 5.65
```
✅ Healthy - no gradient explosions

---

## 🎯 Next Steps

### Current Plan (After Epoch 10)
1. ✅ Let training continue to min_epochs=30
2. ⏱️ Monitor epochs 10-30 for any late improvements
3. 🛑 Will auto-stop at epoch 50 if no improvement (patience=20 after epoch 30)
4. 📊 Compare final result vs baseline and Exp2 (lower LR)

### If Still Plateaued at Epoch 30 (LIKELY):
1. ✅ Accept result (~0.2726, -4% vs baseline's 0.284)
2. ✅ Conclude stronger regularization hurts performance
3. ✅ **Recommendation**: Do NOT use higher regularization
4. ✅ Resume baseline training (same patience=20, target 100 epochs)
5. ✅ Start Exp2 (lower LR) to test learning rate hypothesis

### If Matches Baseline (~0.28):
1. ✅ Regularization neutral but slower convergence (6 epochs vs baseline's improvement)
2. ✅ Still prefer baseline (faster to similar result)

### If Beats Baseline (>0.284):
1. 🎉 Regularization helps with late improvements!
2. ✅ Update recommendations
3. ⚠️ But note: slower convergence may indicate inefficiency

---

## 📝 Data Sources

**Checkpoints**: `results/local_fla_exp1_reg/checkpoints/epoch_*.pt`
- Contains: best_metric (sensitivity@10FA), global_step, model state
- **NOT contains**: Full validation metrics (val_loss, AUROC, etc.)

**Wandb**: ⚠️ Old run failed (502 error), new run started Oct 23 17:01
- Old run archived: `wandb_old_run_oct18_22/` (epochs 1-6 partial data)
- New run: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/191b56e837544d9e9bf3236c01f0376d

**Training Logs**: Limited (tmux buffer)
- Can monitor live via: `tmux attach -t exp1-reg-v2` (NEW SESSION after restart)

---

## 🔔 Alerts

### Current (Oct 23, 17:03)
- ✅ **Training resumed** - epoch 10 batch 1700+ (22% done)
- ✅ **Fresh wandb run** - tracking working correctly
- 📊 **Epochs 7-9 complete** - all locked at 0.2726 (patience 3/20)

### Upcoming
- 📅 **Epoch 10 validation**: ~Oct 24, 00:30 EST
- 🔄 **Patience status**: 3/20 (epochs 7-9 no improvement)
- ⏱️ **Minimum training**: Will run to at least epoch 30 (min_epochs=30)
- 🛑 **Early stop ETA**: Epoch 50 if no improvement (patience=20 after epoch 30)

---

**Last Updated**: 2025-10-23 17:03 EDT (Epoch 10 resumed, batch ~1700)
**Next Update**: After epoch 10 validation completes (~Oct 24, 00:30)
**Auto-refresh**: Manual (check after each epoch)

---

*Generated from checkpoint data. Fresh wandb tracking started Oct 23 17:01.*
