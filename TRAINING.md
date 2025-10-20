# Active Training Plan

**Created**: October 20, 2025
**Status**: 🟢 ACTIVE - Exp1 Running, Baseline Queued
**Last Updated**: October 20, 2025

---

## 📊 Current Situation

### Baseline FLA (COMPLETE - Waiting for Resume Decision)
- **Status**: Stopped at epoch 13 (benign crash, not resumed)
- **Best checkpoint**: Epoch 9 → **0.284 sensitivity @ 10 FA/24h**, 0.95 TAES
- **Location**: `results/local_fla_training/checkpoints/best.pt`
- **Early stopping**: patience=5 (would have triggered at epoch 14 anyway)
- **Val loss trajectory**: Rising from 0.027 → 0.053 (overfitting signal)
- **Decision**: Pragmatic stop, looked like overfitting

### Exp1 - Stronger Regularization (ACTIVE)
- **Status**: 🔄 Running at epoch 5
- **Best checkpoint**: Epoch 2 → **0.271 sensitivity @ 10 FA/24h**
- **Gap vs baseline**: -4.6% (WORSE than baseline)
- **Early stopping**: patience=5, counter=3 (3 epochs without improvement)
- **Expected stop**: Epoch 7 (2 more epochs, ~20 hours)
- **Hypothesis verdict**: ❌ Stronger regularization HURTS performance
- **Location**: `results/local_fla_exp1_reg/`

### Exp2 & Exp3 (QUEUED)
- **Status**: ⏸️ PAUSED - Will decide after baseline resume completes
- **Rationale**: If exp1 failed, similar hyperparameter tweaks likely won't help
- **Better strategy**: Resume baseline first, establish true ceiling, then design new experiments

---

## 🎯 Sequential Execution Plan

### Phase 1: Complete Exp1 Naturally (IN PROGRESS)

**Timeline**: NOW → ~2 days (epoch 5 → epoch 7)

**Action**: ✅ **Let it run, don't touch it**

**Why:**
- Test that early stopping logic works correctly
- Clean completion: "Exp1 tested fully, patience=5 worked as designed"
- Confirms hypothesis: Regularization hurts (0.271 vs 0.284 baseline)

**Expected outcome:**
```
Epoch 6: No improvement → counter = 4
Epoch 7: No improvement → counter = 5 (STOP TRIGGER)
```

**When exp1 stops:**
- Review final metrics
- Document lessons learned
- Proceed to Phase 2

---

### Phase 2: Resume Baseline with patience=20 (AFTER EXP1)

**Timeline**: Day 2 → Day 10-12 (~8-10 days, 17-20 more epochs)

**Objective**: Test "second peak" hypothesis
- Medical imaging literature: Best clinical performance often comes 10-20 epochs AFTER minimum val_loss
- Question: Does sensitivity improve at epochs 25-40 even as val_loss rises?

**Config** (already updated in `configs/local/train_fla.yaml`):
```yaml
training:
  epochs: 100  # Keep as-is
  early_stopping:
    patience: 20        # Was 5 (now 4x more tolerant)
    min_epochs: 30      # Don't allow early stop before epoch 30
    metric: sensitivity_at_10fa
```

**What this does:**
```
Resume from epoch 13
├─ Epochs 13-29: Can't early stop (min_epochs=30 prevents it)
├─ Epoch 30+: Early stopping enabled
└─ Will stop when 20 epochs pass without improvement

Expected scenarios:
  70% chance: No improvement, stops at epoch 30-31
  20% chance: Small improvement (0.29-0.31), stops at epoch 40-50
  10% chance: Significant improvement (>0.32), major win
```

**Command to run:**
```bash
# After exp1 stops, run this:
tmux new -s baseline-resume
cd /home/jj/proj/brain-go-brr-v2
export BGB_NAN_DEBUG=1
python -m src train configs/local/train_fla.yaml \
  --resume results/local_fla_training/checkpoints/last.pt
# Detach: Ctrl+B D
```

**What to monitor:**
- Watch for sensitivity @ 10 FA/24h
- Compare to baseline best (0.284 @ epoch 9)
- Check if val_loss stabilizes or continues rising
- Look for any "second peak" at epochs 25-40

---

### Phase 3: Decision Tree After Baseline Resume (Day 10-12)

**Three Possible Outcomes:**

#### Outcome A: No Improvement (70% Expected)
```
Best checkpoint: Still epoch 9 (0.284)
Early stop trigger: Epoch 30-31
Conclusion: Overfitting signal at epoch 13 was real
```

**Action:**
- ✅ Accept 0.284 @ 10 FA/24h as best for FLA architecture
- Compare to Temple SOTA (4 FA/24h @ 50% sensitivity)
- Decide: BiMamba2 comparison OR deploy as-is OR accept results
- Archive exp2/exp3 (unlikely to beat 0.284 if exp1 failed)

#### Outcome B: Small Improvement (20% Chance)
```
Best checkpoint: Epoch 25-35 (0.29-0.31)
Early stop trigger: Epoch 45-55
Conclusion: "Second peak" was real, baseline had more potential
```

**Action:**
- ✅ New baseline: 0.29-0.31 @ 10 FA/24h
- Design NEW experiments targeting this higher bar
- Update experiment configs to use patience=20
- Test different hypotheses (architecture changes, not just regularization)

#### Outcome C: Significant Improvement (10% Chance)
```
Best checkpoint: Epoch 30-45 (>0.32)
Early stop trigger: Epoch 50-65
Conclusion: Major win, "second peak" dramatic
```

**Action:**
- 🎉 Celebrate! This would be a major finding
- New baseline: >0.32 @ 10 FA/24h
- Compare to Temple SOTA and SeizureTransformer
- Potentially publishable result
- Rethink entire experiment strategy

---

## 📋 Checklist

### Before Resuming Baseline (After Exp1 Stops)

- [ ] Verify exp1 stopped cleanly (check logs)
- [ ] Document exp1 final metrics in STATUS.md
- [ ] Confirm `train_fla.yaml` has patience=20, min_epochs=30
- [ ] Check disk space for 20 more checkpoints (~4GB needed)
- [ ] Start tmux session for baseline resume
- [ ] Export BGB_NAN_DEBUG=1
- [ ] Run resume command with `--resume` flag
- [ ] Monitor first 2-3 epochs for crashes

### During Baseline Resume

- [ ] Check progress every 2-3 days
- [ ] Watch for any new best checkpoints
- [ ] Note if val_loss stabilizes or continues rising
- [ ] Monitor GPU memory/temperature
- [ ] Keep tmux session alive (Ctrl+B D to detach)

### After Baseline Completes

- [ ] Identify final best epoch
- [ ] Compare to original best (epoch 9, 0.284)
- [ ] Update STATUS.md with results
- [ ] Make decision per "Decision Tree" above
- [ ] Update configs/README.md if needed

---

## 🔬 Scientific Rationale

### Why We're Doing This

**Problem:** Baseline stopped at epoch 13 (13% complete) with:
- Val loss rising (0.027 → 0.053)
- Sensitivity declining after epoch 9
- Looked like classic overfitting

**But:** patience=5 was too aggressive for 100-epoch plan

**Literature evidence:**
- Medical imaging: Best clinical metrics often lag minimum val_loss by 10-20 epochs
- "Second peak" phenomenon: Model learns rare patterns (seizures) after general patterns (background)
- Standard practice: Don't allow early stopping before 30% complete

**What we're testing:**
- H0 (null): Overfitting signal was real, epoch 9 is the best
- H1 (alternative): "Second peak" exists at epochs 25-40

**Why this is good science:**
- Changes one variable (patience 5→20)
- Tests specific hypothesis ("second peak")
- Has clear success criteria (improvement beyond 0.284)
- Falsifiable (if no improvement, we know epoch 9 is best)

---

## 📊 Expected Timeline

```
Day 0 (Oct 20):   Exp1 @ epoch 5
Day 1:            Exp1 @ epoch 6
Day 2:            Exp1 @ epoch 7 → STOPS (patience=5)
                  Resume baseline from epoch 13
Day 3:            Baseline @ epoch 14-15
Day 5:            Baseline @ epoch 17-18
Day 7:            Baseline @ epoch 20-22
Day 10:           Baseline @ epoch 25-27
Day 12:           Baseline @ epoch 30-31

                  IF no improvement:
                    → Stops at epoch 30-31 (patience triggers)

                  IF improvement found:
                    → Continues to epoch 40-50
                    → Stops when 20 epochs pass without new improvement
```

**Total time investment:**
- Exp1 completion: 2 days
- Baseline resume: 8-10 days (pessimistic) or 15-20 days (optimistic)
- **Total: 10-22 days** to definitive answer

---

## 🎯 Success Criteria

### Minimum Success (What We Need)
- ✅ Exp1 completes cleanly (tests early stopping works)
- ✅ Baseline resumes successfully from checkpoint
- ✅ Baseline runs to min_epochs=30 without crashes
- ✅ Clear data on whether "second peak" exists

### Stretch Success (What We Hope For)
- 🎯 Baseline improves beyond 0.284
- 🎯 Clear evidence of "second peak" at epochs 25-40
- 🎯 New baseline >0.30 @ 10 FA/24h

### Dream Success (Unlikely But Possible)
- 🌟 Baseline reaches >0.35 @ 10 FA/24h
- 🌟 Approaches Temple SOTA territory (4 FA/24h @ 50%)
- 🌟 Publishable result from architecture alone

---

## 📝 Notes & Lessons

### Why Exp1 Failed
- Hypothesis: Insufficient regularization causing overfitting
- Reality: Stronger regularization (dropout 0.1→0.2, weight_decay 0.01→0.05) HURT performance
- Lesson: Model was NOT overfitting in traditional sense (underfitting rare seizure class)
- Better strategy: Increase model capacity or seizure sampling, not regularization

### Why Exp2/Exp3 Are Paused
- If exp1 (highest confidence hypothesis) failed badly, similar tweaks likely fail
- Better: Establish true baseline ceiling first
- Then: Design experiments against confirmed baseline

### Key Insight
Medical ML is different from computer vision:
- Rare class (seizures) needs MORE capacity, not less
- Validation loss rising ≠ clinical performance declining
- Early stopping on val_loss can be premature
- Need patience for model to learn rare patterns

---

## 🔗 Related Documentation

- `STATUS.md` - Overall project status and history
- `configs/README.md` - Config rationale and two-tier patience strategy
- `REALISTIC_PERFORMANCE_TARGETS.md` - What "good" looks like (Temple SOTA: 4 FA/24h @ 50%)
- `docs/05-training/training-methodology.md` - General training approach
- `docs/05-training/HYPERPARAMETER_EXPERIMENTS.md` - Experiment design philosophy

---

**Last updated**: October 20, 2025
**Next update**: When exp1 completes (~2 days)
**Status**: Let exp1 finish, then resume baseline with patience=20 🎯
