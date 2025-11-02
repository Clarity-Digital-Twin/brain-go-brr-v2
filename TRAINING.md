# Active Training Plan

**Created**: October 20, 2025
**Status**: 🟢 ACTIVE - Baseline Running (Plateaued), Exp4 Ready
**Last Updated**: November 1, 2025

---

## 📊 Current Situation

### Baseline FLA (🔄 RUNNING - PLATEAUED)
- **Status**: Running at epoch 30+, plateaued at 0.257 for 13 epochs (17-29)
- **Best checkpoint**: Epoch 9 → **0.284 sensitivity @ 10 FA/24h**, 0.95 TAES
- **Current performance**: 0.257 (stuck in local minimum)
- **Patience**: 13/20 (early stop triggers at 20)
- **Expected stop**: Epoch 36 (~Nov 4-5)
- **Location**: `results/local_fla_training/checkpoints/`
- **Config**: `configs/local/train_fla.yaml` (patience=20, min_epochs=30)

### Exp1 - Stronger Regularization (✅ COMPLETED - FAILED)
- **Status**: ✅ Completed, early stopped at epoch 6
- **Best checkpoint**: Epoch 2 → **0.272 sensitivity @ 10 FA/24h**
- **Gap vs baseline**: -2.7% (WORSE than baseline 0.284)
- **Hypothesis verdict**: ❌ Stronger regularization HURTS performance
- **Conclusion**: Model NOT overfitting (underfitting rare seizure class instead)
- **Location**: `results/local_fla_exp1_reg/`

### Exp4 - Cyclic LR (SGDR) (🚀 READY TO LAUNCH)
- **Status**: ⏳ Waiting for baseline early stop (~epoch 36)
- **Config**: `configs/local/train_fla_exp4_cyclic.yaml`
- **Hypothesis**: Stuck in local minimum at 0.257, SGDR restarts will escape
- **Strategy**: Cyclic LR (1e-4 ↔ 1e-6) with warm restarts every 10→20→40 epochs
- **Expected**: Break plateau, potentially recover 0.28+ region
- **Launch**: After baseline stops (~Nov 4-5)

### Exp2 & Exp3 (❌ PAUSED/REJECTED)
- **Exp2 (Lower LR)**: ⏸️ Available but not prioritized (Exp4 more promising)
- **Exp3 (Smaller Model)**: ❌ Rejected (capacity reduction not justified)

---

## 🎯 Sequential Execution Plan

### Phase 1: Wait for Baseline Early Stop (IN PROGRESS)

**Timeline**: NOW → ~Nov 4-5 (epoch 30+ → epoch 36)

**Status**: ✅ **Baseline running, patience 13/20**

**What's happening:**
- Baseline resumed Oct 16 with patience=20, min_epochs=30
- Reached epoch 30+, plateaued at 0.257 for 13 epochs (17-29)
- Best still epoch 9 (0.284)
- Early stop will trigger at epoch 36 when patience exhausted

**When baseline stops:**
- Review final metrics (confirm no improvement after epoch 9)
- Confirm hypothesis: Stuck in local minimum at 0.257
- Proceed to Phase 2 (launch Exp4)

---

### Phase 2: Launch Exp4 (SGDR) (AFTER BASELINE STOPS)

**Timeline**: Nov 4-5 → Dec 14-18 (~40 days, 100 epochs)

**Objective**: Test local minimum escape via cyclic LR restarts

**Hypothesis**:
- Stuck in local minimum at 0.257 (lost 0.284 peak after resume chaos)
- SGDR LR spikes will help escape and rediscover better solutions
- May recover 0.28+ region or find new optimum

**Config** (`configs/local/train_fla_exp4_cyclic.yaml`):
```yaml
training:
  learning_rate: 1.0e-4            # Peak LR
  scheduler:
    type: cosine_restarts          # SGDR with warm restarts
    t_initial: 10                  # First cycle: 10 epochs
    t_mult: 2                      # Double each cycle: 10→20→40
    eta_min: 1.0e-6                # Min LR before restart to 1e-4
  early_stopping:
    patience: 15                   # Faster verdict than baseline (20)
    min_epochs: 30                 # Don't stop before epoch 30
```

**How SGDR works:**
```
Cycle 1 (epochs 3-13):   LR 1e-4 → 1e-6, then restart to 1e-4
Cycle 2 (epochs 13-33):  LR 1e-4 → 1e-6, then restart to 1e-4
Cycle 3 (epochs 33-73):  LR 1e-4 → 1e-6, then restart to 1e-4
Goal: LR spikes help escape 0.257 plateau, rediscover 0.28+ region
```

**Command to run:**
```bash
# After baseline stops at epoch 36:
tmux new -s exp4
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp4_cyclic.yaml
# Detach: Ctrl+B D
```

**What to monitor:**
- **Critical window**: Epochs 1-15 (should hit 0.28+ if hypothesis correct)
- Baseline hit 0.284 @ epoch 9, so Exp4 should show similar trajectory if working
- Watch for LR restarts: should see metric spikes when LR resets to 1e-4
- Compare to baseline plateau (0.257) - any improvement validates hypothesis

---

### Phase 3: Decision Tree After Exp4 (Dec 14-18)

**Three Possible Outcomes:**

#### Outcome A: No Improvement (50% Expected)
```
Best checkpoint: Still 0.257-0.27 range (no escape from plateau)
Conclusion: Local minimum hypothesis wrong, architecture limit reached
```

**Action:**
- ✅ Accept 0.284 @ epoch 9 as best for FLA architecture
- Compare to Temple SOTA (4 FA/24h @ 50% sensitivity)
- Decide: BiMamba2 comparison OR deploy as-is
- Archive remaining experiments (Exp2/Exp3 unlikely to help)

#### Outcome B: Moderate Improvement (30% Chance)
```
Best checkpoint: 0.28-0.30 range (recovered baseline or slightly better)
Conclusion: SGDR helped escape plateau, found similar/better optimum
```

**Action:**
- ✅ New baseline: 0.28-0.30 @ 10 FA/24h
- Analyze what worked: LR restarts effective for local minimum escape
- Consider applying SGDR to other experiments
- May warrant longer training with SGDR (extend to 150 epochs)

#### Outcome C: Major Improvement (20% Chance)
```
Best checkpoint: >0.32 (significant jump, new optimum found)
Conclusion: SGDR discovered better solution space, major win
```

**Action:**
- 🎉 Major research finding! SGDR enabled substantial improvement
- New baseline: >0.32 @ 10 FA/24h
- Compare to Temple SOTA and SeizureTransformer
- Publish findings: "SGDR enables local minimum escape in seizure detection"
- Consider BiMamba2 with SGDR for comparison

---

## 📋 Checklist

### Before Baseline Stops (~Nov 4-5)

- [x] Exp1 completed and documented
- [x] Baseline running with patience=20, min_epochs=30
- [x] Exp4 config validated (`train_fla_exp4_cyclic.yaml`)
- [x] SGDR scheduler tested and validated
- [ ] Monitor baseline for early stop (~epoch 36)
- [ ] Confirm disk space for Exp4 checkpoints (~20GB needed for 100 epochs)

### When Baseline Stops

- [ ] Verify baseline stopped cleanly at epoch 36
- [ ] Document final baseline metrics in STATUS.md
- [ ] Confirm best checkpoint is still epoch 9 (0.284)
- [ ] Archive baseline run (save logs, checkpoints)
- [ ] Update BASELINE_METRICS.md with final summary

### Before Launching Exp4

- [ ] Verify `train_fla_exp4_cyclic.yaml` validated
- [ ] Check isolated output directory: `results/local_fla_exp4_cyclic/`
- [ ] Start fresh tmux session: `tmux new -s exp4`
- [ ] Export BGB_NAN_DEBUG=1
- [ ] Launch: `.venv/bin/python -m src train configs/local/train_fla_exp4_cyclic.yaml`
- [ ] Monitor first epoch for crashes
- [ ] Detach safely: Ctrl+B D

### During Exp4 Training

- [ ] **Critical**: Monitor epochs 1-15 closely (should hit 0.28+ if working)
- [ ] Check progress every 2-3 days
- [ ] Watch for LR restart spikes (should see metric jumps)
- [ ] Note if escaping 0.257 plateau
- [ ] Monitor GPU memory/temperature
- [ ] Keep tmux session alive

### After Exp4 Completes

- [ ] Identify final best epoch
- [ ] Compare to baseline best (epoch 9, 0.284)
- [ ] Analyze SGDR effectiveness (did restarts help?)
- [ ] Update STATUS.md with results
- [ ] Make decision per "Decision Tree" above
- [ ] Update EXPERIMENTAL_PLAN.md

---

## 🔬 Scientific Rationale

### Why Exp4 (SGDR) Over Other Approaches

**Problem:** Baseline plateaued at 0.257 for 13 epochs (17-29)
- Best: 0.284 @ epoch 9 (lost after resume chaos at epoch 13)
- Current: 0.257 (stuck in local minimum)
- Hypothesis: Resume broke optimizer momentum, trapped in suboptimal basin

**Why SGDR (Cyclic LR Restarts)?**

1. **Proven technique** (Loshchilov & Hutter, 2017):
   - LR spikes help escape local minima
   - Periodic restarts explore new solution spaces
   - Used successfully in ImageNet, NLP, medical imaging

2. **Better than alternatives**:
   - **Exp2 (Lower LR)**: Would make plateau WORSE (too conservative)
   - **Exp3 (Smaller model)**: Might reduce capacity below 0.284 peak
   - **SGDR**: Actively tries to ESCAPE the plateau via LR spikes

3. **Addresses root cause**:
   - Resume chaos likely broke optimizer momentum
   - SGDR gives fresh starts every 10/20/40 epochs
   - Multiple chances to rediscover 0.28+ region

**What we're testing:**
- H0 (null): 0.257 plateau is architecture ceiling, SGDR won't help
- H1 (alternative): Stuck in local minimum, SGDR will escape to ≥0.28

**Why this is good science:**
- Isolated variable: Only changes LR schedule (architecture/data same)
- Testable hypothesis: Should hit 0.28+ by epoch 15 if working
- Falsifiable: If no improvement, accept 0.284 as ceiling
- Fresh start: Eliminates resume chaos confound

---

## 📊 Expected Timeline

```
Oct 20:           Exp1 completed (FAILED, 0.272 vs 0.284 baseline)
Oct 16-Nov 1:     Baseline resumed, epochs 13-30+, plateaued at 0.257
Nov 1 (TODAY):    Baseline @ epoch 30+, patience 13/20
Nov 4-5:          Baseline early stop @ epoch 36 (patience exhausted)
                  → Launch Exp4 immediately

Nov 5:            Exp4 epoch 1 (monitor closely!)
Nov 6-7:          Exp4 epochs 2-4 (watch for early signal)
Nov 10:           Exp4 epoch 9 (baseline peaked here)
Nov 15:           Exp4 epoch 15 (CRITICAL: should hit 0.28+ if hypothesis correct)
Nov 20:           Exp4 epoch 20 (end of first full restart cycle)
Nov 30:           Exp4 epoch 30 (min_epochs met)
Dec 14-18:        Exp4 completion (epoch 100 or early stop)

                  IF no improvement by epoch 15:
                    → Hypothesis likely wrong, let it run to completion for data

                  IF improvement by epoch 15:
                    → Hypothesis validated, expect continued improvement
                    → May exceed 0.284 baseline
```

**Total time investment:**
- Baseline completion: ~3 days (Nov 1-5)
- Exp4 training: ~40 days (Nov 5 - Dec 15)
- **Total: ~43 days** to definitive answer on SGDR hypothesis

---

## 🎯 Success Criteria

### Minimum Success (What We Need)
- ✅ Exp1 completed cleanly (confirmed regularization doesn't help)
- ✅ Baseline running to patience exhaustion (confirming plateau)
- ⏳ Baseline stops at epoch 36 (patience 20/20)
- ⏳ Exp4 launches successfully from epoch 1
- ⏳ Exp4 runs to min_epochs=30 without crashes
- ⏳ Clear data on whether SGDR escapes local minimum

### Stretch Success (What We Hope For)
- 🎯 Exp4 reaches 0.28+ by epoch 15 (validates hypothesis)
- 🎯 Exp4 recovers baseline 0.284 peak
- 🎯 Clear evidence SGDR restarts help escape plateau
- 🎯 New baseline >0.30 @ 10 FA/24h

### Dream Success (20% Chance)
- 🌟 Exp4 reaches >0.32 @ 10 FA/24h
- 🌟 SGDR discovers substantially better solution space
- 🌟 Publishable finding: "SGDR enables local minimum escape in seizure detection"
- 🌟 Approaches Temple SOTA territory (4 FA/24h @ 50%)

---

## 📝 Notes & Lessons

### Why Exp1 Failed
- **Hypothesis**: Model overfitting, needs stronger regularization
- **Reality**: Stronger regularization (dropout 0.1→0.2, weight_decay 0.01→0.05) HURT performance (-2.7%)
- **Lesson**: Model was NOT overfitting in traditional sense
- **Insight**: Likely underfitting rare seizure class, needs more capacity/exploration, not less

### Why SGDR (Exp4) Over Exp2/Exp3
- **Exp2 (Lower LR)**: Current LR already decayed to 8.3e-5, going lower would worsen plateau
- **Exp3 (Smaller model)**: Capacity reduction risks losing ability to reach 0.284 peak
- **Exp4 (SGDR)**: Actively tries to ESCAPE plateau via LR restarts, addresses root cause (local minimum)

### Why Starting Fresh (Not Resuming)
- **Resume chaos**: Baseline crashed at epoch 13, lost 0.284 peak, never recovered
- **Fresh start**: Eliminates optimizer state corruption, clean test of SGDR
- **Scientific validity**: Isolated experiment (only variable is SGDR scheduler)
- **Multiple chances**: SGDR gives 3+ restart cycles to rediscover good solutions

### Key Insights - Medical ML
- **Rare class dynamics**: Seizures (8% of data) need special handling
- **Local minima**: Easy to get trapped, especially after resume/interruption
- **Validation ≠ Clinical**: Val loss rising doesn't mean clinical metrics declining
- **Patience required**: Model needs time to explore, but also tools to escape traps
- **SGDR promise**: Periodic fresh starts may help rediscover rare pattern solutions

---

## 🔗 Related Documentation

- `STATUS.md` - Overall project status and history
- `configs/README.md` - Config rationale and two-tier patience strategy
- `REALISTIC_PERFORMANCE_TARGETS.md` - What "good" looks like (Temple SOTA: 4 FA/24h @ 50%)
- `docs/05-training/training-methodology.md` - General training approach
- `docs/05-training/HYPERPARAMETER_EXPERIMENTS.md` - Experiment design philosophy

---

**Last updated**: November 1, 2025
**Next update**: When baseline stops (~Nov 4-5)
**Status**: Wait for baseline early stop @ epoch 36, then launch Exp4 (SGDR) 🚀
