# 🧪 Experimental Plan - FLA Hyperparameter Study

**Version**: 1.0
**Date**: 2025-10-22
**Status**: Active
**Target**: Optimize FLA (Gated DeltaNet) architecture for clinical EEG seizure detection

---

## 🎯 Research Question

**Can we improve upon baseline FLA performance (0.28 sensitivity @ 10 FA/24h) through systematic hyperparameter optimization?**

Sub-questions:
1. Was baseline training stopped too early? (stopped epoch 13, best epoch 9)
2. Is the model overfitting? (would stronger regularization help?)
3. Is the learning rate too high? (causing late-training instability?)
4. How do these interventions affect long-term convergence (epochs 20-30+)?

---

## ⚖️ Fair Comparison Principles

### Critical Requirements for Valid Comparison:

1. **Same Early Stopping Criteria** ✅
   - `patience: 20` (NOT 5!)
   - `min_epochs: 30` (ensure adequate training)
   - `metric: sensitivity_at_10fa`

2. **Same Evaluation Protocol** ✅
   - NEDC OVERLAP scoring
   - FA rates: [10, 5, 2.5, 1] per 24h
   - Dev set: 1,832 files (natural 8% seizure distribution)

3. **Same Infrastructure** ✅
   - RTX 4090 local training
   - Batch size: 8
   - Mixed precision: OFF (stability)
   - Gradient clip: 0.5

4. **Same Random Seed** ✅
   - `seed: 42` (all experiments)

**Why This Matters**:
- Baseline ran with `patience=5` (stopped at epoch 13)
- Exp1 currently has `patience=5` (will stop ~epoch 11)
- **NOT FAIR** - experiments need same stopping criteria!

---

## 📊 Experimental Design

### Baseline (Control)

**Config**: `configs/local/train_fla.yaml`
**Status**: ⚠️ **NEEDS RESUME** (stopped too early)
**History**: Epochs 8-13 (best: epoch 9 @ 0.2801)

**Parameters**:
```yaml
model:
  mamba:
    n_layers: 6
    d_model: 512
    dropout: 0.1
  graph:
    dropout: 0.1
  tcn:
    dropout: 0.15

training:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  batch_size: 8
  early_stopping:
    patience: 20        # ✨ UPDATED (was 5)
    min_epochs: 30      # ✨ UPDATED (was 0)
```

**Why Resume?**:
- Stopped at epoch 13 with OLD config (patience=5)
- Best was epoch 9 (0.2801) - only 4 epochs of patience used
- NEW config has patience=20, min_epochs=30
- Could improve at epochs 15, 20, 25, 30+
- **Need to know TRUE baseline performance before comparing experiments**

**Action Required**:
```bash
# Resume baseline to epoch 30+ with patience=20
tmux new -s baseline-resume
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla.yaml --resume
```

---

### Experiment 1: Stronger Regularization

**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Status**: ✅ Running (Epoch 8, best: epoch 6 @ 0.2726)
**Hypothesis**: Model overfitting → stronger regularization helps long-term

**Changes vs Baseline**:
```yaml
model:
  mamba.dropout: 0.1 → 0.2       # +100%
  graph.dropout: 0.1 → 0.2       # +100%
  tcn.dropout: 0.15 → 0.20       # +33%

training:
  weight_decay: 0.01 → 0.05      # +400%
```

**Config Status**: ✅ **FIXED** (patience: 5→20, min_epochs: 0→30)

**Restart Required**: ⚠️ **YES** (config changed mid-run)

**Action Required**:
```bash
# Wait for epoch 8 to complete (validation at 91%)
# Then restart from epoch 8 checkpoint with new config
tmux kill-session -t exp1-reg-resume
tmux new -s exp1-reg-resume
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml --resume
```

**Wandb Fix Opportunity**: ✅ Can attempt to fix 502 error on restart

**Expected Outcome**:
- **IF overfitting**: Exp1 beats baseline at epochs 15-30
- **IF underfitting**: Exp1 plateaus below baseline
- **Current**: -4% vs baseline (0.2726 vs 0.2801) at epoch 6/9

---

### Experiment 2: Lower Learning Rate

**Config**: `configs/local/train_fla_exp2_lr.yaml`
**Status**: ⏸️ **NOT STARTED**
**Hypothesis**: LR too high → causing late-training instability / oscillation

**Changes vs Baseline**:
```yaml
training:
  learning_rate: 1.0e-4 → 5.0e-5     # -50%
  scheduler:
    warmup_ratio: 0.03 → 0.05         # Longer warmup
```

**Why This Hypothesis**:
- Baseline best at epoch 9, then plateaued (10-13)
- Could be overshooting optimum with LR=1e-4
- Lower LR might enable continued improvement

**Expected Outcome**:
- **Slower** initial convergence (epochs 1-10)
- **More stable** late training (epochs 15-30)
- **Potentially higher** final performance if LR was issue

**Action Required**:
```bash
# Ensure patience=20, min_epochs=30 in config, then start
tmux new -s exp2-lr
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp2_lr.yaml
```

---

### Experiment 3: ~~Smaller Model~~ → **REJECTED**

**Config**: `configs/local/train_fla_exp3_smaller.yaml`
**Status**: ❌ **DO NOT RUN** (flawed hypothesis)

**Original Hypothesis**: "Model too large for dataset size (4,667 files)"

**Why REJECTED**:

1. **Dataset is NOT small**:
   - 4,667 training files
   - 60s windows, 10s stride → ~27 windows/file
   - Total: ~126,000 training windows
   - 31M params / 126k samples = 246 params/sample (REASONABLE!)

2. **Precedent from literature**:
   - Vision transformers: 86M params on ImageNet (1.2M images)
   - BERT: 110M params on BookCorpus+Wiki (3.3B words)
   - Our ratio is CONSERVATIVE

3. **Other issues more likely**:
   - Regularization (Exp1)
   - Learning rate (Exp2)
   - Architecture choices (future work)

4. **Downsizing = throwing away capacity**:
   - 31M→17M params = 45% reduction
   - Seizure detection needs capacity (complex spatiotemporal patterns)
   - No evidence current model is too large

**Alternative**: If model capacity is concern, test AFTER baseline/exp1/exp2 results. Not as primary experiment.

---

## 📅 Execution Timeline

### Phase 1: Resume & Start (Oct 22-23)

| Run | Action | Start | ETA |
|-----|--------|-------|-----|
| **Baseline** | Resume from epoch 13 | Oct 22, 20:00 | Nov 8 (~17 days) |
| **Exp1** | Update config (patience=20), continue | Running | Nov 5 (~14 days) |
| **Exp2** | Start fresh | Oct 23, 09:00 | Nov 12 (~20 days) |

### Phase 2: Monitor & Early Stopping (ongoing)

- **Check epochs 15, 20, 25, 30**: Record best metrics
- **Early stopping**: Automatic at patience=20 from each run's best epoch
- **Expected stop range**: Epochs 25-50 (depending on when peaks occur)

### Phase 3: Analysis & Decision (Nov 12+)

- Compare final best metrics
- Analyze convergence curves
- Decide on optimal hyperparameters
- Plan follow-up experiments (if needed)

---

## 📏 Success Metrics

### Primary Metric
**Sensitivity @ 10 FA/24h** (higher is better)

Target hierarchy:
- ✅ **Good**: ≥0.28 (match baseline)
- 🎯 **Better**: ≥0.30 (+7% improvement)
- 🚀 **Excellent**: ≥0.32 (+14% improvement)

### Secondary Metrics
- **Sensitivity @ 5 FA/24h**: Stricter false alarm tolerance
- **AUROC**: Overall discrimination ability
- **Training stability**: Gradient norms, loss curves

---

## 🔬 Hypotheses & Predictions

### Hypothesis 1: Baseline Stopped Too Early
**Test**: Resume baseline to epoch 30+

**Predictions**:
- **Optimistic**: Improves to 0.29-0.30 (another "second peak")
- **Realistic**: Plateaus at 0.28 (epoch 9 was true optimum)
- **Pessimistic**: Degrades (overfitting visible)

**Impact**: If baseline improves, raises bar for experiments

---

### Hypothesis 2: Model is Overfitting
**Test**: Exp1 (stronger regularization) vs Baseline

**Predictions**:
- **IF overfitting**: Exp1 > Baseline at epochs 20-30
- **IF NOT overfitting**: Exp1 < Baseline (regularization hurts)

**Current Data** (epochs 6-9):
- Exp1: 0.2726 (epoch 6)
- Baseline: 0.2801 (epoch 9)
- Exp1 is -4% → suggests NO overfitting OR regularization too strong

**Key Question**: Does Exp1 catch up / overtake at epochs 15-25?

---

### Hypothesis 3: Learning Rate Too High
**Test**: Exp2 (lower LR) vs Baseline

**Predictions**:
- **IF LR too high**: Exp2 more stable, higher final performance
- **IF LR optimal**: Exp2 slower convergence, similar final performance

**What to look for**:
- Baseline: Oscillations in validation metrics?
- Exp2: Smoother curves, continued improvement past epoch 15?

---

## 📊 Data Collection & Tracking

### Per-Epoch Metrics (Save to checkpoints)
```python
{
  'epoch': int,
  'best_metric': float,  # sensitivity_at_10fa
  'global_step': int,
  'val_metrics': {
    'val_loss': float,
    'auroc': float,
    'taes': float,
    'sensitivity_at_10.0fa': float,
    'sensitivity_at_5.0fa': float,
    'sensitivity_at_2.5fa': float,
    'sensitivity_at_1.0fa': float,
  }
}
```

### Tracking Files
- **Baseline**: `TRAINING_METRICS_BASELINE.md` (to be created)
- **Exp1**: `TRAINING_METRICS_EXP1_REG.md` (✅ exists)
- **Exp2**: `TRAINING_METRICS_EXP2_LR.md` (to be created)

### Update Frequency
- After each epoch validation completes
- Use checkpoint extraction script (see Exp1 tracker for template)

---

## ⚠️ Known Issues & Mitigations

### Issue 1: Wandb Upload Failures
**Problem**: 502 errors on file_stream endpoint (since 12:35 EDT Oct 22)
**Impact**: No live web UI monitoring for epochs 7-8
**Root Cause**: Wandb backend hit server error, stopped uploading (but local logging continues)

**Immediate Mitigation**: ✅ Use local checkpoint-based tracking (TRAINING_METRICS_*.md files)

**Potential Fix on Exp1 Restart**:
When restarting Exp1 with new config, wandb will re-initialize. Options:

1. **Fresh Start** (safest):
   - Delete old wandb run directory
   - Start with new run ID
   - ❌ Lose connection to epochs 1-6 history (already lost due to 502)
   - ✅ Clean slate, no upload corruption

2. **Resume Attempt** (risky):
   - Keep same run ID
   - Hope wandb backend recovered
   - ⚠️ May hit same 502 error
   - ⚠️ Step numbers might conflict (epochs 7-8 logged twice)

**Recommendation**: Fresh start with new run ID when restarting Exp1
```bash
# Before restart, archive old wandb data
mv results/local_fla_exp1_reg/wandb results/local_fla_exp1_reg/wandb_old_run
# Then restart will create fresh wandb run
```

### Issue 2: Inconsistent Early Stopping
**Problem**: Configs had different patience values
**Impact**: Unfair comparison (baseline patience=5, updated to 20)
**Mitigation**: ✅ Standardize all experiments to patience=20, min_epochs=30

### Issue 3: WSL2 Stability
**Problem**: Long-running tmux sessions can crash
**Mitigation**:
- Mid-epoch checkpoints every 30 min
- `--resume` flag recovers from any crash
- Monitor with `tmux attach -t <session>`

---

## 🎓 Lessons from Baseline Run

### What Went Right
✅ Training stable (no NaN/Inf issues)
✅ Achieved 0.28 sensitivity @ 10FA (publishable)
✅ Checkpointing worked perfectly

### What Went Wrong
❌ Stopped too early (epoch 13, only 4 epochs past best)
❌ patience=5 too aggressive for 100-epoch plan
❌ Wandb upload failed (502 error) - lost training curves

### Improvements Applied
✅ patience: 5 → 20
✅ min_epochs: 0 → 30
✅ Local metrics tracking (*.md files)
✅ Clear experimental protocol (this document)

---

## 🚀 Next Actions (Priority Order)

### 1. ~~Fix Exp1 Config~~ ✅ DONE
**Status**: Config updated (patience: 5→20, min_epochs: 0→30)
**Also fixed**: Exp2 config (same issue)

**Restart Pending**: Waiting for epoch 8 validation to complete (~5 min remaining)

### 2. Restart Exp1 with Fixed Config (NEXT)
```bash
# After epoch 8 completes:
# 1. Archive old wandb data (fix 502 issue)
mv results/local_fla_exp1_reg/wandb results/local_fla_exp1_reg/wandb_old_run

# 2. Restart training
tmux kill-session -t exp1-reg-resume
tmux new -s exp1-reg-resume
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml --resume
```

**Why now**: Config changed mid-run, need fresh start with new early stopping

### 3. Resume Baseline (CRITICAL)
```bash
tmux new -s baseline-resume
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla.yaml --resume
```

**Why critical**: Need TRUE baseline performance to compare against

### 3. Start Exp2 (WHEN READY)
```bash
# After baseline resumes
tmux new -s exp2-lr
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp2_lr.yaml
```

### 4. Create Tracking Files
- `TRAINING_METRICS_BASELINE.md`
- `TRAINING_METRICS_EXP2_LR.md`

---

## 📖 References

- **Baseline config**: `configs/local/train_fla.yaml`
- **Exp1 config**: `configs/local/train_fla_exp1_reg.yaml`
- **Exp2 config**: `configs/local/train_fla_exp2_lr.yaml`
- **Exp1 tracker**: `TRAINING_METRICS_EXP1_REG.md`
- **Wandb audit**: `WANDB_AUDIT_REPORT.md`

---

## 🎯 Decision Criteria

After all experiments complete (epochs 25-50):

### Scenario 1: Baseline Best
**Decision**: Use baseline config for production
**Next**: Explore architecture changes (different Mamba variants, etc.)

### Scenario 2: Exp1 (Regularization) Best
**Decision**: Adopt stronger regularization
**Next**: Fine-tune dropout/weight_decay ratios

### Scenario 3: Exp2 (Lower LR) Best
**Decision**: Use lr=5e-5 for future training
**Next**: Explore even lower LRs (1e-5) or adaptive schedules

### Scenario 4: All Similar
**Decision**: Baseline is robust, hyperparameters not critical
**Next**: Focus on data augmentation, architecture search

---

**Generated**: 2025-10-22 19:45 EDT
**Owner**: @jj
**Status**: Active
**Next Review**: After Exp2 completes (~Nov 12)
