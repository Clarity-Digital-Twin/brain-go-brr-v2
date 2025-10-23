# 🧪 Experimental Plan - FLA Hyperparameter Study

**Version**: 1.1
**Date**: 2025-10-23 (Updated after Exp1 cancellation)
**Status**: Active - Baseline Resume
**Target**: Optimize FLA (Gated DeltaNet) architecture for clinical EEG seizure detection

---

## 📌 Executive Summary (Oct 23 Update)

**Current Status**:
- ✅ **Baseline**: Running (epoch 13+), wandb tracking perfectly
- ❌ **Exp1 (Regularization)**: Cancelled @ epoch 10, negative result (-2.7% vs baseline)
- ⏸️ **Exp2 (Lower LR)**: On hold, decision after baseline epoch ~17-20

**Key Finding**: **Model is NOT overfitting**
- Exp1 regularization made performance worse (0.2726 vs baseline 0.2801)
- Baseline plateau (epochs 9-12) due to other causes (LR barrier or local optimum)

**Next Milestone**: Baseline epoch 17 (~Oct 25) → Decide if Exp2 needed

**Quick Links**:
- Baseline wandb: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90
- Exp1 analysis: `docs/archive/METRICS_EXP1_REGULARIZATION.md`
- Baseline tracker: `docs/archive/METRICS_BASELINE_FLA.md`

---

## 🎯 Research Question

**Can we improve upon baseline FLA performance (0.28 sensitivity @ 10 FA/24h) through systematic hyperparameter optimization?**

Sub-questions:
1. ✅ **ANSWERED**: Was baseline training stopped too early? → **YES** (stopped epoch 12, best epoch 9)
2. ✅ **ANSWERED**: Is the model overfitting? → **NO** (Exp1 regularization made it worse: 0.2726 vs baseline's 0.2801)
3. ⏸️ **PENDING**: Is the learning rate too high? → Testing via baseline resume first, Exp2 later if needed
4. ⏸️ **PENDING**: Will baseline improve past 0.2801 with more epochs (13-30+)?

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
**Status**: ✅ **RUNNING** (resumed @ epoch 13, Oct 23 17:31)
**History**: Epochs 8-12 (best: epoch 9 @ 0.2801)
**Wandb**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90

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

**Why Resumed** (Oct 23):
- Stopped at epoch 12 with OLD config (patience=5)
- Best was epoch 9 (0.2801) - only 3 epochs of patience used
- NEW config has patience=20, min_epochs=30
- Could improve at epochs 15, 20, 25, 30+
- **Exp1 proved model NOT overfitting** (regularization made it worse)

**✅ COMPLETED**:
```bash
# Baseline resumed successfully @ epoch 13
tmux attach -t baseline-resume
# Wandb continuing RED LINE from previous run
```

---

### Experiment 1: Stronger Regularization

**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Status**: ❌ **CANCELLED** @ Epoch 10 (Oct 23, 17:31)
**Result**: **NEGATIVE** - Regularization hurt performance (-2.7% vs baseline)
**Conclusion**: ✅ **Model is NOT overfitting** (hypothesis disproven)

**Changes vs Baseline**:
```yaml
model:
  mamba.dropout: 0.1 → 0.2       # +100%
  graph.dropout: 0.1 → 0.2       # +100%
  tcn.dropout: 0.15 → 0.20       # +33%

training:
  weight_decay: 0.01 → 0.05      # +400%
```

**Final Performance**:
| Epoch | Sens@10FA | Status |
|-------|-----------|--------|
| 6 | 0.2726 | ⭐ BEST |
| 7-9 | 0.2726 | Plateau |
| 10 | (cancelled) | |

**vs Baseline**: 0.2726 vs 0.2801 = **-2.7% worse**

**Why Cancelled**:
1. ✅ Clear negative result after 4-epoch plateau (epochs 6-9)
2. ✅ Proved hypothesis FALSE (regularization hurts, not helps)
3. ⏱️ Resources better spent on baseline resume
4. 📊 See `METRICS_EXP1_REGULARIZATION.md` for full analysis

**Key Learning**:
- Model is **NOT overfitting** at baseline regularization
- Baseline plateau (epochs 9-12) is NOT due to overfitting
- Likely causes: Learning rate barrier OR true local optimum

---

### Experiment 2: Lower Learning Rate

**Config**: `configs/local/train_fla_exp2_lr.yaml`
**Status**: ⏸️ **ON HOLD** (waiting for baseline epochs 13-20 data)
**Hypothesis**: LR too high → causing late-training instability / plateau at suboptimal point

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

**Decision Criteria for Starting Exp2**:

**IF baseline improves past 0.2801** (epochs 13-20):
- ✅ **DON'T start Exp2** - baseline config is working, just needs time
- ✅ Let baseline continue to convergence

**IF baseline stays at 0.2801** (epochs 13-20):
- ✅ **START Exp2** - test if lower LR can escape plateau
- ✅ Hypothesis: LR=1e-4 too high, preventing fine-tuning

**Action When Ready**:
```bash
tmux new -s exp2-lr
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp2_lr.yaml
```

**Timeline**: Decision after baseline epoch ~17-20 (Oct 26-28)

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

## 📅 Execution Timeline (UPDATED Oct 23)

### Phase 1: ✅ COMPLETED (Oct 23)

| Run | Action | Status |
|-----|--------|--------|
| **Baseline** | Resume from epoch 13 | ✅ **RUNNING** (started Oct 23, 17:31) |
| **Exp1** | Test regularization hypothesis | ❌ **CANCELLED** (negative result @ epoch 10) |
| **Exp2** | Start if needed | ⏸️ **ON HOLD** (decision after baseline epoch ~17-20) |

### Phase 2: Baseline Monitoring (Oct 23 - Nov ~5)

**Key Milestones**:
- ✅ **Epoch 13**: Oct 23, ~21:00 (baseline resumes, first check)
- 📅 **Epoch 15**: Oct 24, ~19:00 (early signal if improving)
- 📅 **Epoch 17**: Oct 25, ~17:00 (Exp2 decision point)
- 📅 **Epoch 20**: Oct 26, ~15:00 (clear pattern should emerge)
- 📅 **Epoch 30**: Nov 2, ~09:00 (min_epochs reached, patience countdown starts)

**What We're Looking For**:
- ✅ Improvement past 0.2801 → Baseline working, just needed time!
- ⏸️ Stays at 0.2801 → LR barrier, start Exp2
- ⚠️ Degradation → Re-evaluate (unlikely based on Exp1)

### Phase 3: Decision & Next Steps (Nov ~5+)

**Scenario A**: Baseline improves
- ✅ Continue baseline to convergence
- ✅ Validate improvement is stable
- ⏸️ Hold Exp2 unless baseline plateaus again

**Scenario B**: Baseline stays flat
- ✅ Start Exp2 (lower LR hypothesis)
- ✅ Compare Exp2 vs baseline convergence
- 📊 Decide optimal LR for architecture

**Scenario C**: Both plateau similarly
- 🔬 Architecture limitations identified
- 🎯 Plan architectural experiments (future work)

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

### Hypothesis 2: Model is Overfitting → ✅ **DISPROVEN**
**Test**: Exp1 (stronger regularization) vs Baseline

**Result** (Exp1 cancelled @ epoch 10):
- Exp1: 0.2726 (best @ epoch 6, plateaued 7-9)
- Baseline: 0.2801 (best @ epoch 9)
- **Exp1 is -2.7% WORSE** → Model is **NOT overfitting**

**Conclusion**:
- ✅ Regularization made performance worse, not better
- ✅ Baseline plateau (epochs 9-12) NOT due to overfitting
- ✅ Other causes: Learning rate barrier OR local optimum
- 📊 Full analysis: `docs/archive/METRICS_EXP1_REGULARIZATION.md`

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
- **Baseline**: `docs/archive/METRICS_BASELINE_FLA.md` (✅ created, archived)
- **Exp1**: `docs/archive/METRICS_EXP1_REGULARIZATION.md` (✅ cancelled, archived)
- **Exp2**: TBD (if needed based on baseline results)

### Update Frequency
- After each epoch validation completes
- Use checkpoint extraction script (see Exp1 tracker for template)

---

## ⚠️ Known Issues & Mitigations

### Issue 1: Wandb Upload Failures → ✅ **RESOLVED**
**Problem**: Exp1 had 502 errors on file_stream endpoint (Oct 22)
**Impact**: Lost epochs 7-8 tracking for Exp1
**Root Cause**: Wandb backend hit server error

**Resolution**:
- ✅ Exp1 cancelled, wandb data archived to `wandb_old_run_oct18_22/`
- ✅ Baseline wandb working perfectly (continuing RED LINE seamlessly)
- ✅ Fresh start avoided upload corruption
- 📊 Audit report: `docs/archive/WANDB_AUDIT_REPORT.md`

### Issue 2: Inconsistent Early Stopping → ✅ **FIXED**
**Problem**: Configs had different patience values (baseline=5, experiments varied)
**Impact**: Would have caused unfair comparison
**Resolution**: ✅ All configs standardized to patience=20, min_epochs=30

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

## 🚀 Next Actions (Updated Oct 23)

### ✅ COMPLETED
1. ✅ Cancelled Exp1 (negative result confirmed)
2. ✅ Resumed baseline @ epoch 13 with patience=20
3. ✅ Fixed all config early stopping inconsistencies
4. ✅ Created tracking files (archived to `docs/archive/`)
5. ✅ Wandb tracking restored (RED LINE continuing)

### 🔄 IN PROGRESS
**Baseline Training** (tmux: `baseline-resume`)
- Currently: Epoch 13, batch ~7600+ (99% done)
- Next checkpoint: Epoch 13 validation (~Oct 23, 21:00)
- Monitor: `tmux attach -t baseline-resume`

### ⏸️ PENDING DECISION (Oct 26-28)
**Exp2 (Lower LR)**: Wait for baseline epochs 15-20 data

**Decision Tree**:
- Baseline improves → Don't start Exp2 (baseline works!)
- Baseline plateaus → Start Exp2 (test LR hypothesis)

### 📊 MONITORING SCHEDULE
| Epoch | ETA | Check |
|-------|-----|-------|
| 13 | Oct 23, 21:00 | First post-resume metric |
| 15 | Oct 24, 19:00 | Early improvement signal |
| 17 | Oct 25, 17:00 | **Exp2 decision point** |
| 20 | Oct 26, 15:00 | Clear pattern confirmation |
| 30 | Nov 2, 09:00 | Min epochs, patience starts |

---

## 📖 References

**Configs**:
- Baseline: `configs/local/train_fla.yaml` ✅ Running
- Exp1: `configs/local/train_fla_exp1_reg.yaml` ❌ Cancelled
- Exp2: `configs/local/train_fla_exp2_lr.yaml` ⏸️ On hold

**Documentation**:
- Baseline tracker: `docs/archive/METRICS_BASELINE_FLA.md`
- Exp1 analysis: `docs/archive/METRICS_EXP1_REGULARIZATION.md`
- Wandb audit: `docs/archive/WANDB_AUDIT_REPORT.md`
- Config review: `docs/archive/CONFIG_REVIEW_FOR_CONSENSUS.md`

**Wandb**:
- Baseline (running): https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90

---

## 🎯 Decision Criteria

After baseline epochs 15-30 complete:

### Scenario 1: Baseline Improves (>0.2801)
**Decision**: ✅ Baseline config works, just needed more epochs!
**Next**:
- Continue to convergence
- No need for Exp2
- Validate on eval set when ready

### Scenario 2: Baseline Plateaus (=0.2801)
**Decision**: ⏸️ Start Exp2 (lower LR) to test if can escape plateau
**Next**:
- Launch Exp2 with lr=5e-5
- Compare convergence patterns
- Choose best performer

### Scenario 3: Exp2 Also Plateaus (~0.28)
**Decision**: 🔬 Hit architecture/data limitation
**Next**:
- Consider architectural experiments
- Explore data augmentation
- Review BiMamba2 alternative

### Scenario 4: Baseline Degrades (<0.2801)
**Decision**: ⚠️ Re-evaluate (unlikely based on Exp1 evidence)
**Next**:
- Investigate why Exp1 ruled out overfitting but baseline degraded
- Likely other issue (data quality, hardware, etc.)

---

**Generated**: 2025-10-22 19:45 EDT
**Updated**: 2025-10-23 17:45 EDT (after Exp1 cancellation)
**Owner**: @jj
**Status**: Active - Baseline Resume
**Next Review**: After baseline epoch 17-20 (Oct 25-26) for Exp2 decision
