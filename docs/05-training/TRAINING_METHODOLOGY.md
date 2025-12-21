# Training Methodology & Hyperparameter Search Plan

**Status**: ⚠️ Historical planning doc (Exp4 complete; see SSOT)
**Created**: 2025-10-18
**Last Updated**: 2025-12-20
**SSOT**: `results/local_fla_exp4_cyclic/eval_results_v2.json` (held-out TUSZ eval)

---

## Current Best (Held-Out TUSZ Eval)

- ✅ FLA Exp4 (Gated DeltaNet): **35.9% sensitivity @ 10 FA/24h** (AUROC 0.8654)
- SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- For updated comparison/targets: `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md`

---

## 🎯 Snapshot (October 20, 2025)

### Baseline Training Run (FLA - RTX 4090) ✅ COMPLETE

**Config**: `configs/local/train_fla.yaml`
**Status**: **COMPLETE** - Stopped at epoch 13 (benign crash, not resumed)

**Final Results**:
- **Best checkpoint**: `best.pt` at epoch 9 → **sensitivity_at_10fa = 0.284** (28.4% @ 10 FA/24h)
- **Stopping point**: Epoch 13 (crashed during validation, metrics showed overfitting)
- **Early stopping config**: patience=5 (would have triggered at epoch 14 anyway)
- **Decision**: Pragmatic stop - val_loss rising, sensitivity declining, no upward trend

**Post-Mortem Analysis**:
- Validation loss: 0.027 (epoch 3) → **0.053 (epoch 13)** ❌ Rising
- Sensitivity: 0.194 (epoch 0) → **0.284 (epoch 9)** → 0.248 (epoch 11) ❌ Declining
- **Root cause**: patience=5 too aggressive for 100-epoch plan (stopped at 13% complete)
- **Lesson**: Medical imaging best practices suggest min_epochs=30% to catch "second peak"

**Config Updated** (for future runs):
```yaml
early_stopping:
  patience: 20        # Increased from 5 (4x more tolerant)
  min_epochs: 30      # NEW - prevents premature stopping
  metric: sensitivity_at_10fa
```

See `STATUS.md` lines 242-317 for complete baseline training history.

### Current Experiment (FLA Exp1 - Regularization) 🔄 RUNNING

**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Status**: Running at epoch 5 (as of Oct 20, 2025)

**Hypothesis**: Stronger regularization will improve beyond baseline 0.284
**Changes**: dropout 0.1→0.2, weight_decay 0.01→0.05

**Early Results**:
- Best: Epoch 2 → sensitivity_at_10fa = 0.271 (27.1% @ 10 FA/24h)
- Current: Epoch 5, patience counter = 3/5
- **Verdict (preliminary)**: Underperforming baseline by 4.6%

**Expected**: Will stop at epoch 7 (patience=5), then proceed to baseline resume

### Next: Baseline Resume with patience=20 ⏳ QUEUED

**Objective**: Test "second peak" hypothesis
**Timeline**: After exp1 completes (~2 days)
**Config**: Same `train_fla.yaml` with patience=20, min_epochs=30
**Expected outcome**: 70% no improvement, 20% small gain, 10% major win

See `TRAINING.md` for complete execution plan and decision tree.

---

## 📚 ML Research Workflow: Train / Dev / Eval Split Strategy

### The Three Splits (TUSZ Dataset)

| Split | Size | Purpose | When to Use | How Often |
|-------|------|---------|-------------|-----------|
| **train** | 4,667 files | Train models | Every experiment | Unlimited |
| **dev** | 1,832 files | Validate & tune hyperparameters | Every experiment | Unlimited |
| **eval** | 865 files | Final test (unbiased performance) | **ONCE at end** | **ONE TIME ONLY** |

### Critical Rules:

1. **NEVER tune hyperparameters on eval** - This causes data leakage!
2. **dev is your "proxy test set"** - Use it freely for model selection
3. **eval is sacred** - Only look at it once you've picked your final model
4. **Report eval metrics in papers** - dev metrics are for internal tuning only

---

## 🔄 The Correct Research Workflow

### Phase 1: BASELINE ✅ COMPLETE (October 11-18, 2025)

**Status**: ✅ **COMPLETE** - Baseline established at 0.284 @ 10 FA/24h

**Goal**: Establish baseline performance with default hyperparameters

**Process**:
1. Train on `train` split ✅
2. Validate on `dev` split after each epoch ✅
3. Early stopping based on `dev` metric (`sensitivity_at_10fa`) ✅
4. Stopped at epoch 13 (best at epoch 9) ✅
5. **DID NOT touch eval set** ✅

**Output**:
- Best checkpoint: `results/local_fla_training/checkpoints/best.pt` (epoch 9)
- Baseline metric: `sensitivity_at_10fa = 0.284` (28.4% @ 10 FA/24h) ← **TARGET TO BEAT**
- Training curves: W&B dashboard
- Analysis: Val_loss rising, sensitivity declining after epoch 9
- Config updated: patience 5→20, min_epochs=30 added (for future runs)

---

### Phase 2: HYPERPARAMETER SEARCH 🔄 IN PROGRESS (October 18-30, 2025)

**Status**: 🔄 **Exp1 running**, Exp2/3 queued (paused pending baseline resume decision)

**Goal**: Test if hyperparameter changes improve beyond baseline 0.284 @ 10 FA/24h

**Process**:
1. Run experiments sequentially (one GPU) ✅
2. For each experiment:
   - Train on `train` split ✅
   - Validate on `dev` split ✅
   - Early stopping: patience=5 (matches baseline for fair comparison) ✅
   - Track best `dev` metric ✅
3. Compare all to baseline 0.284
4. Pick winner OR resume baseline with patience=20 if all fail
5. **DO NOT touch eval set yet** ✅

**Experiments (Defined in `HYPERPARAMETER_EXPERIMENTS.md`)**:

#### Experiment 1: Stronger Regularization 🔄 RUNNING
**Status**: Epoch 5, best @ epoch 2 (0.271) - **underperforming baseline by 4.6%**

**Config**: `configs/local/train_fla_exp1_reg.yaml`
**Changes**: dropout 0.1→0.2, weight_decay 0.01→0.05

**Hypothesis**: Insufficient regularization causing overfitting
**Reality (prelim)**: Regularization HURTS performance (underfitting rare seizure class)
**Expected**: Will stop at epoch 7, proceed to baseline resume

---

#### Experiment 2: Lower Learning Rate ⏸️ QUEUED
**Status**: Paused pending baseline resume results

**Config**: `configs/local/train_fla_exp2_lr.yaml`
**Changes**: learning_rate 1e-4→5e-5, warmup_ratio 0.03→0.05

**Hypothesis**: LR too high causing late-training instability
**Decision**: Run only if baseline resume shows potential OR exp1 succeeds

---

#### Experiment 3: Smaller Model ⏸️ QUEUED
**Status**: Paused pending baseline resume results

**Config**: `configs/local/train_fla_exp3_smaller.yaml`
**Changes**: mamba layers 6→4, d_model 512→384, GNN layers 2→1 (31M→17M params)

**Hypothesis**: Model too large for 4,667 training files
**Decision**: Run only if baseline resume shows potential OR exp1/2 succeed

---

### Phase 2.5: Baseline Resume with patience=20 ⏳ NEXT (After Exp1)

**Status**: ⏳ **QUEUED** - Will start after exp1 completes (~2 days)

**Goal**: Test "second peak" hypothesis (can baseline improve beyond 0.284 with more patience?)

**Config**: `configs/local/train_fla.yaml` (same as baseline, but patience=20, min_epochs=30)

**Rationale**:
- Medical imaging literature: Best clinical checkpoints often 10-20 epochs after min val_loss
- Baseline stopped at 13% complete (epoch 13/100)
- "Second peak" phenomenon: Sensitivity can improve at epochs 30-50 even as val_loss rises

**Process**:
1. Resume from `results/local_fla_training/checkpoints/last.pt` ✅
2. Config already updated with patience=20, min_epochs=30 ✅
3. Will train to epoch 30-50 (stops when 20 epochs pass without improvement)
4. Compare to baseline 0.284

**Expected outcomes**:
- 70% chance: No improvement, stops at epoch 30-31 (confirms overfitting)
- 20% chance: Small improvement (0.29-0.31), new baseline established
- 10% chance: Significant improvement (>0.32), major win

**Decision tree**: See `TRAINING.md` lines 268-287 for complete decision logic

---

### Phase 3: FINAL EVALUATION (When Ready)

**Status**: 🔜 **Not started** - Waiting for best model from Phases 1-2

**Goal**: Get unbiased estimate on held-out `eval` set

**Process**:
1. Pick THE BEST model (baseline, exp1/2/3, or resumed baseline) ✅
2. Load best checkpoint weights ✅
3. Run NEDC evaluation **ONE TIME** on eval set ✅
4. Report metrics in documentation/paper ✅
5. **DO NOT re-tune after seeing eval results** ✅

**Output**:
- Final reported metric: sensitivity @ FA rates (1/2.5/5/10)
- NEDC OVERLAP + TAES scores
- Comparison to Temple SOTA (4 FA/24h @ 50% sensitivity)
- Confusion matrix + per-patient breakdown

**Command**:
```bash
python -m src evaluate \
  results/<best_model>/checkpoints/best.pt \
  data_ext4/tusz/edf/eval/ \
  --output-json results/final_eval_metrics.json \
  --nedc-score
```
- Example predictions

---

## 📊 Experiment Tracking

### Spreadsheet Format

| Exp ID | Config | dropout | weight_decay | lr | Best Epoch | Dev Sens@10FA | Val Loss | Notes |
|--------|--------|---------|--------------|-----|-----------|---------------|----------|-------|
| baseline | train_fla.yaml | 0.1 | 0.01 | 1e-4 | 8 | 0.2801 | 0.052 | Overfitting detected |
| exp1 | train_fla_exp1_reg.yaml | 0.2 | 0.05 | 1e-4 | TBD | TBD | TBD | Stronger regularization |
| exp2 | train_fla_exp2_lr.yaml | 0.1 | 0.01 | 5e-5 | TBD | TBD | TBD | Lower LR |
| exp3 | train_fla_exp3_patience.yaml | 0.1 | 0.01 | 1e-4 | TBD | TBD | TBD | Early stop earlier |
| exp4 | train_fla_exp4_smaller.yaml | 0.1 | 0.01 | 1e-4 | TBD | TBD | TBD | Smaller model |

### W&B Organization

**Project**: `seizure-v3-rtx4090`
**Runs**:
- `full_training_fla` (baseline)
- `full_training_fla_exp1_reg`
- `full_training_fla_exp2_lr`
- `full_training_fla_exp3_patience`
- `full_training_fla_exp4_smaller`

---

## 🎓 Professional Research Lab Best Practices

### How Top Labs Approach This Stage:

#### 1. **Google DeepMind / OpenAI / Meta AI**
- **Baseline first** (1-2 weeks)
- **Grid search** 10-50 hyperparameter combinations (distributed)
- **Analyze failures** (not just best model)
- **Ablation studies** (what components matter?)
- **Final test on held-out set** (publish those numbers)

#### 2. **Academic Research Labs**
- **Baseline + 3-5 targeted experiments** (resource constraints)
- **Focus on interpretability** (why did it work?)
- **Compare to prior work** (is it actually better?)
- **Statistical significance** (multiple seeds)

#### 3. **Production ML Teams (Industry)**
- **Baseline quickly** (1 week)
- **2-3 high-confidence experiments** (based on prior experience)
- **Pick winner fast** (business deadlines)
- **A/B test in production** (real-world validation)

### Our Approach (Research Project with Limited Compute):

**Week 1**: Baseline (current, almost done)
**Week 2-3**: Run 3-5 targeted experiments (hyperparameter search)
**Week 4**: Analyze results, pick winner
**Week 5**: (Optional) Architecture changes if needed
**Week 6**: Final eval set validation, document results

---

## 🛑 Decision Criteria: When to Stop Iterating?

### Stop Iterating When:

1. **`dev` performance plateaus** - No improvement after 5+ experiments
2. **Time/compute budget exhausted** - Diminishing returns on effort
3. **"Good enough" for application** - Meets clinical requirements
4. **Ready to publish** - Sufficient novelty and performance

### Clinical Performance Targets (from REALISTIC_PERFORMANCE_TARGETS.md):

| FA Rate | Target Sensitivity | Status |
|---------|-------------------|--------|
| 10 FA/24h | 40-60% | **Baseline: 28%** ❌ |
| 5 FA/24h | 30-50% | TBD |
| 2.5 FA/24h | 20-40% | TBD |
| 1 FA/24h | 10-30% | TBD |

**Minimum bar**: 40% sensitivity @ 10 FA/24h to be clinically useful

**Current status**: **28%** - Need ~43% improvement (or ~12 percentage points)

---

## 📋 Action Plan: Next Steps After Baseline Completes

### Immediate Actions (This Week):

1. 🔄 **Monitor baseline resume** - Current run is active; verify it finishes validation so early stopping fires cleanly
2. ✅ **Document baseline so far** - Capture metrics, checkpoint hashes, and crash stack trace
3. ✅ **Analyze overfitting** - Confirm validation loss trend and train/val gap
4. 🔜 **Draft experiment configs** - Keep templates ready but block launches until baseline rerun is stable

### Phase 2 Execution (Next Week):

1. **Run Experiment 1** (strongest priority - regularization)
   ```bash
   tmux new -s exp1
   export BGB_NAN_DEBUG=1
   .venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml
   ```

2. **Run Experiment 2** (if exp1 shows promise)
3. **Run Experiment 3** (if still overfitting)
4. **Run Experiment 4** (if capacity issue)

### Analysis (Week 3):

1. Compare all experiments in W&B
2. Plot validation curves side-by-side
3. Identify winner based on:
   - Best `dev` sensitivity@10fa
   - Stable training (no val_loss spike)
   - Generalizable (small train/val gap)

### Final Validation (Week 4):

1. Load best checkpoint
2. Run inference on `eval` set **ONCE**
3. Document final metrics
4. Compare to TUSZ leaderboard
5. Write results in paper/documentation

---

## 🔬 Terminology Clarification

### Checkpoints vs Weights

**Question**: "Are checkpoints turned into weights?"

**Answer**: Checkpoints **contain** weights (plus optimizer state, etc.)

- **Checkpoint** = full training snapshot
  - Model weights (parameters)
  - Optimizer state (Adam momentum, etc.)
  - Scheduler state (learning rate)
  - Epoch number, best metric, RNG state
  - Used for **resuming training**

- **Weights** = just model parameters
  - Can extract from checkpoint with `state_dict['model_state_dict']`
  - Used for **inference only** (no training)

**In practice**:
- During training: Save full checkpoints (`best.pt`, `last.pt`)
- For deployment: Extract weights only (smaller file)
- For experiments: Keep full checkpoints (might resume)

**Example**:
```python
# Load checkpoint for inference
checkpoint = torch.load('best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Inference mode

# Run on eval set
predictions = model(eval_data)
```

---

## 📖 Key Insights

### What We've Learned from Baseline:

1. **Model has sufficient capacity** - Training loss fell from 0.0507 → 0.0120 before the crash
2. **Regularization insufficient** - Validation loss rose from 0.0270 → 0.0386 across epochs 3-8
3. **Early stopping not reached** - Patience counter sits at 3/5 because the run aborted mid-validation
4. **28% sensitivity @ 10 FA/24h** - Still below the 40%+ clinical bar

### What This Means:

- ✅ Architecture is learning (TCN + FLA + GNN stack trains without NaNs)
- ⚠️ Training pipeline hit a CUDA validation crash (needs repro + fix or reliable resume)
- ❌ Hyperparameters need tuning (overfitting is fixable)
- ❌ Performance gap to close (28% → 40%+)

### Why We're Optimistic:

1. **First try** - Baseline hyperparameters, not optimized
2. **Clear problem** - Overfitting has known solutions
3. **Room to grow** - Model learns (not stuck at random)
4. **Professional setup** - Reproducible, well-documented

---

## 🚀 Timeline Estimate

| Phase | Duration | Compute Time | Wall Time |
|-------|----------|--------------|-----------|
| Phase 1: Baseline | ~96h planned (current attempt aborted at epoch 13) | ~96h GPU | 4 days |
| Phase 2: Experiments (5×) | ~480h total | ~480h GPU | 20 days (parallel) or 4 days (sequential planning) |
| Phase 3: Architecture (optional) | ~192h | ~192h GPU | 8 days |
| Phase 4: Final eval | ~10h | ~10h GPU | 1 day |
| **Total (worst case)** | ~778h | ~778h GPU | 33 days |
| **Total (best case)** | ~106h | ~106h GPU | 5 days (if exp1 works!) |

**Note**: With RTX 4090 running 24/7, we can do 1 experiment per 4 days. With smart experiment ordering (run most promising first), we might find winner in 1-2 experiments.

---

## 📝 Summary

### Where We Are:
- 🔄 Baseline run is currently resumed after an epoch 13 validation crash (monitoring for completion)
- ✅ Overfitting pattern confirmed (training vs validation gap)
- ✅ Best checkpoint saved (epoch 9, 28% sensitivity @ 10 FA/24h)

### Where We're Going:
- 🔄 Let the resumed baseline reach a clean early-stopping exit (or document a second failure)
- 🔜 Phase 2: Hyperparameter search (3-5 experiments)
- 🔜 Fix overfitting with stronger regularization
- 🔜 Improve `dev` set performance to 40%+ sensitivity
- 🔜 Final validation on `eval` set (once only!)

### How We'll Get There:
- ✅ Professional ML research workflow (train/dev/eval split)
- ✅ Systematic experiments with version control
- ✅ W&B tracking for all runs
- ✅ Clear decision criteria for iteration
- ✅ Preserve eval set sanctity (no peeking!)

---

**Next Action**: Wait for baseline early stopping, then create experiment configs and start Phase 2. 🚀

**Questions?** Review this document and update as we learn more from experiments.
