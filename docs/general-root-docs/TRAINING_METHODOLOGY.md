# Training Methodology & Hyperparameter Search Plan

**Status**: Baseline run resumed after prior epoch 13 validation crash (patience 3/5, training in progress)
**Created**: 2025-10-18
**Last Updated**: 2025-10-18

---

## 🎯 Current Situation

### Baseline Training Run (FLA - RTX 4090)

**Config**: `configs/local/train_fla.yaml`
**Status**: Resumed and currently training (after validation crash on prior attempt)

**Performance**:
- **Best checkpoint**: `best.pt` saved after epoch 9 with `sensitivity_at_10fa = 0.2801` (28.0%)
- **Latest completed epoch**: 12 (patience counter 3/5 with no new best since epoch 9)
- **Run status**: Current session is a resume from `last.pt`; earlier attempt hit `CUDA error: unknown error` during epoch 13 validation
- **Follow-up**: Monitor the active run until early stopping triggers cleanly; if it aborts again, treat epoch 9 as the baseline checkpoint

**⚠️ OVERFITTING DETECTED**:
- Validation loss: 0.0270 (epoch 3) → **0.0386 (epoch 8)** ❌
- Training loss: 0.0507 (epoch 1) → 0.0120 (epoch 7) ✅
- **Gap widening** = classic overfitting signature (and crash prevented seeing later metrics)

**Decision**: Let the resumed run reach natural early stopping (or confirm a second crash) before starting the hyperparameter search.

---

## 📚 ML Research Workflow: Train / Dev / Eval Split Strategy

### The Three Splits (TUSZ Dataset)

| Split | Size | Purpose | When to Use | How Often |
|-------|------|---------|-------------|-----------|
| **train** | 4,667 files | Train models | Every experiment | Unlimited |
| **dev** | 1,832 files | Validate & tune hyperparameters | Every experiment | Unlimited |
| **eval** | ~1,000 files | Final test (unbiased performance) | **ONCE at end** | **ONE TIME ONLY** |

### Critical Rules:

1. **NEVER tune hyperparameters on eval** - This causes data leakage!
2. **dev is your "proxy test set"** - Use it freely for model selection
3. **eval is sacred** - Only look at it once you've picked your final model
4. **Report eval metrics in papers** - dev metrics are for internal tuning only

---

## 🔄 The Correct Research Workflow

### Phase 1: BASELINE (Current - Week 1)

**Status**: 🔄 In progress (resumed after validation crash; waiting for clean early stopping)

**Goal**: Establish baseline performance with default hyperparameters

**Process**:
1. Train on `train` split
2. Validate on `dev` split after each epoch
3. Early stopping based on `dev` metric (`sensitivity_at_10fa`)
4. Save best checkpoint (epoch 9 in current run)
5. **DO NOT touch eval set**

**Output**:
- Best checkpoint: `results/local_fla_training/checkpoints/best.pt`
- Baseline metric: `sensitivity_at_10fa = 0.2801` (28% @ 10 FA/24h)
- Training curves: W&B dashboard
- Analysis: Overfitting detected (val_loss spike)

---

### Phase 2: HYPERPARAMETER SEARCH (Next - Week 2-3)

**Status**: 🔜 Not started (waiting for Phase 1 completion)

**Goal**: Fix overfitting and improve generalization on `dev` set

**Process**:
1. Define 3-5 targeted experiments (see below)
2. For each experiment:
   - Train on `train` split
   - Validate on `dev` split
   - Track best `dev` metric
   - Save checkpoint and config
3. Compare all experiments in spreadsheet/W&B
4. Pick winner based on `dev` performance
5. **DO NOT touch eval set yet**

**Experiments to Run**:

#### Experiment 1: Stronger Regularization (HIGH PRIORITY)
**Hypothesis**: Model overfitting due to insufficient regularization

**Config**: `configs/local/train_fla_exp1_reg.yaml`
```yaml
training:
  dropout: 0.2              # UP from 0.1 (Mamba)
  weight_decay: 0.05        # UP from 0.01
  gradient_clip: 0.5        # Keep same
model:
  graph:
    dropout: 0.2            # UP from 0.1 (GNN)
```

**Expected outcome**: Slower training, but validation loss doesn't spike

---

#### Experiment 2: Lower Learning Rate (MEDIUM PRIORITY)
**Hypothesis**: Learning rate too high, causing late-training instability

**Config**: `configs/local/train_fla_exp2_lr.yaml`
```yaml
training:
  learning_rate: 5.0e-5     # DOWN from 1e-4
  warmup_ratio: 0.05        # UP from 0.03 (longer warmup)
```

**Expected outcome**: More stable convergence, less spiking

---

#### Experiment 3: Early Stop Earlier (LOW PRIORITY)
**Hypothesis**: Patience too high, allows overfitting to develop

**Config**: `configs/local/train_fla_exp3_patience.yaml`
```yaml
training:
  early_stopping:
    patience: 3             # DOWN from 5
```

**Expected outcome**: Stops at epoch 5-6 before overfitting kicks in

---

#### Experiment 4: Reduce Model Capacity (MEDIUM PRIORITY)
**Hypothesis**: Model too large for dataset size (4,667 train files)

**Config**: `configs/local/train_fla_exp4_smaller.yaml`
```yaml
model:
  mamba:
    n_layers: 4             # DOWN from 6
    d_model: 384            # DOWN from 512
  graph:
    n_layers: 1             # DOWN from 2
```

**Expected outcome**: Less overfitting, possibly lower peak performance

---

#### Experiment 5: Data Augmentation (OPTIONAL)
**Hypothesis**: Implicit regularization via input noise

**Config**: `configs/local/train_fla_exp5_augment.yaml`
```yaml
data:
  augmentation:
    enabled: true
    gaussian_noise_std: 0.01
    time_shift_ms: 50
```

**Expected outcome**: Better generalization, more robust model

---

### Phase 3: ARCHITECTURE SEARCH (Optional - Week 4-5)

**Status**: 🔜 Not started

**Goal**: Try architectural changes if hyperparameter search insufficient

**Potential changes**:
- Swap FLA for BiMamba2 baseline
- Try different GNN architectures (GAT, GCN vs SSGConv)
- Add attention mechanisms
- Try different fusion strategies

**Decision criteria**: Only do this if Phase 2 doesn't improve `dev` performance by >5%

---

### Phase 4: FINAL EVALUATION (Once Only! - Week 6)

**Status**: 🔜 Not started

**Goal**: Get unbiased estimate of true performance on held-out `eval` set

**Process**:
1. Pick THE BEST model from Phases 1-3 (based on `dev` performance)
2. Load best checkpoint weights
3. Run inference on `eval` set **ONE TIME**
4. Report those metrics in documentation/paper
5. **DO NOT re-tune after seeing eval results** (that's cheating!)

**Output**:
- Final reported metric: `sensitivity_at_10fa` on eval set
- Full clinical metrics table (FA rates 1/2.5/5/10)
- TAES score
- Confusion matrix
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
