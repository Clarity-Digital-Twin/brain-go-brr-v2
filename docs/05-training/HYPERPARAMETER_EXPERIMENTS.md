# Hyperparameter Experiments Plan (Historical)

**Status**: ⚠️ Historical planning doc (Exp4 complete; see SSOT)
**Created**: 2025-10-18
**Last Updated**: 2025-12-20
**SSOT**: `results/local_fla_exp4_cyclic/eval_results_v2.json` (held-out TUSZ eval)

---

## Current Best (Held-Out TUSZ Eval)

- ✅ FLA Exp4 (Gated DeltaNet): **35.9% sensitivity @ 10 FA/24h** (AUROC 0.8654)
- SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- For updated comparison/targets: `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md`

## 🎯 Baseline Snapshot (Oct 2025)

### Baseline Run: `configs/local/train_fla.yaml` (historical snapshot)

**Final Performance**:
- **Best checkpoint**: epoch 9 (`best.pt`) with `sensitivity_at_10fa = 0.284` (28.4% @ 10 FA/24h) ← **TARGET TO BEAT**
- **Stopping point**: This baseline was later resumed; see `BASELINE_METRICS.md` for the historical timeline snapshot
- **Validation loss**: 0.027 (epoch 3) → 0.053 (epoch 13) ⚠️ Rising
- **Sensitivity**: 0.194 (epoch 0) → 0.284 (epoch 9) → 0.248 (epoch 11) ❌ Declining
- **Early stopping**: patience=5 (would have triggered at epoch 14 anyway)
- **W&B run**: `full_training_fla`
- **Checkpoint directory**: `results/local_fla_training/checkpoints/`

**Config Updated for Future Runs**:
- patience: 5 → 20 (4x more tolerant of plateaus)
- min_epochs: 0 → 30 (prevents premature stopping before 30% complete)
- Rationale: Medical imaging best practices + "second peak" hypothesis

### Key Hyperparameters (Baseline)

```yaml
experiment:
  name: full_training_fla
  output_dir: results/local_fla_training
  seed: 42

model:
  mamba:
    n_layers: 6                    # Mamba layers
    d_model: 512                   # Model dimension
    dropout: 0.1                   # Mamba dropout

  graph:
    enabled: true
    n_layers: 2                    # GNN layers
    dropout: 0.1                   # GNN dropout
    edge_mamba_layers: 2           # Edge Mamba layers
    edge_mamba_d_model: 32         # Edge dimension

training:
  epochs: 100
  batch_size: 8                    # RTX 4090 optimized
  learning_rate: 1.0e-4            # Base LR
  weight_decay: 0.01               # L2 regularization
  gradient_clip: 0.5               # Gradient clipping
  mixed_precision: false           # Disabled on RTX 4090

  loss: focal                      # Focal loss
  focal_alpha: 0.5                 # Neutral weighting
  focal_gamma: 2.0                 # Hard example focus

  scheduler:
    type: cosine
    warmup_ratio: 0.03             # 3% warmup (462 steps)

  early_stopping:
    patience: 20                   # UPDATED from 5 (Oct 2025)
    min_epochs: 30                 # NEW - prevents premature stopping
    metric: sensitivity_at_10fa    # Primary metric
    # NOTE: Experiments use patience=5 for fair comparison to baseline

  checkpoint_interval: 1           # Save every epoch
  mid_checkpoint_interval_s: 1800  # Save every 30 min

data:
  cache_dir: cache/tusz_mmap       # Local NPY cache
  use_balanced_sampling: true      # Oversample seizures
  num_workers: 0                   # WSL2 fix
  pin_memory: true
  prefetch_factor: 2
```

---

## 📊 Problem Diagnosis

### Overfitting Signature (from training.log / W&B):

| Metric | Early (epoch 2-3) | Best checkpoint (epoch 9) | Latest logged (pre-crash) | Status |
|--------|-------------------|---------------------------|----------------------------|--------|
| **val_loss** | 0.0270 | 0.0386 | — (validation crashed) | ❌ Rising |
| **train_loss** | 0.0507 | 0.0120 | — (not recorded post-crash) | ✅ Decreasing |
| **sensitivity@10fa** | 0.1942 | **0.2801** | 0.2801 (unchanged since epoch 9) | 📉 Plateaued |

**Root Cause**: Insufficient regularization → model memorizing training data, compounded by a validation-time CUDA crash before early stopping could fire.

**Solution**: Harden regularization, then re-run with crash mitigation (check GPU thermals / driver state before launching).

---

## 🧪 Planned Experiments (3 Total)

### Strategy:
- **Run sequentially** (not parallel - only 1 GPU)
- **Start with highest-confidence fix** (regularization)
- **Adapt based on results** (decision tree below)
- **Each takes ~4 days** (100 epochs with early stopping)
- **Pre-flight**: Resume the baseline run (from `last.pt`) and confirm it exits via early stopping without a CUDA error before starting Exp1

---

## Experiment 1: Stronger Regularization 🔥

**Priority**: **HIGHEST** (80% confidence this fixes overfitting)

**Hypothesis**: Current dropout (0.1) and weight_decay (0.01) are too weak for 31M parameter model on 4,667 training files.

**Changes from Baseline**:
```yaml
model:
  mamba:
    dropout: 0.2              # UP from 0.1 (100% increase)
  graph:
    dropout: 0.2              # UP from 0.1 (100% increase)

training:
  weight_decay: 0.05          # UP from 0.01 (5× increase)
```

**All other hyperparameters**: Same as baseline

### Config File: `configs/local/train_fla_exp1_reg.yaml`

**Create from**:
```bash
cp configs/local/train_fla.yaml configs/local/train_fla_exp1_reg.yaml
```

**Edit these lines**:
- Line 13: `name: full_training_fla_exp1_reg`
- Line 16: `output_dir: results/local_fla_exp1_reg`
- Line 58: `dropout: 0.2` (under `model.mamba`)
- Line 101: `dropout: 0.2` (under `model.graph`)
- Line 144: `weight_decay: 0.05` (under `training`)

### Run Command:
```bash
# Create tmux session
tmux new -s exp1_reg

# Enable NaN debugging
export BGB_NAN_DEBUG=1

# Start training
.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml

# Detach: Ctrl+B then D
```

### Expected Outcome:
- ✅ **Validation loss stays low** (no renewed climb toward ~0.04)
- ✅ **Training loss decreases slower** (less memorization)
- ✅ **Sensitivity improves** (better generalization)
- ✅ **Later early stopping** (epoch 12-15 instead of 9)

### Success Criteria:
- `val_loss` at best epoch < 0.035 (no spike)
- `sensitivity_at_10fa` > 0.30 (30%+, improvement over 28%)
- Train/val loss gap < 0.026 (beats baseline gap 0.0266)

---

## Experiment 2: Lower Learning Rate ⚡

**Priority**: **MEDIUM** (60% confidence this helps)

**Hypothesis**: Learning rate 1e-4 is too aggressive, causing late-training instability (val_loss spike after epoch 8).

**Changes from Baseline**:
```yaml
training:
  learning_rate: 5.0e-5       # DOWN from 1e-4 (50% reduction)

  scheduler:
    warmup_ratio: 0.05        # UP from 0.03 (longer warmup)
```

**All other hyperparameters**: Same as baseline (including dropout 0.1)

### Config File: `configs/local/train_fla_exp2_lr.yaml`

**Create from**:
```bash
cp configs/local/train_fla.yaml configs/local/train_fla_exp2_lr.yaml
```

**Edit these lines**:
- Line 13: `name: full_training_fla_exp2_lr`
- Line 16: `output_dir: results/local_fla_exp2_lr`
- Line 143: `learning_rate: 5.0e-5` (under `training`)
- Line 166: `warmup_ratio: 0.05` (under `training.scheduler`)

### Run Command:
```bash
# Create tmux session
tmux new -s exp2_lr

# Enable NaN debugging
export BGB_NAN_DEBUG=1

# Start training
.venv/bin/python -m src train configs/local/train_fla_exp2_lr.yaml

# Detach: Ctrl+B then D
```

### Expected Outcome:
- ✅ **More stable convergence** (smoother curves)
- ✅ **Later early stopping** (takes longer to converge)
- ⚠️ **Possibly lower peak performance** (underfitting risk)
- ✅ **Validation loss more stable** (less spiking)

### Success Criteria:
- `val_loss` stable throughout training (no sudden spikes)
- `sensitivity_at_10fa` > 0.28 (at least match baseline)
- Smooth W&B curves (less noise)

---

## Experiment 3: Smaller Model Capacity 🏗️

**Priority**: **MEDIUM** (50% confidence this helps)

**Hypothesis**: 31M parameters is too large for 4,667 training files (6,600 parameters per file). Reducing capacity forces model to learn generalizable features instead of memorizing.

**Changes from Baseline**:
```yaml
model:
  mamba:
    n_layers: 4               # DOWN from 6 (33% reduction)
    d_model: 384              # DOWN from 512 (25% reduction)

  graph:
    n_layers: 1               # DOWN from 2 (50% reduction)
```

**All other hyperparameters**: Same as baseline

### New Model Size:
- **Baseline**: 31M parameters
- **Exp3**: ~17M parameters (45% reduction)

### Config File: `configs/local/train_fla_exp3_smaller.yaml`

**Create from**:
```bash
cp configs/local/train_fla.yaml configs/local/train_fla_exp3_smaller.yaml
```

**Edit these lines**:
- Line 13: `name: full_training_fla_exp3_smaller`
- Line 16: `output_dir: results/local_fla_exp3_smaller`
- Line 54: `n_layers: 4` (under `model.mamba`)
- Line 55: `d_model: 384` (under `model.mamba`)
- Line 100: `n_layers: 1` (under `model.graph`)

**IMPORTANT**: Keep the edge stream at 32 (FLA/Triton requirement from `AGENTS.md`):
- `edge_mamba_d_model` stays **32**
- `gdn_edge_num_heads` and `gdn_edge_headdim` remain 3 and 8 (already satisfy 0.75 × 32 = 24)

### Run Command:
```bash
# Create tmux session
tmux new -s exp3_smaller

# Enable NaN debugging
export BGB_NAN_DEBUG=1

# Start training
.venv/bin/python -m src train configs/local/train_fla_exp3_smaller.yaml

# Detach: Ctrl+B then D
```

### Expected Outcome:
- ✅ **Less overfitting** (smaller capacity = harder to memorize)
- ⚠️ **Possibly lower peak performance** (less model power)
- ✅ **Faster training** (fewer parameters = faster forward/backward)
- ✅ **Better train/val gap** (closer losses)

### Success Criteria:
- `val_loss` doesn't spike (stays < 0.040)
- `sensitivity_at_10fa` > 0.26 (allow slight drop if overfitting fixed)
- Train/val loss gap < 0.020 (clear improvement over baseline's 0.0266 gap)

---

## 📋 Experiment Tracking

### Spreadsheet Format:

| Exp ID | Config | Mamba Layers | d_model | Dropout | Weight Decay | LR | Best Epoch | Dev Sens@10FA | Val Loss (best) | Val Loss (final) | Train/Val Gap | Notes |
|--------|--------|--------------|---------|---------|--------------|-----|-----------|---------------|-----------------|------------------|---------------|-------|
| **baseline** | `train_fla.yaml` | 6 | 512 | 0.1 | 0.01 | 1e-4 | 9 | **0.2801** | 0.0270 | **0.0386** | 0.0266 | ❌ Overfitting + validation crash |
| exp1 | `train_fla_exp1_reg.yaml` | 6 | 512 | **0.2** | **0.05** | 1e-4 | TBD | TBD | TBD | TBD | TBD | Stronger reg |
| exp2 | `train_fla_exp2_lr.yaml` | 6 | 512 | 0.1 | 0.01 | **5e-5** | TBD | TBD | TBD | TBD | TBD | Lower LR |
| exp3 | `train_fla_exp3_smaller.yaml` | **4** | **384** | 0.1 | 0.01 | 1e-4 | TBD | TBD | TBD | TBD | TBD | Smaller model |

### W&B Organization:

**Project**: `seizure-v3-rtx4090`

**Runs**:
- ⚠️ `full_training_fla` (baseline attempt - crashed at epoch 13 validation)
- 🔜 `full_training_fla_exp1_reg` (regularization)
- 🔜 `full_training_fla_exp2_lr` (learning rate)
- 🔜 `full_training_fla_exp3_smaller` (smaller model)

**Compare URL**:
```
https://wandb.ai/<your-username>/seizure-v3-rtx4090/table?workspace=user-<username>
```

---

## 🌲 Decision Tree: What to Do Based on Exp1 Results

### Scenario 1: Exp1 WORKS ✅

**Signs**:
- `sensitivity_at_10fa` > 0.30 (30%+)
- `val_loss` stays low (< 0.035, no spike)
- Train/val gap < 0.020

**Next Steps**:
1. ✅ **Declare Exp1 winner** (use for Phase 4 eval)
2. 🤔 **Optional**: Run Exp2 to see if we can do even better
3. 🤔 **Optional**: Combine Exp1 + Exp2 (stronger reg + lower LR)
4. ✅ **Skip Exp3** (problem solved, don't need smaller model)

---

### Scenario 2: Exp1 PARTIALLY WORKS ⚠️

**Signs**:
- `sensitivity_at_10fa` improved but < 0.30 (e.g., 0.29)
- `val_loss` better but still spikes slightly (e.g., 0.040)
- Train/val gap improved but still large (e.g., 0.030)

**Next Steps**:
1. ✅ **Run Exp2** (lower LR might help stability)
2. ✅ **Run Exp3** (smaller model might be needed)
3. 🤔 **Consider combo**: Exp1 + Exp2 (reg + LR)
4. 📊 **Compare all 3** and pick best

---

### Scenario 3: Exp1 FAILS ❌

**Signs**:
- `sensitivity_at_10fa` same or worse (≤ 0.28)
- `val_loss` still spikes (> 0.045)
- Train/val gap same or worse (> 0.040)

**Next Steps**:
1. ✅ **Run Exp2 immediately** (might be LR issue, not regularization)
2. ✅ **Run Exp3 immediately** (might be capacity issue)
3. 🔬 **Reassess hypothesis** (might need architectural changes)
4. 📖 **Review training logs** (look for other issues - NaNs, gradient spikes)

---

### Scenario 4: Exp1 OVERFITS LESS BUT UNDERFITS ⚠️

**Signs**:
- `val_loss` doesn't spike (✅)
- BUT `train_loss` stays high (e.g., 0.020 instead of 0.007)
- `sensitivity_at_10fa` drops (< 0.26)

**Diagnosis**: Regularization too strong!

**Next Steps**:
1. ✅ **Run Exp2** (lower LR + baseline regularization)
2. 🔧 **Create Exp1b**: Moderate regularization
   - `dropout: 0.15` (between 0.1 and 0.2)
   - `weight_decay: 0.03` (between 0.01 and 0.05)
3. 📊 **Compare**: Exp1b might be sweet spot

---

## 🚀 Execution Plan (Next 12-16 Days)

### Week 1 (Days 1-4): Exp1 (Regularization)
```bash
# Day 1: Create config
cp configs/local/train_fla.yaml configs/local/train_fla_exp1_reg.yaml
# Edit: dropout 0.2, weight_decay 0.05

# Day 1-4: Train
tmux new -s exp1_reg
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml

# Day 4: Analyze results
# - Check W&B dashboard
# - Compare with baseline
# - Decide next experiment based on decision tree
```

### Week 2 (Days 5-8): Exp2 or Exp3 (Based on Exp1)

**If Exp1 works**:
- ✅ Optional: Run Exp2 to see if we can improve further
- ✅ If Exp1 good enough (>35%), skip to Phase 4 (eval)

**If Exp1 partial/fails**:
- ✅ Run Exp2 (lower LR)

### Week 3 (Days 9-12): Exp3 or Finalize

**If needed**:
- ✅ Run Exp3 (smaller model)

**If winner found**:
- ✅ Document best config
- ✅ Prepare for Phase 4 (eval set validation)

---

## 📊 Phase 4: Final Evaluation (After Winner Found)

### Goal: Get true performance on held-out `eval` set

**Process**:
1. ✅ **Pick THE BEST model** from {baseline, exp1, exp2, exp3}
2. ✅ **Load best checkpoint** (e.g., `results/local_fla_exp1_reg/checkpoints/best.pt`)
3. ✅ **Run inference on eval set ONCE**:
   ```bash
   .venv/bin/python -m src eval \
     --checkpoint results/local_fla_exp1_reg/checkpoints/best.pt \
     --split eval \
     --output results/eval_final.json
   ```
4. ✅ **Document metrics**:
   - Sensitivity @ 1/2.5/5/10 FA/24h
   - TAES score
   - Confusion matrix
   - Example predictions
5. ✅ **Compare to TUSZ leaderboard**
6. ✅ **Write results** in `RESULTS.md` or paper

**CRITICAL**: Do NOT tune hyperparameters after seeing eval results!

---

## 🔧 Monitoring Tips

### During Training:
```bash
# Check tmux session
tmux ls
tmux attach -t exp1_reg

# Live training log
tail -f results/local_fla_exp1_reg/training.log

# Latest checkpoint
ls -lht results/local_fla_exp1_reg/checkpoints/ | head

# W&B dashboard
# https://wandb.ai/<your-username>/seizure-v3-rtx4090
```

### After Each Epoch:
```bash
# Check validation metrics
grep "sensitivity_at_10" results/local_fla_exp1_reg/training.log | tail -5

# Check for overfitting
grep "val_loss" results/local_fla_exp1_reg/training.log | tail -10
grep "train_loss" results/local_fla_exp1_reg/training.log | tail -10
```

---

## 🎯 Success Metrics (Phase 2 Goal)

### Minimum Success:
- ✅ Fix overfitting (val_loss doesn't spike)
- ✅ Match or beat baseline (sensitivity > 0.28)
- ✅ Stable training (smooth W&B curves)

### Good Success:
- ✅ Sensitivity > 0.32 (32% @ 10 FA/24h)
- ✅ Val loss < 0.035 at best epoch
- ✅ Train/val gap < 0.020

### Excellent Success:
- ✅ Sensitivity > 0.35 (35% @ 10 FA/24h)
- ✅ Val loss < 0.030 at best epoch
- ✅ Train/val gap < 0.015
- ✅ Ready for production testing

### Clinical Target (Phase 4):
- 🎯 Sensitivity > 0.40 (40% @ 10 FA/24h) on eval set
- 🎯 This is minimum for clinical utility

---

## 📝 Notes & Observations

### Experiment Log:

**Baseline** (aborted mid-validation):
- Crash: `CUDA error: unknown error` during epoch 13 validation sweep
- Best checkpoint: epoch 9 (`sensitivity_at_10fa = 0.2801`, val_loss 0.0386, train_loss 0.0120)
- Overfitting signature: val_loss climbed from 0.0270 → 0.0386 while train_loss kept falling
- Patience counter: 3/5 (no improvement since epoch 9)
- Action: Resume from `last.pt` (epoch 12) before launching experiments

**Exp1** (not started):
- TBD

**Exp2** (not started):
- TBD

**Exp3** (not started):
- TBD

---

**Next Action**: Let the resumed baseline finish cleanly, capture metrics, then launch Exp1! 🚀

**Questions?** Update this document as experiments complete and results come in.
