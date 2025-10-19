# Next Steps - Brain-Go-Brr Training & Evaluation

**Status**: Baseline training incomplete (stopped epoch 13/100), Exp1 training in progress (epoch 3/100)

**Date**: October 19, 2025

---

## Current Training Status

### Baseline Run (local_fla_training)
- **Status**: Manually stopped during epoch 13
- **Best performance**: Epoch 9 - 28.01% sensitivity@10FA
- **Problem identified**: Clear overfitting after epoch 6
  - val_loss rising (0.027 → 0.053)
  - train_loss falling (0.05 → 0.008)
  - Sensitivity metrics declining after peak at epochs 6-9

**Available checkpoints**:
- `best.pt` - Epoch 9, 28.01% sensitivity@10FA (USE THIS for eval!)
- `epoch_012.pt` - Last completed epoch
- `signal_exit.pt` - Mid-epoch 13 (incomplete)

**Location**: `/results/local_fla_training/checkpoints/`

### Exp1 Run (local_fla_exp1_reg) - Stronger Regularization
- **Status**: Running epoch 3/100 (training)
- **Config changes** (stronger regularization):
  - TCN dropout: 0.15 → 0.20
  - Mamba dropout: 0.1 → 0.2
  - Weight decay: 0.01 → 0.05
- **Epoch 1 results**: 20.45% sensitivity@10FA
  - Lower than baseline epoch 1 (~25%) - EXPECTED with stronger regularization
- **Epoch 2 results**: 25.30% sensitivity@10FA (improving! +4.85% from epoch 1)
  - Goal: Prevent overfitting seen in baseline, maintain/improve past epoch 6

**Available checkpoints**:
- `best.pt` - Epoch 2, 25.30% sensitivity@10FA
- `epoch_001.pt` - Epoch 1 complete
- `epoch_002.pt` - Epoch 2 complete
- `mid_epoch_003_004151.pt` - Latest mid-epoch checkpoint (epoch 3 in progress)

**Location**: `/results/local_fla_exp1_reg/checkpoints/`

---

## Understanding Checkpoint Files

### What's in a `.pt` checkpoint file?

**File size**: ~189 MB (baseline), ~198 MB (Exp1 - slightly larger due to optimizer state growth)

**Contents** (from `best.pt` example):
```
Top-level keys:
├── model_state_dict      ← **FULL MODEL WEIGHTS** (411 tensors, ~180MB)
│   ├── tcn_encoder.*     - TCN conv layers
│   ├── mamba.*           - BiGatedDeltaNet/FLA layers
│   ├── gnn.*             - Graph neural network layers
│   └── classifier.*      - Output head
├── optimizer_state_dict  - Adam optimizer (for resuming training)
├── scheduler_state_dict  - Learning rate scheduler
├── scaler_state_dict     - Mixed precision training
├── rng_state             - Random number generators
├── config                - Full training config
├── epoch                 - Which epoch this checkpoint is from
├── best_metric           - Best validation metric achieved
├── global_step           - Total training steps
└── timestamp             - When checkpoint was saved
```

**For inference, we only need**:
1. `model_state_dict` (the trained weights)
2. Model architecture code (in `src/brain_brr/models/`)

**How to load for inference**:
```python
import torch
from src.brain_brr.models.detector import SeizureDetector

# Load checkpoint
ckpt = torch.load('results/local_fla_training/checkpoints/best.pt', map_location='cpu')

# Create model with same architecture
model = SeizureDetector(config)

# Load trained weights
model.load_state_dict(ckpt['model_state_dict'])

# Ready for inference!
model.eval()
```

---

## TUSZ Dataset Split Status

### Current Usage
- **train/** (4667 files): Used for training both baseline and Exp1
- **dev/** (1832 files): Used for validation during training both runs

### NOT TOUCHED YET
- **eval/** or **test/**: Official TUSZ test set - **WE HAVEN'T EVALUATED ON THIS YET!**

**Key insight**: All our current metrics (28.01% sensitivity@10FA) are from the **dev set**, not the official test set!

---

## Should We Run Eval/Test Despite Overfitting? YES! 🎯

**Critical Question**: The baseline model clearly overfits (val_loss rising after epoch 6). Does this mean test set results would be meaningless?

**Answer: NO! Test results are EXTREMELY meaningful, even more so BECAUSE of the overfitting!**

### Why Test Set Results Matter Despite Overfitting

#### 1. Overfitting is on TRAINING data, not TEST data

**What actually happened:**
- Model memorized patterns in **train set** (4667 files)
- Model validated against **dev set** (1832 files) during training
- Model has **NEVER seen** the eval/test set

**What overfitting means:**
- Gap between train performance (great) and dev performance (declining)
- Does NOT mean the test set results are invalid
- Test set shows **true generalization** to unseen data

#### 2. Test Performance Quantifies HOW BAD the Overfitting Is

**Expected scenario:**
```
Baseline Epoch 9:
├── Train loss: ~0.015 (low, model learned training data well)
├── Dev sensitivity@10FA: 28.01% (validation set during training)
└── Test sensitivity@10FA: ??? (we expect LOWER, maybe 23-26%)
```

**The dev→test gap is the KEY metric:**
- Small gap (28% → 27%): Overfitting not too bad, model generalizes okay
- Large gap (28% → 20%): Severe overfitting, model memorized training data
- **This gap quantifies the overfitting problem objectively!**

#### 3. Even Overfit Models Can Be Clinically Useful

**Perspective:**
- Random baseline: ~8% (class distribution)
- Overfit model test result: Maybe 24-26%?
- That's still **3x better than random!**

**This tells us:**
- The architecture CAN learn seizure patterns
- The approach is fundamentally sound
- We just need better regularization (Exp1!)

#### 4. Establishes Scientific Baseline for All Future Work

**You NEED this for rigorous research:**

| Experiment | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | Analysis |
|------------|---------------|----------------|--------------|----------|
| Baseline (overfit) | 28.01% | **???** | **???** | Overfitting hurts generalization by X% |
| Exp1 (stronger reg) | TBD | TBD | TBD | Regularization reduces gap to Y% |

**Without baseline test results:**
- ❌ Can't quantify overfitting impact
- ❌ Can't prove Exp1 actually improved generalization
- ❌ Can't compare to literature benchmarks
- ❌ Can't write a proper research paper

#### 5. Tests Your Full Evaluation Pipeline

**Even if model performance is suboptimal, you validate:**
- ✅ Inference code works correctly
- ✅ Data loading works on test set
- ✅ Metric calculation is correct
- ✅ Post-processing (hysteresis, morphology) works
- ✅ No bugs in evaluation pipeline
- ✅ End-to-end pipeline is sound

**Better to find bugs NOW with baseline than later with your best model!**

#### 6. Required for Research/Publication

**Any ML paper requires:**
```
Table 1: Model Performance on TUSZ Test Set

Model               | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | Notes
--------------------|---------------|----------------|--------------|------------------
Baseline            | 28.01%        | XX.XX%         | X.XX%        | Overfitting observed
+ Regularization    | XX.XX%        | XX.XX%         | X.XX%        | Improved generalization
+ Data Augmentation | XX.XX%        | XX.XX%         | X.XX%        | Further improvement
```

**You can't skip the baseline test results just because it overfit - that's scientifically invalid!**

### What Test Results WILL Tell You

✅ **How much overfitting hurt real-world performance** (dev-test gap quantification)
✅ **Whether the model beats random baseline** (~8% seizure rate)
✅ **Actual clinical utility** (even if not SOTA, might be useful)
✅ **Distribution shift** (if any) between dev/test splits
✅ **Baseline to beat** for all future experiments (critical reference point)
✅ **Pipeline correctness** (validates entire inference/eval pipeline)
✅ **Proof of concept** (architecture can learn seizure detection)

### What Test Results WON'T Tell You

❌ **Model's true potential** (limited by overfitting, not architecture)
❌ **Best possible performance** (that's what Exp1+ experiments are for)
❌ **Whether architecture is optimal** (need more experiments to determine)
❌ **Final production-ready model** (needs more refinement)

### The Scientific Value

**Overfitting makes test results MORE meaningful, not less:**

1. **Quantifies the problem**: Measures exact generalization gap
2. **Validates the solution**: Exp1 must reduce this gap to prove effectiveness
3. **Establishes rigor**: Shows you understand train/dev/test methodology
4. **Enables comparison**: Future experiments measured against this baseline

**Example narrative for paper:**
```
"Our baseline model achieved 28.01% sensitivity@10FA on the development set but
only 24.3% on the test set (3.7% gap), indicating overfitting. With stronger
regularization (dropout 0.4, weight_decay 1e-3), we achieved 29.2% on dev and
27.8% on test (1.4% gap), demonstrating both improved performance AND better
generalization (62% reduction in overfitting gap)."
```

### Recommendation: Two-Track Approach 🚀

**Track 1: Baseline Eval (Do This NOW)**
```bash
# Use best.pt (epoch 9, 28.01% dev)
python -m src eval --checkpoint results/local_fla_training/checkpoints/best.pt --split test

# Expected outcome:
# - Test sensitivity@10FA: 23-27% (lower than dev 28.01%)
# - Quantifies overfitting impact
# - Validates evaluation pipeline
# - Establishes baseline to beat
```

**Track 2: Exp1 Training (Ongoing)**
```
- Let it run to epoch 9+
- Compare dev AND test results to baseline
- Prove regularization helped both:
  a) Performance (higher test score)
  b) Generalization (smaller dev-test gap)
```

**The combination tells the complete story:**
- Baseline: "Here's the problem" (overfitting quantified)
- Exp1: "Here's the solution" (regularization improves generalization)
- Science: Hypothesis → Evidence → Conclusion ✅

### Bottom Line: RUN THE EVAL!

**The overfitting doesn't invalidate test results - it makes them ESSENTIAL!**

Running eval on the overfit baseline:
1. ✅ Quantifies the overfitting problem objectively
2. ✅ Establishes baseline for all future experiments
3. ✅ Tests your evaluation pipeline thoroughly
4. ✅ Required for scientific rigor and publication
5. ✅ Proves architecture viability (even if not optimal)
6. ✅ Enables meaningful comparison with Exp1/Exp2/Exp3

**Not running eval would be like:**
- A doctor diagnosing a problem but never measuring vital signs
- An engineer building a bridge without stress testing
- A scientist proposing a hypothesis without collecting data

**DO IT!** The results will be valuable regardless of the numbers. 🔬

---

## Immediate Next Steps (Priority Order)

### 1. Let Exp1 Continue Training
- Currently at epoch 3/100, training (mid-epoch checkpoint 4151/7702 batches)
- **ETA**: ~5-6 hours for epoch 3 to complete (4.1h train + validation)
- **Key checkpoint**: Epoch 9 (compare to baseline epoch 9)
- **Watch for**: Does stronger regularization prevent overfitting?
- **Current progress**: Epoch 2 showed improvement (25.30% vs 20.45% in epoch 1)

### 2. Locate TUSZ Eval/Test Set
**Questions to answer**:
- Do we have the TUSZ eval/test set downloaded?
- Where is it located?
- Does it need preprocessing like train/dev?

**Action**: Search for TUSZ test data
```bash
# Check common locations
ls -lh /path/to/tusz/data/eval/
ls -lh /path/to/tusz/data/test/
```

### 3. Create Inference Script
**What we need**:
- Load `best.pt` checkpoint (baseline epoch 9)
- Run inference on eval/test set
- Calculate official metrics:
  - Sensitivity @ 10FA/24h
  - Sensitivity @ 5FA/24h
  - Sensitivity @ 1FA/24h
  - TAES score

**Benefits**:
- Get official baseline numbers to beat
- Validate dev set results generalize to test set
- Establish ground truth for experiment comparisons

### 4. Set Up Experiment Tracking
**Create comparison table**:
| Experiment | Config | Best Epoch | Sens@10FA (dev) | Sens@10FA (test) | Status |
|------------|--------|------------|-----------------|------------------|--------|
| Baseline   | TCN_drop=0.15, Mamba_drop=0.1, wd=0.01 | 9 | 28.01% | TBD | Stopped epoch 13 |
| Exp1       | TCN_drop=0.20, Mamba_drop=0.2, wd=0.05 | 2 (so far) | 25.30% | TBD | Running (3/100) |

---

## Key Questions to Answer

### Training Strategy
1. Should we resume baseline training from epoch 13 to 100?
   - Pro: See if it recovers from overfitting
   - Con: Already showing clear overfitting pattern
   - **Recommendation**: No, wait for Exp1 results first

2. When should we stop Exp1 if it's not improving?
   - Early stopping patience currently set in config
   - Monitor validation metrics closely

### Evaluation Strategy
3. Do we have the TUSZ official eval/test set?
   - If yes: Where is it and is it preprocessed?
   - If no: How do we download/access it?

4. Should we evaluate baseline best.pt on test set now or wait?
   - **Recommendation**: Do it now to establish baseline
   - Gives us target to beat while Exp1 trains

### Next Experiments
5. If Exp1 prevents overfitting but sensitivity is lower, what's next?
   - Exp2: Different regularization balance
   - Exp3: Data augmentation
   - Exp4: Architecture changes

6. Should we consider ensemble methods?
   - Combine multiple checkpoints (epochs 6,7,8,9)
   - Might smooth out overfitting

---

## Resource Tracking

### Disk Space
- Baseline checkpoints: ~2.1 GB (11 checkpoints @ 189 MB each)
- Exp1 checkpoints: ~1.3 GB (growing, currently epoch 3 - 7 checkpoints @ 198 MB each)
- Total checkpoint storage: ~3.4 GB

### Compute Time (RTX 4090)
- Baseline: Ran ~13 epochs (stopped Oct 18 13:47)
- Exp1: Started Oct 18 14:52
- **Cost**: Free (local training)
- **Time per epoch**: ~9.6 hours (4.1h train + 5.5h val)

---

## Success Criteria

### Exp1 Success Metrics
- **Minimal success**: Match baseline 28.01% @ epoch 9 without overfitting
- **Good success**: Exceed 28.01% and maintain through epoch 20+
- **Great success**: Achieve 30%+ sensitivity@10FA with stable val_loss

### Evaluation Success
- Test set performance within 2-3% of dev set performance
- No major distribution shift issues
- Reproducible metrics matching TUSZ benchmarks

---

## Open Questions / Brainstorming

1. **Overfitting analysis**: Why did baseline overfit so hard after epoch 6?
   - Model capacity too high (31M params)?
   - Training data too small (4667 files)?
   - Need different regularization strategy?

2. **Best checkpoint selection**: Should we use epoch 9 or ensemble epochs 6-9?
   - Single best epoch might be lucky
   - Ensemble might be more robust

3. **Hyperparameter search**: After Exp1, should we do systematic grid search?
   - Dropout: [0.3, 0.35, 0.4, 0.45]
   - Weight decay: [1e-4, 5e-4, 1e-3, 5e-3]
   - Learning rate: [1e-4, 5e-4, 1e-3]

4. **Training time optimization**: 9.6h per epoch × 100 = 40 days
   - Can we speed up validation? (Currently 5.5h - 57% of epoch time!)
   - Should we validate less frequently after epoch 20?

5. **Early stopping**: What's the right patience?
   - Current setting: TBD (check config)
   - Baseline would have stopped around epoch 12-13 anyway

---

## Action Items (TODO)

- [ ] Check Exp1 epoch 2 results when validation completes (~2-3 hours)
- [x] ~~Locate TUSZ eval/test set~~ (DONE - found at data_ext4/tusz/edf/eval/)
- [ ] Implement NEDCScorer wrapper (~100 lines) for NEDC v6.0.0 scoring
- [ ] Extend `python -m src evaluate` CLI with `--nedc-score` flag
- [ ] Run baseline best.pt on eval set with NEDC scoring
- [ ] Create experiment tracking spreadsheet
- [ ] Monitor Exp1 training through epoch 9 (critical comparison point)
- [ ] Decide on Exp2/Exp3 parameters based on Exp1 results

---

## NEDC Evaluation Pipeline 🎯

**STATUS**: Documentation complete, ready for TDD implementation (October 19, 2025)

**IMPLEMENTATION DOCS** (3 iron-clad docs, zero drift):
- 📋 **`NEDC_EVALUATION_OVERVIEW.md`** - What/Why, architecture, 2-week timeline, expected outputs
- 🔨 **`NEDC_IMPLEMENTATION_GUIDE.md`** - Step-by-step TDD phases (Phase 0-3), complete code samples
- 📚 **`NEDC_REFERENCE.md`** - Dataclasses, CSV_BI format, error tables, bash commands

**CRITICAL**: We only need to implement NEDCScorer (~100 lines) + extend existing CLI (~50 lines)
- ✅ `python -m src evaluate` command exists (cli.py:305)
- ✅ `export_csv_bi()` function exists (events/export.py:15)
- ✅ `validate_epoch()` for inference exists (train/val_step.py:375)
- 🆕 NEDCScorer wrapper (~100 lines) - Direct Python import of nedc-bench
- 🔧 Add `--nedc-score` flag to evaluate CLI (~50 lines)

**IMPLEMENTATION PLAN** (2 weeks, TDD approach):
- **Week 1**: Extend build-cache CLI + preprocess eval set + write 9 tests FIRST (TDD)
- **Week 2**: Implement NEDCScorer (~100 lines) + extend evaluate CLI (~50 lines) + run baseline

**TOTAL EFFORT**: ~150 lines new code, ~500 lines tests

**Start here**: `NEDC_IMPLEMENTATION_GUIDE.md` for TDD step-by-step instructions

---

## Immediate Action Items (Priority Order)

**P0 - Running Now**:
- [ ] Monitor Exp1 epoch 2 validation completion (~2-3 hours)
- [ ] Check if early stopping triggers (patience 4/5)

**P1 - Evaluation Readiness**:
- [ ] Extend build-cache CLI to support eval split (cli.py:207)
- [ ] Preprocess TUSZ eval set to cache (2-4 hours one-time)
- [ ] Update evaluation dataloader to resolve cache/tusz_mmap/{split} automatically
- [ ] Implement NEDCScorer wrapper (nedc_wrapper.py)
- [ ] Add --nedc-score flag to evaluate CLI

**P2 - Baseline Evaluation**:
- [ ] Run baseline best.pt on eval set with NEDC scoring
- [ ] Get official metrics (sensitivity@10FA/5FA/1FA, TAES, F1)
- [ ] Quantify dev-test gap (measures overfitting)
- [ ] Document results for publication

**P3 - Exp1 Comparison** (when Exp1 reaches epoch 9+):
- [ ] Run Exp1 best checkpoint on eval set
- [ ] Compare baseline vs Exp1 (dev AND test metrics)
- [ ] Prove regularization effectiveness
- [ ] Decide on Exp2/Exp3 parameters

---

## Quick Reference: Key Metrics

**Current Performance** (Dev Set):
- Baseline Epoch 9: 28.01% sensitivity@10FA
- Exp1 Epoch 1: 20.45% sensitivity@10FA (expected with stronger regularization)

**Expected Performance** (Eval Set - not measured yet!):
- Baseline: ~24-26% (3-4% gap due to overfitting)
- Exp1: TBD (goal: smaller gap, better generalization)

**Publication Targets**:
- Sensitivity@10FA: >75% (clinical utility threshold)
- Sensitivity@1FA: >75% (gold standard)
- TAES: >0.85 (NEDC benchmark)

**Literature Baseline**:
- Shah et al. 2018: 89% sens@10FA
- Nejedly et al. 2019: 92% sens@10FA
- We're at 28.01% dev, expect ~24% test → significant room for improvement!
