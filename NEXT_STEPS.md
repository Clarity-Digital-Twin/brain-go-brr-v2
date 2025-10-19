# Next Steps - Brain-Go-Brr Training & Evaluation

**Status**: Baseline training incomplete (stopped epoch 13/100), Exp1 training in progress (epoch 2/100)

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
- **Status**: Running epoch 2/100 (validating)
- **Config changes**:
  - Dropout: 0.3 → 0.4
  - Weight decay: 1e-4 → 1e-3
- **Epoch 1 results**: 20.45% sensitivity@10FA
  - Lower than baseline epoch 1 (~25%) - EXPECTED with stronger regularization
  - Goal: Prevent overfitting seen in baseline, maintain/improve past epoch 6

**Available checkpoints**:
- `epoch_001.pt` - Epoch 1 complete
- `mid_epoch_002_007539.pt` - Currently saving mid-epoch checkpoints

**Location**: `/results/local_fla_exp1_reg/checkpoints/`

---

## Understanding Checkpoint Files

### What's in a `.pt` checkpoint file?

**File size**: ~189 MB each

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
- Currently at epoch 2/100, validating
- **ETA**: ~2-3 hours for epoch 2 to complete
- **Key checkpoint**: Epoch 9 (compare to baseline epoch 9)
- **Watch for**: Does stronger regularization prevent overfitting?

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
| Baseline   | dropout=0.3, wd=1e-4 | 9 | 28.01% | TBD | Stopped epoch 13 |
| Exp1       | dropout=0.4, wd=1e-3 | TBD | TBD | TBD | Running (2/100) |

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
- Baseline checkpoints: ~2.1 GB (11 checkpoints)
- Exp1 checkpoints: ~1.2 GB (growing, currently epoch 2)
- Total checkpoint storage: ~3.3 GB

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
- [ ] Locate TUSZ eval/test set
- [ ] Write inference script for test set evaluation
- [ ] **NEW**: Write CSV_BI format converter for NEDC scorer integration
- [ ] **NEW**: Create NEDC wrapper module in brain-go-brr-v2
- [ ] Run baseline best.pt on test set
- [ ] Create experiment tracking spreadsheet
- [ ] Monitor Exp1 training through epoch 9 (critical comparison point)
- [ ] Decide on Exp2/Exp3 parameters based on Exp1 results

---

## NEDC-BENCH Evaluation Integration 🎯

**Status**: NEDC-BENCH v1.1.0 imported to `reference_repos/nedc-bench/` (October 19, 2025)

### TL;DR - The ULTRA GOD BANGER Plan 🚀

**Architecture**: Direct Python import (NO Docker, NO data transfer!)

**What We Have**:
- Model predictions: `.npy` files (continuous probability timelines)
- Location: `results/*/predictions/epoch_XXX/*.npy`
- Format: `recording_id_probs.npy` (numpy array, shape: `(n_samples,)`)

**What NEDC Needs**:
- Event lists: `.csv_bi` files (discrete start/stop times)
- Format: `TERM,start_s,stop_s,seiz,1.0` per line

**The Pipeline** (3 simple steps):
1. **Convert**: `.npy` → `.csv_bi` (reuse existing `batch_probs_to_events()`)
2. **Import**: `sys.path.insert()` to add nedc-bench (direct Python import!)
3. **Score**: Call nedc-bench API (pure Python, returns official metrics)

**Files to Create**:
- `src/brain_brr/eval/format_converter.py` (~100 lines, reuses existing code)
- `src/brain_brr/eval/nedc_wrapper.py` (~80 lines, direct Python import)
- `src/brain_brr/eval/evaluator.py` (~150 lines, orchestrates everything)

**Why This is Perfect**:
- ✅ NO Docker complexity
- ✅ NO data transfer overhead
- ✅ Reuses existing post-processing code
- ✅ Official NEDC v6.0.0 metrics
- ✅ Publication-ready results
- ✅ Fast, clean, simple!

### The Complete Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 Brain-Go-Brr → NEDC Evaluation                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Inference (brain-go-brr-v2)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Model (.pt) + Test Data → Raw Predictions                │  │
│  │ - Load checkpoint (best.pt)                              │  │
│  │ - Run on TUSZ eval/test split                            │  │
│  │ - Output: Per-window probabilities + binary predictions │  │
│  │ - Apply post-processing (hysteresis + morphology)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Step 2: Format Conversion (NEEDED - TO BE BUILT)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Raw Predictions → CSV_BI Format                          │  │
│  │ - Convert window predictions to event timings           │  │
│  │ - Generate CSV_BI files per recording                    │  │
│  │ - Format: TERM,start_time,stop_time,seiz,1.0            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Step 3: NEDC Scoring (reference_repos/nedc-bench)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ CSV_BI Files → Official Metrics                          │  │
│  │ - Reference (ground truth) vs Hypothesis (predictions)  │  │
│  │ - Algorithms: TAES, Overlap, DP, Epoch, IRA             │  │
│  │ - Output: Sensitivity@FA rates, TAES score, F1, etc.    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Step 4: Analysis & Publication                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Compare to Literature Benchmarks                         │  │
│  │ - Baseline vs Exp1 (both dev AND test)                  │  │
│  │ - Quantify overfitting (dev-test gap)                   │  │
│  │ - Table-ready results for papers                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Answering Critical Questions

#### Q1: What format are our model outputs after running on eval dataset?

**FOUND THE CODE!** 🎯 (See `src/brain_brr/train/val_step.py:267-301`)

**Current Prediction Format** (what validation actually outputs):

**During validation loop** (`val_step.py:486-509`):
```python
for i, fid in enumerate(file_ids):
    current_windows.append({
        "start_s": float(window_starts[i]),      # Window start time in seconds
        "probs": probs[i].cpu(),                 # PyTorch tensor (60s × sampling_rate)
        "labels": labels[i].cpu(),               # PyTorch tensor (60s × sampling_rate)
    })
```

**Saved to disk** (`val_step.py:267-301`, IF `save_predictions: true`):
```python
# One file per recording, TWO .npy files per recording:
predictions/
├── epoch_009/
│   ├── aaaaaaaa_s001_t000_probs.npy   # Probabilities (numpy array)
│   ├── aaaaaaaa_s001_t000_labels.npy  # Ground truth labels (numpy array)
│   ├── aaaaaaab_s002_t001_probs.npy
│   ├── aaaaaaab_s002_t001_labels.npy
│   └── ...
```

**Format Details**:
- **probs.npy**: Continuous probability timeline (0-1 float per sample)
  - Shape: `(n_samples,)` where `n_samples = recording_duration_sec × sampling_rate`
  - Example: 300s recording × 256 Hz = 76,800 samples
  - Values: Sigmoid outputs from model (per-sample seizure probability)

- **labels.npy**: Ground truth binary labels (0/1 per sample)
  - Shape: Same as probs
  - Values: 1 = seizure, 0 = background

**CRITICAL**: These are NOT event lists! They're continuous timelines!

**What we actually have**:
```
Recording: aaaaaaaa_s001_t000 (300 seconds)
probs.npy:  [0.02, 0.03, 0.05, ..., 0.91, 0.93, 0.95, ..., 0.08, 0.05]
            ↑                        ↑                        ↑
         Background              Seizure detected         Background
         (low prob)              (high prob)              (low prob)
```

**What NEDC-BENCH needs**:
```
CSV_BI format (event list):
channel,start_time,stop_time,label,confidence
TERM,45.2000,67.8000,seiz,1.0
TERM,102.5000,115.3000,seiz,1.0
```

**The Gap**: We need to convert continuous probability timelines → discrete event lists!

#### Q2: Does NEDC scorer accept our format directly or need conversion?

**Answer**: NO - conversion required! ❌

**NEDC-BENCH expects**: CSV_BI format (text files)

**CSV_BI Format Structure**:
```
version = csv_bi_v01.00.00
patient = aaaaaajy
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,1.5000,5.3000,seiz,1.0
TERM,10.2000,15.8000,seiz,1.0
TERM,25.6000,30.1000,seiz,1.0
```

**What this means**:
- Each recording needs its own CSV_BI file
- Events must have precise start/stop times (rounded to 4 decimals)
- Channel is always "TERM" (terminal label, standard for NEDC)
- Confidence is typically 1.0 for binary predictions
- Must include metadata: version, patient, session, duration

**Format Converter Needed**: `brain-go-brr-v2/src/brain_brr/eval/format_converter.py`

**THE CONVERSION PIPELINE**:
```python
# Input: probs.npy (continuous timeline)
probs = np.load("aaaaaaaa_s001_t000_probs.npy")
# Shape: (76800,) for 300s recording @ 256 Hz

# Step 1: Apply threshold + hysteresis (ALREADY implemented!)
from src.brain_brr.eval.metrics import batch_probs_to_events
events = batch_probs_to_events(
    probs,
    post_config,  # Contains tau_on, tau_off, morphology params
    sampling_rate=256
)
# Output: [(45.2, 67.8), (102.5, 115.3)]  # List of (start_s, end_s) tuples

# Step 2: Convert to CSV_BI format
csv_bi_content = f"""version = csv_bi_v01.00.00
patient = aaaaaaaa
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
"""
for start, end in events:
    csv_bi_content += f"TERM,{start:.4f},{end:.4f},seiz,1.0\n"

# Step 3: Write to file
with open("aaaaaaaa_s001_t000.csv_bi", "w") as f:
    f.write(csv_bi_content)
```

**KEY INSIGHT**: We ALREADY have the event conversion code (`batch_probs_to_events`), we just need to wrap it in CSV_BI format!

#### Q3: Should nedc-bench repo contain the converter or should we add it to brain-go-brr-v2?

**Answer**: ADD TO BRAIN-GO-BRR-V2! ✅

**Rationale**:
- **Separation of concerns**:
  - `nedc-bench` is a SCORER - it evaluates predictions in standard format
  - `brain-go-brr-v2` is a MODEL - it's our responsibility to format outputs correctly
- **Different models, different formats**:
  - Every seizure detection model outputs differently (windows, segments, probabilities, etc.)
  - The scorer shouldn't need to know about every model's internal format
  - Each model should provide its own converter
- **Clean architecture**:
  - nedc-bench: Pure evaluation (takes CSV_BI → returns metrics)
  - brain-go-brr-v2: Full pipeline (model → predictions → CSV_BI → metrics)

**Where to add converter**: `src/brain_brr/eval/format_converter.py`

**ULTRA GOD BANGER SIMPLE VERSION** (uses existing code!):
```python
from pathlib import Path
import numpy as np
from src.brain_brr.eval.metrics import batch_probs_to_events
from src.brain_brr.config.schemas import PostprocessingConfig

class CSVBIConverter:
    """Convert brain-go-brr-v2 predictions to NEDC CSV_BI format"""

    def __init__(self, post_config: PostprocessingConfig, sampling_rate: int = 256):
        self.post_config = post_config
        self.sampling_rate = sampling_rate

    def convert_recording(
        self,
        probs_path: Path,          # Path to recording_id_probs.npy
        output_path: Path,         # Path to output .csv_bi file
        patient_id: str,           # e.g., "aaaaaaaa"
        session_id: str,           # e.g., "s001"
        duration_sec: float,       # e.g., 300.0
    ) -> Path:
        """Convert one recording from .npy to CSV_BI format"""

        # Load probabilities
        probs = np.load(probs_path)

        # Convert to events using EXISTING post-processing!
        events = batch_probs_to_events(
            torch.from_numpy(probs).unsqueeze(0),  # Add batch dim
            self.post_config,
            self.sampling_rate
        )[0]  # Get first (only) recording

        # Write CSV_BI format
        with open(output_path, "w") as f:
            f.write(f"version = csv_bi_v01.00.00\n")
            f.write(f"patient = {patient_id}\n")
            f.write(f"session = {session_id}\n")
            f.write(f"duration = {duration_sec:.4f} secs\n")
            f.write(f"\n")
            f.write(f"channel,start_time,stop_time,label,confidence\n")

            for start, end in events:
                f.write(f"TERM,{start:.4f},{end:.4f},seiz,1.0\n")

        return output_path

    def convert_directory(
        self,
        predictions_dir: Path,     # e.g., results/eval_baseline/predictions/epoch_009/
        output_dir: Path,          # e.g., results/eval_baseline/csv_bi/
        metadata: dict,            # {file_id: {"patient": "xxx", "session": "sXXX", "duration": 300.0}}
    ) -> list[Path]:
        """Convert all recordings in directory"""

        output_dir.mkdir(parents=True, exist_ok=True)
        converted = []

        # Find all *_probs.npy files
        for probs_file in predictions_dir.glob("*_probs.npy"):
            file_id = probs_file.stem.replace("_probs", "")

            if file_id not in metadata:
                logger.warning(f"No metadata for {file_id}, skipping")
                continue

            meta = metadata[file_id]
            output_file = output_dir / f"{file_id}.csv_bi"

            self.convert_recording(
                probs_path=probs_file,
                output_path=output_file,
                patient_id=meta["patient"],
                session_id=meta["session"],
                duration_sec=meta["duration"],
            )

            converted.append(output_file)

        return converted
```

**That's IT!** We reuse `batch_probs_to_events()` which ALREADY does hysteresis + morphology!

#### Q4: Should we import nedc-bench as a reference repo and create a wrapper?

**Answer**: YES - EXACTLY! That's the best approach! ✅

**Implementation Strategy**:

**1. Keep nedc-bench as reference repo** (NOT git submodule):
- Location: `reference_repos/nedc-bench/` ✅ DONE
- No git linkage ✅ DONE
- Clean snapshot of your maintained code
- Can update independently when needed

**2. Create NEDC wrapper in brain-go-brr-v2** (ULTRA GOD BANGER - Direct Python Import!):
```python
# src/brain_brr/eval/nedc_wrapper.py

import sys
from pathlib import Path
from typing import Dict, List

# Add nedc-bench to Python path (direct import, NO Docker!)
NEDC_BENCH_PATH = Path(__file__).resolve().parents[3] / "reference_repos" / "nedc-bench" / "src"
sys.path.insert(0, str(NEDC_BENCH_PATH))

# Direct Python imports - NO subprocess, NO CLI, NO Docker!
from nedc_bench.orchestration.dual_pipeline import DualPipeline
from nedc_bench.models.annotations import AnnotationFile

class NEDCScorer:
    """Direct Python integration with NEDC-BENCH (NO Docker, NO subprocess!)"""

    def __init__(self):
        self.pipeline = DualPipeline()

    def score_predictions(
        self,
        reference_dir: Path,       # Ground truth CSV_BI files
        hypothesis_dir: Path,      # Model predictions CSV_BI files
        algorithm: str = "overlap", # overlap, taes, dp, epoch, ira
        pipeline: str = "beta",    # beta (modern) or alpha (legacy)
    ) -> dict:
        """
        Score predictions using NEDC-BENCH Python API (direct import!)

        Returns official metrics:
        - Sensitivity @ 10FA/24h, 5FA/24h, 1FA/24h
        - TAES score
        - F1, Precision, Recall
        - TP, FP, FN counts
        """
        # Load CSV_BI files
        ref_files = list(reference_dir.glob("*.csv_bi"))
        hyp_files = list(hypothesis_dir.glob("*.csv_bi"))

        logger.info(f"[NEDC] Scoring {len(hyp_files)} predictions against {len(ref_files)} references")

        # Direct Python API call - FAST! NO Docker! NO subprocess!
        results = self.pipeline.evaluate(
            reference=ref_files,
            hypothesis=hyp_files,
            algorithms=[algorithm],
            pipeline=pipeline,  # Use modern Beta implementation
        )

        # Extract metrics from NEDC results
        metrics = {
            "sensitivity_at_10FA_24h": results["sensitivity_10fa"],
            "sensitivity_at_5FA_24h": results["sensitivity_5fa"],
            "sensitivity_at_1FA_24h": results["sensitivity_1fa"],
            "taes_score": results.get("taes", None),
            "f1": results["f1"],
            "precision": results["precision"],
            "recall": results["recall"],
            "tp": results["tp"],
            "fp": results["fp"],
            "fn": results["fn"],
        }

        logger.info(f"[NEDC] Sensitivity@10FA: {metrics['sensitivity_at_10FA_24h']:.2f}%")
        logger.info(f"[NEDC] F1: {metrics['f1']:.4f}")

        return metrics
```

**KEY BENEFITS**:
- 🚀 **FAST**: Pure Python, no Docker/subprocess overhead
- 💪 **SIMPLE**: Just `sys.path.insert()` + import
- 🧠 **CLEAN**: All in ONE process
- 📊 **RICH**: Full API access, structured data

**NO Docker! NO data transfer! NO CLI parsing!** This is the ULTRA GOD BANGER way! 💯

**3. Complete workflow wrapper**:
```python
# src/brain_brr/eval/evaluator.py

class ModelEvaluator:
    """End-to-end evaluation pipeline"""

    def __init__(self, checkpoint_path: Path, nedc_bench_path: Path):
        self.model = self.load_model(checkpoint_path)
        self.converter = CSVBIConverter()
        self.scorer = NEDCScorer(nedc_bench_path)

    def evaluate_on_test_set(
        self,
        test_data_dir: Path,
        output_dir: Path,
    ) -> dict:
        """
        Complete evaluation pipeline:
        1. Run inference on test set
        2. Convert predictions to CSV_BI
        3. Score with NEDC-BENCH
        4. Return official metrics
        """
        # Step 1: Inference
        predictions = self.run_inference(test_data_dir)

        # Step 2: Convert to CSV_BI
        hyp_dir = output_dir / "hypothesis"
        csv_bi_files = self.converter.convert_predictions_to_csv_bi(
            predictions, hyp_dir, metadata
        )

        # Step 3: Score with NEDC
        ref_dir = test_data_dir / "labels"  # Ground truth
        metrics = self.scorer.score_predictions(
            reference_dir=ref_dir,
            hypothesis_dir=hyp_dir,
            algorithm="overlap",
        )

        # Step 4: Return results
        return {
            "baseline": metrics,
            "official": True,
            "algorithm": "NEDC v6.0.0 Overlap",
            "comparable_to_literature": True,
        }
```

### Why This Approach is CRITICAL

**Without NEDC scoring**:
- ❌ Can't compare to literature (every paper uses NEDC)
- ❌ Non-standard metrics (people won't trust)
- ❌ Can't publish in journals (they require NEDC scores)
- ❌ No validation against established benchmarks

**With NEDC scoring**:
- ✅ Apples-to-apples comparison with all papers
- ✅ Trusted, validated metrics (NEDC v6.0.0 is the gold standard)
- ✅ Publication-ready results
- ✅ Clinical utility assessment (FA/24h is what clinicians care about!)
- ✅ Multiple scoring algorithms (TAES, Overlap, DP, Epoch, IRA)

### Example Usage

**Command-line workflow**:
```bash
# Step 1: Run inference on TUSZ eval set
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/local_fla_training/checkpoints/best.pt \
  --split eval \
  --output results/eval_baseline/

# This internally:
# 1. Loads model and runs inference
# 2. Converts predictions to CSV_BI format
# 3. Calls nedc-bench scorer
# 4. Outputs official metrics JSON

# Step 2: View results
cat results/eval_baseline/metrics.json
```

**Output format**:
```json
{
  "experiment": "baseline",
  "checkpoint": "best.pt (epoch 9)",
  "split": "eval",
  "algorithm": "NEDC v6.0.0 Overlap",
  "metrics": {
    "sensitivity_at_10FA_24h": 24.3,
    "sensitivity_at_5FA_24h": 21.8,
    "sensitivity_at_1FA_24h": 15.2,
    "taes_score": 0.67,
    "f1": 0.31,
    "precision": 0.42,
    "recall": 0.24,
    "tp": 145,
    "fp": 198,
    "fn": 456
  },
  "comparison_to_dev": {
    "dev_sensitivity_10FA": 28.01,
    "test_sensitivity_10FA": 24.3,
    "gap": 3.71,
    "interpretation": "Overfitting hurts generalization by 3.7%"
  }
}
```

### Literature Comparison Table (What You'll Publish)

```
Table 1: Performance on TUSZ Eval Set (NEDC v6.0.0 Overlap Scoring)

Model              | Sens@10FA | Sens@5FA | Sens@1FA | TAES  | F1   | Dev-Test Gap | Notes
-------------------|-----------|----------|----------|-------|------|--------------|------------------
Baseline           | 24.3%     | 21.8%    | 15.2%    | 0.67  | 0.31 | 3.7%         | Overfitting observed
Exp1 (reg)         | TBD       | TBD      | TBD      | TBD   | TBD  | TBD          | Stronger regularization
Literature [1]     | 89%       | 85%      | 72%      | -     | -    | -            | Shah et al. 2018
Literature [2]     | 92%       | 87%      | 78%      | -     | -    | -            | Nejedly et al. 2019
Target (Clinical)  | >75%      | >75%     | >75%     | -     | -    | -            | <1 FA/24h clinical utility
```

### NEDC-BENCH Integration Checklist

#### Phase 1: Format Conversion (Build This First)
- [ ] Create `src/brain_brr/eval/format_converter.py`
- [ ] Implement `CSVBIConverter` class
- [ ] Handle per-recording predictions → CSV_BI files
- [ ] Test with sample predictions (verify format correctness)
- [ ] Unit tests for converter

#### Phase 2: NEDC Wrapper (Build This Second)
- [ ] Create `src/brain_brr/eval/nedc_wrapper.py`
- [ ] Implement `NEDCScorer` class
- [ ] Test calling nedc-bench (API or CLI)
- [ ] Parse nedc-bench outputs to structured JSON
- [ ] Handle all 5 algorithms (TAES, Overlap, DP, Epoch, IRA)

#### Phase 3: End-to-End Evaluator (Build This Third)
- [ ] Create `src/brain_brr/eval/evaluator.py`
- [ ] Implement `ModelEvaluator` class
- [ ] Wire up: inference → conversion → scoring
- [ ] Add CLI interface for evaluation
- [ ] Generate publication-ready results tables

#### Phase 4: Baseline Evaluation (Do This Now!)
- [ ] Locate TUSZ eval/test set ground truth labels
- [ ] Run baseline best.pt inference on test set
- [ ] Convert predictions to CSV_BI
- [ ] Score with NEDC-BENCH
- [ ] Get official metrics
- [ ] Document results in experiment tracking

#### Phase 5: Exp1 Comparison (Do This Later)
- [ ] Wait for Exp1 epoch 9+ (fair comparison point)
- [ ] Run Exp1 best checkpoint on test set
- [ ] Score with NEDC-BENCH
- [ ] Compare: Baseline vs Exp1 (both dev AND test)
- [ ] Prove regularization improved generalization

### Key Benefits of This Architecture

**Modularity**:
- Each component has single responsibility
- Easy to test independently
- Can swap NEDC version if needed

**Maintainability**:
- nedc-bench updates don't break brain-go-brr-v2
- Format converter evolves with model architecture
- Clean separation between model and evaluation

**Scientific Rigor**:
- Official NEDC v6.0.0 scoring (100% parity validated)
- Directly comparable to all literature
- Multiple algorithms for robustness

**Publication Ready**:
- Standard metrics everyone recognizes
- Literature comparison tables ready to go
- Credible, validated results

---

## NEDC Evaluation Pipeline - Technical Implementation Specification (TDD SSOT)

**Purpose**: Iron-clad specification for test-driven development of the NEDC evaluation pipeline.

**Status**: DOCUMENTATION PHASE - No code written yet, this is the blueprint.

---

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                   NEDC Evaluation Pipeline Architecture            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Component 1: CSVBIConverter (format_converter.py)                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Input:  .npy files (n_samples,) float32 probabilities       │ │
│  │ Process: batch_probs_to_events() → event list                │ │
│  │ Output: .csv_bi text files (NEDC format)                     │ │
│  │ Dependencies: metrics.py, PostprocessingConfig               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          ↓                                         │
│  Component 2: NEDCScorer (nedc_wrapper.py)                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Input:  Reference .csv_bi + Hypothesis .csv_bi dirs          │ │
│  │ Process: sys.path.insert() → nedc_bench import → evaluate()  │ │
│  │ Output: Official NEDC metrics dict                           │ │
│  │ Dependencies: nedc-bench (reference_repos/)                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          ↓                                         │
│  Component 3: ModelEvaluator (evaluator.py)                       │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Input:  Checkpoint .pt + test data dir                       │ │
│  │ Process: load model → inference → convert → score            │ │
│  │ Output: Publication-ready metrics JSON + comparison tables   │ │
│  │ Dependencies: All above + detector.py, val_step.py           │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

### Component 1: CSVBIConverter - Detailed Specification

**File**: `src/brain_brr/eval/format_converter.py`

**Purpose**: Convert brain-go-brr-v2 prediction format (.npy) to NEDC CSV_BI format (.csv_bi)

#### 1.1 Class Signature

```python
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from dataclasses import dataclass

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.eval.metrics import batch_probs_to_events

@dataclass
class RecordingMetadata:
    """Metadata for one recording needed for CSV_BI format"""
    file_id: str          # e.g., "aaaaaaaa_s001_t000"
    patient: str          # e.g., "aaaaaaaa"
    session: str          # e.g., "s001"
    token: str            # e.g., "t000"
    duration_sec: float   # e.g., 300.0
    sampling_rate: int    # e.g., 256

class CSVBIConverter:
    """
    Converts brain-go-brr-v2 predictions to NEDC CSV_BI format.

    Responsibilities:
    1. Load .npy probability files
    2. Apply existing post-processing (batch_probs_to_events)
    3. Format events as CSV_BI text
    4. Write to disk with proper NEDC metadata

    Dependencies:
    - batch_probs_to_events() from metrics.py (EXISTING code)
    - PostprocessingConfig (tau_on, tau_off, morphology params)
    """

    def __init__(
        self,
        post_config: PostprocessingConfig,
        sampling_rate: int = 256,
    ):
        """
        Initialize converter with post-processing configuration.

        Args:
            post_config: Configuration for hysteresis + morphology
            sampling_rate: Sampling rate in Hz (default: 256)

        Raises:
            ValueError: If sampling_rate <= 0
            ValueError: If post_config invalid (tau_on >= tau_off)
        """
        pass

    def convert_recording(
        self,
        probs_path: Path,
        output_path: Path,
        metadata: RecordingMetadata,
    ) -> Path:
        """
        Convert one recording from .npy to .csv_bi format.

        Args:
            probs_path: Path to *_probs.npy file
            output_path: Path where .csv_bi file will be written
            metadata: Recording metadata for CSV_BI header

        Returns:
            Path to created .csv_bi file (same as output_path)

        Raises:
            FileNotFoundError: If probs_path doesn't exist
            ValueError: If probs.npy has wrong shape (must be 1D)
            ValueError: If duration mismatch (probs length vs metadata)
            IOError: If cannot write output_path

        CSV_BI Output Format:
            version = csv_bi_v01.00.00
            patient = {metadata.patient}
            session = {metadata.session}
            duration = {metadata.duration_sec:.4f} secs

            channel,start_time,stop_time,label,confidence
            TERM,{start:.4f},{end:.4f},seiz,1.0
            ...

        Behavior:
        1. Load probs from probs_path
        2. Validate shape: (n_samples,) matches duration × sampling_rate
        3. Convert to events via batch_probs_to_events()
        4. Format as CSV_BI text
        5. Write to output_path atomically
        """
        pass

    def convert_directory(
        self,
        predictions_dir: Path,
        output_dir: Path,
        metadata_dict: Dict[str, RecordingMetadata],
    ) -> List[Path]:
        """
        Convert all *_probs.npy files in directory to .csv_bi format.

        Args:
            predictions_dir: Directory containing *_probs.npy files
            output_dir: Directory where .csv_bi files will be written
            metadata_dict: Map from file_id to RecordingMetadata

        Returns:
            List of paths to created .csv_bi files

        Raises:
            FileNotFoundError: If predictions_dir doesn't exist
            ValueError: If no *_probs.npy files found
            KeyError: If file_id not in metadata_dict

        Behavior:
        1. Create output_dir if doesn't exist
        2. Find all *_probs.npy files
        3. For each file:
           - Extract file_id from filename (strip "_probs.npy")
           - Lookup metadata in metadata_dict
           - Call convert_recording()
           - Log success/failure
        4. Return list of successfully created files
        5. Warn (don't fail) if some files skip due to missing metadata
        """
        pass
```

#### 1.2 Test Specifications for CSVBIConverter

**Test file**: `tests/unit/eval/test_format_converter.py`

```python
import pytest
from pathlib import Path
import numpy as np
from src.brain_brr.eval.format_converter import CSVBIConverter, RecordingMetadata
from src.brain_brr.config.schemas import PostprocessingConfig

class TestCSVBIConverter:
    """TDD test suite for CSVBIConverter"""

    @pytest.fixture
    def post_config(self):
        """Standard post-processing config for testing"""
        return PostprocessingConfig(
            tau_on=0.86,
            tau_off=0.78,
            morphology_opening_kernel_size=11,
            morphology_closing_kernel_size=31,
            min_duration_sec=3.0,
            max_duration_sec=600.0,
            merge_threshold_sec=2.0,
        )

    @pytest.fixture
    def converter(self, post_config):
        """CSVBIConverter instance for testing"""
        return CSVBIConverter(post_config, sampling_rate=256)

    @pytest.fixture
    def sample_metadata(self):
        """Standard recording metadata"""
        return RecordingMetadata(
            file_id="aaaaaaaa_s001_t000",
            patient="aaaaaaaa",
            session="s001",
            token="t000",
            duration_sec=300.0,
            sampling_rate=256,
        )

    @pytest.fixture
    def sample_probs_simple(self, tmp_path):
        """
        Simple probability timeline: 300s recording
        - 0-100s: background (prob=0.1)
        - 100-150s: seizure (prob=0.95)
        - 150-300s: background (prob=0.1)

        Expected events (with tau_on=0.86, tau_off=0.78):
        - One event: ~100.0s to ~150.0s
        """
        n_samples = 300 * 256  # 300s × 256 Hz
        probs = np.full(n_samples, 0.1, dtype=np.float32)
        # Seizure from 100-150s
        probs[100*256:150*256] = 0.95

        probs_path = tmp_path / "test_probs.npy"
        np.save(probs_path, probs)
        return probs_path

    @pytest.fixture
    def sample_probs_complex(self, tmp_path):
        """
        Complex probability timeline: 300s recording
        - Multiple seizures with varying confidence
        - Short events (< 3s) that should be filtered
        - Close events (< 2s apart) that should be merged

        Events:
        - 10-20s: seizure (prob=0.90)
        - 22-24s: seizure (prob=0.92) [should merge with next]
        - 25-45s: seizure (prob=0.95) [merged with previous]
        - 100-101s: seizure (prob=0.88) [too short, filtered]
        - 200-250s: seizure (prob=0.93)

        Expected output (after post-processing):
        - Event 1: ~10.0s to ~20.0s
        - Event 2: ~22.0s to ~45.0s (merged)
        - Event 3: ~200.0s to ~250.0s
        """
        n_samples = 300 * 256
        probs = np.full(n_samples, 0.1, dtype=np.float32)

        # Event 1: 10-20s
        probs[10*256:20*256] = 0.90
        # Event 2: 22-24s
        probs[22*256:24*256] = 0.92
        # Event 3: 25-45s
        probs[25*256:45*256] = 0.95
        # Event 4: 100-101s (too short)
        probs[100*256:101*256] = 0.88
        # Event 5: 200-250s
        probs[200*256:250*256] = 0.93

        probs_path = tmp_path / "test_probs_complex.npy"
        np.save(probs_path, probs)
        return probs_path

    # Test 1: Initialization
    def test_init_valid_config(self, post_config):
        """CSVBIConverter initializes with valid config"""
        converter = CSVBIConverter(post_config, sampling_rate=256)
        assert converter.sampling_rate == 256
        assert converter.post_config == post_config

    def test_init_invalid_sampling_rate(self, post_config):
        """CSVBIConverter raises ValueError for invalid sampling rate"""
        with pytest.raises(ValueError, match="sampling_rate must be > 0"):
            CSVBIConverter(post_config, sampling_rate=0)
        with pytest.raises(ValueError, match="sampling_rate must be > 0"):
            CSVBIConverter(post_config, sampling_rate=-256)

    # Test 2: convert_recording() - Simple case
    def test_convert_recording_simple(
        self, converter, sample_probs_simple, sample_metadata, tmp_path
    ):
        """
        Convert simple recording with one seizure event.

        Input:
        - 300s recording
        - Seizure from 100-150s (prob=0.95)

        Expected CSV_BI:
        - Header with metadata
        - One event line: TERM,100.0000,150.0000,seiz,1.0
        """
        output_path = tmp_path / "test_output.csv_bi"

        result = converter.convert_recording(
            sample_probs_simple, output_path, sample_metadata
        )

        # Verify returned path
        assert result == output_path
        assert output_path.exists()

        # Verify CSV_BI format
        content = output_path.read_text()

        # Check header
        assert "version = csv_bi_v01.00.00" in content
        assert "patient = aaaaaaaa" in content
        assert "session = s001" in content
        assert "duration = 300.0000 secs" in content

        # Check column headers
        assert "channel,start_time,stop_time,label,confidence" in content

        # Check event line(s)
        lines = [l for l in content.split("\n") if l.startswith("TERM,")]
        assert len(lines) == 1

        # Parse event
        parts = lines[0].split(",")
        assert parts[0] == "TERM"
        assert parts[3] == "seiz"
        assert parts[4] == "1.0"

        # Start time should be ~100s (allow ±1s for hysteresis)
        start_time = float(parts[1])
        assert 99.0 <= start_time <= 101.0

        # End time should be ~150s (allow ±1s for hysteresis)
        end_time = float(parts[2])
        assert 149.0 <= end_time <= 151.0

    # Test 3: convert_recording() - Complex case
    def test_convert_recording_complex(
        self, converter, sample_probs_complex, sample_metadata, tmp_path
    ):
        """
        Convert complex recording with multiple events, merging, filtering.

        Input:
        - Events at 10-20s, 22-24s, 25-45s, 100-101s, 200-250s

        Expected output (after post-processing):
        - Event 1: ~10-20s
        - Event 2: ~22-45s (merged 22-24s and 25-45s)
        - Event 3: ~200-250s
        - Event at 100-101s filtered (too short < 3s)
        """
        output_path = tmp_path / "test_output_complex.csv_bi"

        result = converter.convert_recording(
            sample_probs_complex, output_path, sample_metadata
        )

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        lines = [l for l in content.split("\n") if l.startswith("TERM,")]

        # Should have 3 events (one filtered, two merged)
        assert len(lines) == 3

        # Verify event times (approximate due to post-processing)
        events = [(float(l.split(",")[1]), float(l.split(",")[2])) for l in lines]

        # Event 1: ~10-20s
        assert 9.0 <= events[0][0] <= 11.0
        assert 19.0 <= events[0][1] <= 21.0

        # Event 2: ~22-45s (merged)
        assert 21.0 <= events[1][0] <= 23.0
        assert 44.0 <= events[1][1] <= 46.0

        # Event 3: ~200-250s
        assert 199.0 <= events[2][0] <= 201.0
        assert 249.0 <= events[2][1] <= 251.0

    # Test 4: Error handling - File not found
    def test_convert_recording_file_not_found(
        self, converter, sample_metadata, tmp_path
    ):
        """CSVBIConverter raises FileNotFoundError for missing probs file"""
        missing_path = tmp_path / "nonexistent_probs.npy"
        output_path = tmp_path / "output.csv_bi"

        with pytest.raises(FileNotFoundError):
            converter.convert_recording(missing_path, output_path, sample_metadata)

    # Test 5: Error handling - Wrong shape
    def test_convert_recording_wrong_shape(
        self, converter, sample_metadata, tmp_path
    ):
        """CSVBIConverter raises ValueError for wrong probs shape"""
        # Create 2D array instead of 1D
        probs_2d = np.random.rand(300, 256).astype(np.float32)
        probs_path = tmp_path / "probs_2d.npy"
        np.save(probs_path, probs_2d)

        output_path = tmp_path / "output.csv_bi"

        with pytest.raises(ValueError, match="must be 1D"):
            converter.convert_recording(probs_path, output_path, sample_metadata)

    # Test 6: Error handling - Duration mismatch
    def test_convert_recording_duration_mismatch(
        self, converter, sample_metadata, tmp_path
    ):
        """CSVBIConverter raises ValueError if probs length != duration × sampling_rate"""
        # Create probs for 100s instead of 300s
        probs = np.random.rand(100 * 256).astype(np.float32)
        probs_path = tmp_path / "probs_short.npy"
        np.save(probs_path, probs)

        output_path = tmp_path / "output.csv_bi"

        # Metadata says 300s but probs is only 100s
        with pytest.raises(ValueError, match="Duration mismatch"):
            converter.convert_recording(probs_path, output_path, sample_metadata)

    # Test 7: convert_directory() - Multiple files
    def test_convert_directory_success(self, converter, tmp_path):
        """Convert multiple recordings in directory"""
        # Create predictions directory
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()

        # Create 3 sample recordings
        metadata_dict = {}
        for i, file_id in enumerate(["aaa_s001_t000", "bbb_s002_t001", "ccc_s003_t002"]):
            # Create probs file
            probs = np.full(300 * 256, 0.1, dtype=np.float32)
            probs[50*256:100*256] = 0.95  # Simple seizure
            np.save(pred_dir / f"{file_id}_probs.npy", probs)

            # Create metadata
            metadata_dict[file_id] = RecordingMetadata(
                file_id=file_id,
                patient=file_id.split("_")[0],
                session=file_id.split("_")[1],
                token=file_id.split("_")[2],
                duration_sec=300.0,
                sampling_rate=256,
            )

        # Convert directory
        output_dir = tmp_path / "csv_bi"
        result_files = converter.convert_directory(pred_dir, output_dir, metadata_dict)

        # Verify results
        assert len(result_files) == 3
        assert output_dir.exists()

        for file_id in metadata_dict.keys():
            csv_bi_file = output_dir / f"{file_id}.csv_bi"
            assert csv_bi_file in result_files
            assert csv_bi_file.exists()

            # Quick sanity check
            content = csv_bi_file.read_text()
            assert "version = csv_bi_v01.00.00" in content
            assert f"patient = {file_id.split('_')[0]}" in content

    # Test 8: convert_directory() - Missing metadata
    def test_convert_directory_missing_metadata(self, converter, tmp_path):
        """convert_directory() warns but continues if metadata missing for some files"""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()

        # Create 2 probs files
        for file_id in ["aaa_s001_t000", "bbb_s002_t001"]:
            probs = np.full(300 * 256, 0.1, dtype=np.float32)
            np.save(pred_dir / f"{file_id}_probs.npy", probs)

        # Only provide metadata for one file
        metadata_dict = {
            "aaa_s001_t000": RecordingMetadata(
                file_id="aaa_s001_t000",
                patient="aaa",
                session="s001",
                token="t000",
                duration_sec=300.0,
                sampling_rate=256,
            )
        }

        output_dir = tmp_path / "csv_bi"

        # Should succeed for aaa but warn about bbb
        result_files = converter.convert_directory(pred_dir, output_dir, metadata_dict)

        # Only one file converted
        assert len(result_files) == 1
        assert (output_dir / "aaa_s001_t000.csv_bi").exists()
        assert not (output_dir / "bbb_s002_t001.csv_bi").exists()

    # Test 9: Edge case - No events detected
    def test_convert_recording_no_events(
        self, converter, sample_metadata, tmp_path
    ):
        """Convert recording with no seizure events detected"""
        # All background (prob=0.1, below tau_off=0.78)
        probs = np.full(300 * 256, 0.1, dtype=np.float32)
        probs_path = tmp_path / "probs_no_events.npy"
        np.save(probs_path, probs)

        output_path = tmp_path / "output_no_events.csv_bi"

        result = converter.convert_recording(probs_path, output_path, sample_metadata)

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()

        # Header should exist
        assert "version = csv_bi_v01.00.00" in content

        # No event lines (only background)
        lines = [l for l in content.split("\n") if l.startswith("TERM,")]
        assert len(lines) == 0

    # Test 10: CSV_BI format compliance
    def test_csv_bi_format_compliance(
        self, converter, sample_probs_simple, sample_metadata, tmp_path
    ):
        """Verify exact CSV_BI format compliance with NEDC spec"""
        output_path = tmp_path / "test_format.csv_bi"

        converter.convert_recording(sample_probs_simple, output_path, sample_metadata)

        content = output_path.read_text()
        lines = content.split("\n")

        # Line 1: version
        assert lines[0] == "version = csv_bi_v01.00.00"

        # Line 2: patient
        assert lines[1] == "patient = aaaaaaaa"

        # Line 3: session
        assert lines[2] == "session = s001"

        # Line 4: duration (4 decimal places)
        assert lines[3] == "duration = 300.0000 secs"

        # Line 5: blank
        assert lines[4] == ""

        # Line 6: column headers
        assert lines[5] == "channel,start_time,stop_time,label,confidence"

        # Line 7+: event lines (if any)
        event_lines = [l for l in lines[6:] if l.strip()]
        for line in event_lines:
            parts = line.split(",")
            assert len(parts) == 5
            assert parts[0] == "TERM"
            # Times should have 4 decimal places
            assert len(parts[1].split(".")[1]) == 4
            assert len(parts[2].split(".")[1]) == 4
            assert parts[3] == "seiz"
            assert parts[4] == "1.0"
```

#### 1.3 Implementation Acceptance Criteria

**CSVBIConverter is complete when**:
- [ ] All 10 unit tests pass
- [ ] Code coverage ≥ 95% for format_converter.py
- [ ] Can convert sample dev set predictions (manual integration test)
- [ ] Output .csv_bi files validated against NEDC-BENCH parser (no errors)
- [ ] Performance: Can convert 1000 recordings in < 60 seconds
- [ ] Error messages are clear and actionable
- [ ] Logging provides visibility into conversion process

---

### Component 2: NEDCScorer - Detailed Specification

**File**: `src/brain_brr/eval/nedc_wrapper.py`

**Purpose**: Direct Python integration with nedc-bench for official NEDC scoring

#### 2.1 Class Signature

```python
from pathlib import Path
from typing import Dict, List, Literal, Optional
import sys
import logging

# Import nedc-bench via sys.path (NO Docker, NO subprocess!)
NEDC_BENCH_PATH = Path(__file__).resolve().parents[3] / "reference_repos" / "nedc-bench" / "src"
if not NEDC_BENCH_PATH.exists():
    raise ImportError(
        f"NEDC-BENCH not found at {NEDC_BENCH_PATH}. "
        "Please clone https://github.com/Clarity-Digital-Twin/nedc-bench to reference_repos/"
    )
sys.path.insert(0, str(NEDC_BENCH_PATH))

from nedc_bench.orchestration.dual_pipeline import DualPipeline
from nedc_bench.models.annotations import AnnotationFile

AlgorithmType = Literal["overlap", "taes", "dp", "epoch", "ira", "all"]
PipelineType = Literal["alpha", "beta", "dual"]

class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    All metrics directly from NEDC v6.0.0 scoring.
    """
    algorithm: str
    sensitivity_at_10FA_24h: float
    sensitivity_at_5FA_24h: float
    sensitivity_at_1FA_24h: float
    taes_score: Optional[float]  # Only for "taes" algorithm
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    total_seizure_duration_sec: float
    total_recording_duration_sec: float

class NEDCScorer:
    """
    Direct Python integration with NEDC-BENCH (NO Docker, NO subprocess!).

    Responsibilities:
    1. Import nedc-bench Python modules via sys.path
    2. Call DualPipeline.evaluate() with .csv_bi files
    3. Parse NEDC results into structured metrics
    4. Handle all 5 scoring algorithms

    Dependencies:
    - nedc-bench at reference_repos/nedc-bench/ (must exist!)
    """

    def __init__(self):
        """
        Initialize NEDC scorer with dual pipeline.

        Raises:
            ImportError: If nedc-bench not found at reference_repos/
            RuntimeError: If nedc-bench import fails (dependency issues)
        """
        pass

    def score_predictions(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithm: AlgorithmType = "overlap",
        pipeline: PipelineType = "beta",
    ) -> NEDCMetrics:
        """
        Score predictions using NEDC-BENCH Python API.

        Args:
            reference_dir: Directory with ground truth .csv_bi files
            hypothesis_dir: Directory with model prediction .csv_bi files
            algorithm: NEDC scoring algorithm to use
                - "overlap": Any-overlap detection (recommended for seizures)
                - "taes": Time-Aligned Event Scoring
                - "dp": Dynamic Programming alignment
                - "epoch": 250ms epoch-based sampling
                - "ira": Inter-Rater Agreement (Cohen's κ)
                - "all": Run all algorithms (returns list)
            pipeline: Which NEDC pipeline to use
                - "beta": Modern reimplementation (faster, recommended)
                - "alpha": Legacy wrapper (100% parity with NEDC v6.0.0)
                - "dual": Both (for parity validation)

        Returns:
            NEDCMetrics object with official NEDC scores

        Raises:
            FileNotFoundError: If reference_dir or hypothesis_dir doesn't exist
            ValueError: If no .csv_bi files found in directories
            ValueError: If file count mismatch (ref vs hyp)
            RuntimeError: If NEDC scoring fails (invalid CSV_BI format)

        Behavior:
        1. Validate directories exist and contain .csv_bi files
        2. Match reference and hypothesis files by filename
        3. Call nedc_bench.orchestration.dual_pipeline.evaluate()
        4. Parse results into NEDCMetrics
        5. Log summary statistics

        Performance:
        - Overlap algorithm: ~100-200 file pairs per second
        - TAES algorithm: ~50-100 file pairs per second
        - Scales linearly with number of recordings
        """
        pass

    def score_predictions_batch(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithms: List[AlgorithmType],
    ) -> Dict[str, NEDCMetrics]:
        """
        Score predictions with multiple algorithms.

        Args:
            reference_dir: Ground truth .csv_bi directory
            hypothesis_dir: Predictions .csv_bi directory
            algorithms: List of algorithms to run

        Returns:
            Dict mapping algorithm name to NEDCMetrics

        Example:
            results = scorer.score_predictions_batch(
                ref_dir, hyp_dir,
                algorithms=["overlap", "taes", "epoch"]
            )
            print(f"Overlap: {results['overlap'].sensitivity_at_10FA_24h:.2f}%")
            print(f"TAES: {results['taes'].taes_score:.4f}")
        """
        pass

    def validate_csv_bi_format(
        self,
        csv_bi_path: Path,
    ) -> bool:
        """
        Validate CSV_BI file format using nedc-bench parser.

        Args:
            csv_bi_path: Path to .csv_bi file to validate

        Returns:
            True if valid, False otherwise

        Behavior:
        1. Attempt to parse with AnnotationFile.from_csv_bi()
        2. Check for required headers
        3. Validate event format
        4. Return True if valid, False if any errors

        Use this to debug format conversion issues!
        """
        pass
```

#### 2.2 Test Specifications for NEDCScorer

**Test file**: `tests/unit/eval/test_nedc_wrapper.py`

```python
import pytest
from pathlib import Path
from src.brain_brr.eval.nedc_wrapper import NEDCScorer, NEDCMetrics

class TestNEDCScorer:
    """TDD test suite for NEDCScorer"""

    @pytest.fixture
    def scorer(self):
        """NEDCScorer instance for testing"""
        return NEDCScorer()

    @pytest.fixture
    def sample_csv_bi_ref(self, tmp_path):
        """
        Create sample reference .csv_bi file.

        Recording: test_001.csv_bi
        Duration: 300s
        Events:
        - 10-30s: seizure
        - 100-150s: seizure
        """
        csv_bi = """version = csv_bi_v01.00.00
patient = test
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_file = ref_dir / "test_001.csv_bi"
        ref_file.write_text(csv_bi)
        return ref_dir

    @pytest.fixture
    def sample_csv_bi_hyp_perfect(self, tmp_path):
        """
        Hypothesis matching reference perfectly (100% TP, 0% FP/FN).
        """
        csv_bi = """version = csv_bi_v01.00.00
patient = test
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis_perfect"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    @pytest.fixture
    def sample_csv_bi_hyp_partial(self, tmp_path):
        """
        Hypothesis with partial match:
        - Detects 10-30s event (TP)
        - Misses 100-150s event (FN)
        - False alarm at 200-220s (FP)
        """
        csv_bi = """version = csv_bi_v01.00.00
patient = test
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,200.0000,220.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis_partial"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    # Test 1: Initialization
    def test_init_success(self, scorer):
        """NEDCScorer initializes successfully with nedc-bench found"""
        assert scorer is not None
        assert hasattr(scorer, 'pipeline')

    def test_init_nedc_bench_not_found(self, monkeypatch, tmp_path):
        """NEDCScorer raises ImportError if nedc-bench not found"""
        # Mock NEDC_BENCH_PATH to nonexistent location
        import src.brain_brr.eval.nedc_wrapper as wrapper
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr(wrapper, "NEDC_BENCH_PATH", nonexistent)

        with pytest.raises(ImportError, match="NEDC-BENCH not found"):
            NEDCScorer()

    # Test 2: Perfect match scoring
    def test_score_predictions_perfect_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect
    ):
        """
        Score perfect predictions (100% match).

        Expected:
        - Sensitivity: 100%
        - Precision: 100%
        - F1: 1.0
        - TP: 2, FP: 0, FN: 0
        """
        metrics = scorer.score_predictions(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithm="overlap",
        )

        assert isinstance(metrics, NEDCMetrics)
        assert metrics.algorithm == "overlap"
        assert metrics.sensitivity_at_10FA_24h == 100.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.tp == 2
        assert metrics.fp == 0
        assert metrics.fn == 0

    # Test 3: Partial match scoring
    def test_score_predictions_partial_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_partial
    ):
        """
        Score partial predictions (1 TP, 1 FN, 1 FP).

        Reference events: 10-30s, 100-150s
        Hypothesis events: 10-30s, 200-220s

        Expected:
        - TP: 1 (10-30s detected)
        - FN: 1 (100-150s missed)
        - FP: 1 (200-220s false alarm)
        - Sensitivity: 50% (1/2 events detected)
        - Precision: 50% (1/2 predictions correct)
        """
        metrics = scorer.score_predictions(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_partial,
            algorithm="overlap",
        )

        assert metrics.tp == 1
        assert metrics.fn == 1
        assert metrics.fp == 1
        assert abs(metrics.recall - 0.5) < 0.01  # 50% sensitivity
        assert abs(metrics.precision - 0.5) < 0.01  # 50% precision

    # Test 4: Multiple algorithms
    def test_score_predictions_batch(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect
    ):
        """Score with multiple algorithms simultaneously"""
        results = scorer.score_predictions_batch(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithms=["overlap", "epoch", "ira"],
        )

        assert isinstance(results, dict)
        assert "overlap" in results
        assert "epoch" in results
        assert "ira" in results

        for alg, metrics in results.items():
            assert isinstance(metrics, NEDCMetrics)
            assert metrics.algorithm == alg

    # Test 5: Error handling - Directory not found
    def test_score_predictions_dir_not_found(self, scorer, tmp_path):
        """NEDCScorer raises FileNotFoundError for missing directories"""
        nonexistent_ref = tmp_path / "nonexistent_ref"
        nonexistent_hyp = tmp_path / "nonexistent_hyp"

        with pytest.raises(FileNotFoundError):
            scorer.score_predictions(nonexistent_ref, nonexistent_hyp)

    # Test 6: Error handling - No CSV_BI files
    def test_score_predictions_no_files(self, scorer, tmp_path):
        """NEDCScorer raises ValueError if no .csv_bi files found"""
        empty_ref = tmp_path / "empty_ref"
        empty_hyp = tmp_path / "empty_hyp"
        empty_ref.mkdir()
        empty_hyp.mkdir()

        with pytest.raises(ValueError, match="No .csv_bi files found"):
            scorer.score_predictions(empty_ref, empty_hyp)

    # Test 7: CSV_BI format validation
    def test_validate_csv_bi_format_valid(self, scorer, sample_csv_bi_ref):
        """validate_csv_bi_format() returns True for valid file"""
        csv_bi_file = list(sample_csv_bi_ref.glob("*.csv_bi"))[0]
        assert scorer.validate_csv_bi_format(csv_bi_file) is True

    def test_validate_csv_bi_format_invalid(self, scorer, tmp_path):
        """validate_csv_bi_format() returns False for invalid file"""
        invalid_file = tmp_path / "invalid.csv_bi"
        invalid_file.write_text("This is not valid CSV_BI format!")

        assert scorer.validate_csv_bi_format(invalid_file) is False

    # Test 8: Integration with real nedc-bench
    @pytest.mark.integration
    def test_integration_with_nedc_bench_sample_data(self, scorer):
        """
        Integration test with nedc-bench sample data.

        Uses actual sample files from nedc-bench repository.
        Verifies we can call nedc-bench and get real results.
        """
        nedc_bench_root = Path("reference_repos/nedc-bench")
        sample_ref = nedc_bench_root / "data/csv_bi_parity/csv_bi_export_clean/ref"
        sample_hyp = nedc_bench_root / "data/csv_bi_parity/csv_bi_export_clean/hyp"

        if not sample_ref.exists() or not sample_hyp.exists():
            pytest.skip("nedc-bench sample data not found")

        metrics = scorer.score_predictions(
            reference_dir=sample_ref,
            hypothesis_dir=sample_hyp,
            algorithm="overlap",
        )

        # Should get real results (values from nedc-bench parity tests)
        assert metrics.tp > 0
        assert metrics.fp > 0
        assert 0 < metrics.sensitivity_at_10FA_24h < 100
```

#### 2.3 Implementation Acceptance Criteria

**NEDCScorer is complete when**:
- [ ] All 8 unit tests pass (7 unit + 1 integration)
- [ ] Code coverage ≥ 90% for nedc_wrapper.py
- [ ] Integration test with real nedc-bench sample data passes
- [ ] Can score 1000 file pairs in < 30 seconds (overlap algorithm)
- [ ] Error messages clearly indicate format issues
- [ ] Logging provides visibility into scoring process
- [ ] Documentation includes example usage for all 5 algorithms

---

### Component 3: ModelEvaluator - Detailed Specification

**File**: `src/brain_brr/eval/evaluator.py`

**Purpose**: End-to-end evaluation pipeline orchestrator

#### 3.1 Class Signature

```python
from pathlib import Path
from typing import Dict, List, Optional, Literal
import json
import logging
from dataclasses import dataclass, asdict
import torch

from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.config.schemas import TrainingConfig
from src.brain_brr.eval.format_converter import CSVBIConverter, RecordingMetadata
from src.brain_brr.eval.nedc_wrapper import NEDCScorer, NEDCMetrics
from src.brain_brr.train.val_step import run_validation

@dataclass
class EvaluationResults:
    """Complete evaluation results for publication"""
    experiment_name: str
    checkpoint_path: str
    checkpoint_epoch: int
    split: Literal["dev", "eval"]
    algorithm: str
    metrics: NEDCMetrics
    comparison_to_dev: Optional[Dict[str, float]]  # If eval split
    timestamp: str

    def to_json(self, path: Path):
        """Save results to JSON file"""
        pass

    def to_markdown_table(self) -> str:
        """Format as markdown table for publication"""
        pass

class ModelEvaluator:
    """
    End-to-end evaluation pipeline for brain-go-brr-v2.

    Complete workflow:
    1. Load checkpoint
    2. Run inference on test/eval split
    3. Convert predictions to CSV_BI
    4. Score with NEDC-BENCH
    5. Generate publication-ready results

    Usage:
        evaluator = ModelEvaluator(
            checkpoint_path="results/baseline/checkpoints/best.pt",
            output_dir="results/eval_baseline"
        )
        results = evaluator.evaluate_on_split(split="eval", algorithm="overlap")
        print(results.to_markdown_table())
    """

    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        config_override: Optional[TrainingConfig] = None,
    ):
        """
        Initialize evaluator with checkpoint and output directory.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            output_dir: Directory for outputs (predictions, metrics, etc.)
            config_override: Optional config override (uses checkpoint config by default)

        Raises:
            FileNotFoundError: If checkpoint_path doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        pass

    def evaluate_on_split(
        self,
        split: Literal["dev", "eval"],
        algorithm: str = "overlap",
        save_predictions: bool = True,
        save_csv_bi: bool = True,
    ) -> EvaluationResults:
        """
        Complete evaluation pipeline on dev or eval split.

        Args:
            split: Which data split to evaluate on
                - "dev": Validation set (used during training)
                - "eval": Official test set (held-out, never seen)
            algorithm: NEDC scoring algorithm
            save_predictions: Save .npy predictions to disk
            save_csv_bi: Save .csv_bi files to disk (needed for NEDC)

        Returns:
            EvaluationResults with official NEDC metrics

        Workflow:
        1. Load test data for split
        2. Run inference (reuse validation code from train/val_step.py)
        3. Extract metadata from data (patient, session, duration)
        4. Convert predictions to CSV_BI format
        5. Score with NEDC-BENCH
        6. Generate comparison to dev results (if split=="eval")
        7. Save results to JSON
        8. Return EvaluationResults

        Output Structure:
            output_dir/
            ├── predictions/
            │   └── {split}/
            │       ├── file_001_probs.npy
            │       ├── file_001_labels.npy
            │       └── ...
            ├── csv_bi/
            │   ├── reference/
            │   │   ├── file_001.csv_bi (ground truth)
            │   │   └── ...
            │   └── hypothesis/
            │       ├── file_001.csv_bi (predictions)
            │       └── ...
            ├── metrics/
            │   └── {split}_{algorithm}_metrics.json
            └── results_summary.md
        """
        pass

    def compare_experiments(
        self,
        baseline_results: EvaluationResults,
        experiment_results: EvaluationResults,
    ) -> Dict[str, float]:
        """
        Compare two experiment results (e.g., baseline vs Exp1).

        Args:
            baseline_results: Results from baseline experiment
            experiment_results: Results from comparison experiment

        Returns:
            Dict with comparison metrics:
            - sensitivity_improvement: Absolute % improvement
            - sensitivity_relative: Relative % improvement
            - overfitting_reduction: Dev-test gap reduction
            - f1_improvement: Absolute F1 improvement

        Example:
            baseline = evaluator1.evaluate_on_split("eval")
            exp1 = evaluator2.evaluate_on_split("eval")
            comparison = ModelEvaluator.compare_experiments(baseline, exp1)
            print(f"Sensitivity improved by {comparison['sensitivity_improvement']:.2f}%")
        """
        pass

    def generate_publication_table(
        self,
        results_list: List[EvaluationResults],
        literature_benchmarks: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate publication-ready markdown table.

        Args:
            results_list: List of evaluation results to include
            literature_benchmarks: Optional dict of literature results

        Returns:
            Markdown table string ready for papers/docs

        Example Output:
            ```
            | Model | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | F1 | Notes |
            |-------|---------------|----------------|--------------|----|----|
            | Baseline | 28.01% | 24.3% | 3.7% | 0.31 | Overfitting |
            | Exp1 | 26.5% | 25.8% | 0.7% | 0.33 | Better gen. |
            | Shah et al. [1] | - | 89% | - | - | SOTA |
            ```
        """
        pass
```

#### 3.2 Test Specifications for ModelEvaluator

**Test file**: `tests/integration/eval/test_evaluator.py`

```python
import pytest
from pathlib import Path
from src.brain_brr.eval.evaluator import ModelEvaluator, EvaluationResults

class TestModelEvaluator:
    """Integration tests for ModelEvaluator"""

    @pytest.fixture
    def sample_checkpoint(self, tmp_path):
        """
        Create minimal checkpoint file for testing.

        Note: This is a simplified checkpoint, real ones are ~189MB.
        """
        import torch
        from src.brain_brr.config.schemas import TrainingConfig

        # Create minimal config
        config = TrainingConfig.from_yaml("configs/local/smoke_fla.yaml")

        # Create checkpoint dict
        checkpoint = {
            "model_state_dict": {},  # Empty for testing
            "optimizer_state_dict": {},
            "epoch": 9,
            "best_metric": 0.2801,
            "config": config,
        }

        ckpt_path = tmp_path / "test_best.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def sample_test_data(self, tmp_path):
        """
        Create minimal test dataset.

        For real testing, this would point to actual TUSZ cache.
        """
        # Mock test data structure
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        # ... create minimal NPY cache files ...
        return test_dir

    # Test 1: Initialization
    def test_init_success(self, sample_checkpoint, tmp_path):
        """ModelEvaluator initializes with valid checkpoint"""
        output_dir = tmp_path / "eval_output"

        evaluator = ModelEvaluator(
            checkpoint_path=sample_checkpoint,
            output_dir=output_dir,
        )

        assert evaluator is not None
        assert evaluator.checkpoint_path == sample_checkpoint
        assert evaluator.output_dir == output_dir

    def test_init_checkpoint_not_found(self, tmp_path):
        """ModelEvaluator raises FileNotFoundError for missing checkpoint"""
        nonexistent_ckpt = tmp_path / "nonexistent.pt"
        output_dir = tmp_path / "eval_output"

        with pytest.raises(FileNotFoundError):
            ModelEvaluator(nonexistent_ckpt, output_dir)

    # Test 2: End-to-end evaluation (mocked)
    @pytest.mark.integration
    def test_evaluate_on_split_dev(
        self, sample_checkpoint, sample_test_data, tmp_path, monkeypatch
    ):
        """
        End-to-end evaluation on dev split.

        Note: This test mocks heavy operations (inference, scoring)
        to keep runtime reasonable. Full integration test requires
        real data and GPU.
        """
        output_dir = tmp_path / "eval_output"

        evaluator = ModelEvaluator(sample_checkpoint, output_dir)

        # Mock inference (skip actual model forward pass)
        def mock_inference(*args, **kwargs):
            # Return dummy predictions
            return {
                "file_001": {"probs": ..., "labels": ...},
            }
        monkeypatch.setattr(evaluator, "_run_inference", mock_inference)

        # Mock NEDC scoring (skip actual nedc-bench call)
        from src.brain_brr.eval.nedc_wrapper import NEDCMetrics
        def mock_score(*args, **kwargs):
            return NEDCMetrics(
                algorithm="overlap",
                sensitivity_at_10FA_24h=25.0,
                sensitivity_at_5FA_24h=22.0,
                sensitivity_at_1FA_24h=18.0,
                f1=0.30,
                precision=0.40,
                recall=0.25,
                tp=100,
                fp=150,
                fn=300,
                taes_score=None,
                total_seizure_duration_sec=500.0,
                total_recording_duration_sec=10000.0,
            )
        monkeypatch.setattr(evaluator.scorer, "score_predictions", mock_score)

        # Run evaluation
        results = evaluator.evaluate_on_split(split="dev", algorithm="overlap")

        assert isinstance(results, EvaluationResults)
        assert results.split == "dev"
        assert results.algorithm == "overlap"
        assert results.metrics.sensitivity_at_10FA_24h == 25.0
        assert results.checkpoint_epoch == 9

    # Test 3: Experiment comparison
    def test_compare_experiments(self, tmp_path):
        """Compare baseline vs experiment results"""
        from src.brain_brr.eval.nedc_wrapper import NEDCMetrics

        baseline_metrics = NEDCMetrics(
            algorithm="overlap",
            sensitivity_at_10FA_24h=24.3,
            sensitivity_at_5FA_24h=21.8,
            sensitivity_at_1FA_24h=15.2,
            f1=0.31,
            precision=0.42,
            recall=0.24,
            tp=145,
            fp=198,
            fn=456,
            taes_score=None,
            total_seizure_duration_sec=1000.0,
            total_recording_duration_sec=20000.0,
        )

        exp1_metrics = NEDCMetrics(
            algorithm="overlap",
            sensitivity_at_10FA_24h=27.8,
            sensitivity_at_5FA_24h=25.1,
            sensitivity_at_1FA_24h=19.3,
            f1=0.35,
            precision=0.45,
            recall=0.28,
            tp=168,
            fp=205,
            fn=433,
            taes_score=None,
            total_seizure_duration_sec=1000.0,
            total_recording_duration_sec=20000.0,
        )

        baseline_results = EvaluationResults(
            experiment_name="baseline",
            checkpoint_path="results/baseline/best.pt",
            checkpoint_epoch=9,
            split="eval",
            algorithm="overlap",
            metrics=baseline_metrics,
            comparison_to_dev={"dev_sens_10FA": 28.01, "gap": 3.71},
            timestamp="2025-10-19T12:00:00",
        )

        exp1_results = EvaluationResults(
            experiment_name="exp1_reg",
            checkpoint_path="results/exp1/best.pt",
            checkpoint_epoch=12,
            split="eval",
            algorithm="overlap",
            metrics=exp1_metrics,
            comparison_to_dev={"dev_sens_10FA": 29.2, "gap": 1.4},
            timestamp="2025-10-19T14:00:00",
        )

        comparison = ModelEvaluator.compare_experiments(baseline_results, exp1_results)

        # Exp1 improved sensitivity by 3.5% absolute
        assert abs(comparison["sensitivity_improvement"] - 3.5) < 0.1

        # Exp1 reduced overfitting gap from 3.71% to 1.4% (2.31% reduction)
        assert abs(comparison["overfitting_reduction"] - 2.31) < 0.1

    # Test 4: Publication table generation
    def test_generate_publication_table(self):
        """Generate markdown table for publication"""
        # ... create sample results list ...
        # ... call generate_publication_table() ...
        # ... verify markdown format ...
        pass
```

#### 3.3 Implementation Acceptance Criteria

**ModelEvaluator is complete when**:
- [ ] All integration tests pass (with mocking for speed)
- [ ] Full end-to-end test on real dev set passes (manual, with GPU)
- [ ] Can evaluate full dev set (1832 files) in < 30 minutes
- [ ] Generates publication-ready markdown tables
- [ ] JSON output is well-formatted and complete
- [ ] CLI interface works (`python -m src.brain_brr.eval.evaluator --help`)
- [ ] Documentation includes complete usage examples

---

### Data Requirements

#### Metadata Extraction

**Source**: TUSZ dataset file naming convention

**Format**: `{patient}_{session}_{token}.edf`

**Example**: `aaaaaaaa_s001_t000.edf`

**Metadata needed for CSV_BI**:
- `patient`: First part (e.g., "aaaaaaaa")
- `session`: Second part (e.g., "s001")
- `token`: Third part (e.g., "t000")
- `duration_sec`: Recording duration from EDF header
- `sampling_rate`: From config (256 Hz)

**Implementation**:
```python
def extract_metadata_from_filename(file_id: str, edf_path: Path) -> RecordingMetadata:
    """
    Extract metadata from TUSZ filename and EDF file.

    Args:
        file_id: e.g., "aaaaaaaa_s001_t000"
        edf_path: Path to .edf file for duration extraction

    Returns:
        RecordingMetadata with all required fields
    """
    parts = file_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid file_id format: {file_id}")

    patient, session, token = parts

    # Extract duration from EDF
    import pyedflib
    with pyedflib.EdfReader(str(edf_path)) as edf:
        duration_sec = edf.file_duration

    return RecordingMetadata(
        file_id=file_id,
        patient=patient,
        session=session,
        token=token,
        duration_sec=duration_sec,
        sampling_rate=256,
    )
```

#### Ground Truth CSV_BI Files

**Where are they?**

**Option 1: TUSZ provides CSV_BI ground truth**
- Check if TUSZ eval/test split includes .csv_bi label files
- Location: `{tusz_root}/edf/eval/*/label/*.csv_bi`

**Option 2: Convert from existing label format**
- TUSZ provides labels in different format (.lbl or .tse)
- Need to convert to CSV_BI using NEDC tools or custom converter

**Action Required**:
- [ ] Locate TUSZ eval/test set ground truth labels
- [ ] Verify format (CSV_BI or need conversion)
- [ ] Document location in this spec

---

### Error Handling Specifications

#### Component 1: CSVBIConverter

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| probs file not found | FileNotFoundError | "Probabilities file not found: {path}" | Skip file, log warning |
| Invalid probs shape | ValueError | "Probabilities must be 1D array, got shape {shape}" | Skip file, log error |
| Duration mismatch | ValueError | "Duration mismatch: {actual}s != {expected}s" | Skip file, log error |
| Cannot write output | IOError | "Failed to write CSV_BI file: {path}" | Skip file, log error |
| Missing metadata | KeyError | "No metadata found for {file_id}" | Skip file, log warning |

#### Component 2: NEDCScorer

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| nedc-bench not found | ImportError | "NEDC-BENCH not found at {path}. Clone from ..." | Fail fast |
| Reference dir not found | FileNotFoundError | "Reference directory not found: {path}" | Fail fast |
| No CSV_BI files | ValueError | "No .csv_bi files found in {path}" | Fail fast |
| File count mismatch | ValueError | "File count mismatch: {ref_count} ref vs {hyp_count} hyp" | Fail fast |
| NEDC scoring failed | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |
| Invalid CSV_BI format | RuntimeError | "Invalid CSV_BI format in {file}: {error}" | Skip file, log error |

#### Component 3: ModelEvaluator

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| Checkpoint not found | FileNotFoundError | "Checkpoint not found: {path}" | Fail fast |
| Checkpoint load failed | RuntimeError | "Failed to load checkpoint: {error}" | Fail fast |
| Test data not found | FileNotFoundError | "Test data not found for split {split}" | Fail fast |
| Inference failed | RuntimeError | "Inference failed on {file_id}: {error}" | Skip file, continue |
| Conversion failed | RuntimeError | "Conversion to CSV_BI failed: {error}" | Log error, continue |
| Scoring failed | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |

---

### Performance Requirements

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Convert 1 recording to CSV_BI | < 100ms | < 100MB | Per-recording overhead |
| Convert 1000 recordings | < 60s | < 500MB | Batch processing |
| Score 1000 file pairs (overlap) | < 30s | < 1GB | NEDC-BENCH overhead |
| Full eval on dev set (1832 files) | < 30min | < 24GB | Includes inference |
| Full eval on eval set (~2000 files) | < 40min | < 24GB | Includes inference |

---

### Logging Specifications

#### Log Levels

**DEBUG**: Detailed execution flow
```
DEBUG: Loading probs from /path/to/file_001_probs.npy
DEBUG: Converted 153 samples to 3 events (post-processing applied)
DEBUG: Writing CSV_BI to /path/to/file_001.csv_bi
```

**INFO**: High-level progress
```
INFO: [CSVBIConverter] Converting 1832 recordings...
INFO: [CSVBIConverter] Converted 1832/1832 files (100%)
INFO: [NEDCScorer] Scoring 1832 file pairs with 'overlap' algorithm...
INFO: [NEDCScorer] Sensitivity@10FA: 25.3%
```

**WARNING**: Non-fatal issues
```
WARNING: No metadata found for file_042, skipping
WARNING: Duration mismatch for file_135 (299.5s != 300.0s), skipping
```

**ERROR**: Failed operations
```
ERROR: Failed to convert file_256: Invalid probs shape (2, 76800)
ERROR: NEDC scoring failed for file_312: Invalid CSV_BI format
```

---

### CLI Interface Specification

**Command**: `python -m src.brain_brr.eval.evaluator`

**Usage**:
```bash
# Evaluate checkpoint on dev set
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/baseline/checkpoints/best.pt \
  --split dev \
  --algorithm overlap \
  --output results/eval_baseline_dev/

# Evaluate on eval/test set
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/baseline/checkpoints/best.pt \
  --split eval \
  --algorithm overlap \
  --output results/eval_baseline_test/

# Run all algorithms
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/baseline/checkpoints/best.pt \
  --split eval \
  --algorithm all \
  --output results/eval_baseline_test_all/

# Compare two experiments
python -m src.brain_brr.eval.evaluator compare \
  --baseline results/eval_baseline_test/metrics.json \
  --experiment results/eval_exp1_test/metrics.json \
  --output results/comparison_baseline_vs_exp1.md
```

**Arguments**:
```
Positional:
  {evaluate,compare}  Subcommand (default: evaluate)

Evaluate options:
  --checkpoint PATH   Path to .pt checkpoint file (required)
  --split {dev,eval}  Which data split to evaluate (required)
  --algorithm {overlap,taes,dp,epoch,ira,all}  NEDC algorithm (default: overlap)
  --output PATH       Output directory (default: results/eval_{timestamp}/)
  --save-predictions  Save .npy predictions (default: True)
  --save-csv-bi       Save .csv_bi files (default: True)
  --verbose           Enable debug logging

Compare options:
  --baseline PATH     Path to baseline metrics.json (required)
  --experiment PATH   Path to experiment metrics.json (required)
  --output PATH       Output markdown file (default: stdout)
```

---

### File Structure After Implementation

```
src/brain_brr/eval/
├── __init__.py
├── format_converter.py    # Component 1: CSVBIConverter
├── nedc_wrapper.py        # Component 2: NEDCScorer
├── evaluator.py           # Component 3: ModelEvaluator
└── README.md              # Usage documentation

tests/unit/eval/
├── test_format_converter.py  # 10 unit tests
└── test_nedc_wrapper.py       # 8 unit tests

tests/integration/eval/
└── test_evaluator.py          # 4 integration tests

results/eval_{experiment}/
├── predictions/
│   └── {split}/
│       ├── file_001_probs.npy
│       ├── file_001_labels.npy
│       └── ...
├── csv_bi/
│   ├── reference/
│   │   ├── file_001.csv_bi
│   │   └── ...
│   └── hypothesis/
│       ├── file_001.csv_bi
│       └── ...
├── metrics/
│   ├── dev_overlap_metrics.json
│   └── eval_overlap_metrics.json
└── results_summary.md
```

---

### Implementation Phases (Test-Driven Development)

#### Phase 1: CSVBIConverter (Week 1)

**Day 1-2: Write tests**
- [ ] Create `tests/unit/eval/test_format_converter.py`
- [ ] Write all 10 test cases (they will fail initially)
- [ ] Verify test infrastructure works

**Day 3-4: Implement CSVBIConverter**
- [ ] Create `src/brain_brr/eval/format_converter.py`
- [ ] Implement `CSVBIConverter.__init__()`
- [ ] Implement `convert_recording()` (reuse `batch_probs_to_events`)
- [ ] Implement `convert_directory()`
- [ ] Run tests, iterate until all pass

**Day 5: Integration & validation**
- [ ] Test on real dev set predictions (manual)
- [ ] Validate CSV_BI format with nedc-bench parser
- [ ] Fix any edge cases discovered
- [ ] Update documentation

#### Phase 2: NEDCScorer (Week 2)

**Day 1-2: Write tests**
- [ ] Create `tests/unit/eval/test_nedc_wrapper.py`
- [ ] Write all 8 test cases
- [ ] Create sample CSV_BI fixtures

**Day 3-4: Implement NEDCScorer**
- [ ] Create `src/brain_brr/eval/nedc_wrapper.py`
- [ ] Add `sys.path.insert()` for nedc-bench
- [ ] Implement `NEDCScorer.__init__()`
- [ ] Implement `score_predictions()`
- [ ] Implement `score_predictions_batch()`
- [ ] Implement `validate_csv_bi_format()`
- [ ] Run tests, iterate until all pass

**Day 5: Integration test**
- [ ] Run on nedc-bench sample data
- [ ] Verify parity with nedc-bench CLI
- [ ] Document all 5 algorithms

#### Phase 3: ModelEvaluator (Week 3)

**Day 1-2: Write tests**
- [ ] Create `tests/integration/eval/test_evaluator.py`
- [ ] Write integration tests (with mocking)
- [ ] Create sample checkpoint fixture

**Day 3-5: Implement ModelEvaluator**
- [ ] Create `src/brain_brr/eval/evaluator.py`
- [ ] Implement `ModelEvaluator.__init__()`
- [ ] Implement `evaluate_on_split()`
- [ ] Implement `compare_experiments()`
- [ ] Implement `generate_publication_table()`
- [ ] Add CLI interface (`if __name__ == "__main__":`)
- [ ] Run tests, iterate until all pass

**Day 6-7: End-to-end validation**
- [ ] Run full evaluation on dev set with real checkpoint
- [ ] Verify metrics match internal validation
- [ ] Test all CLI commands
- [ ] Update documentation

#### Phase 4: Production Evaluation (Week 4)

**Baseline Evaluation**
- [ ] Locate TUSZ eval/test set
- [ ] Convert ground truth labels to CSV_BI (if needed)
- [ ] Run baseline best.pt on eval set
- [ ] Get official NEDC metrics
- [ ] Compare to dev set (quantify overfitting)
- [ ] Document results

**Exp1 Evaluation** (when ready)
- [ ] Wait for Exp1 to reach epoch 9+
- [ ] Run Exp1 best checkpoint on eval set
- [ ] Compare to baseline
- [ ] Generate publication tables
- [ ] Update experiment tracking

---

### Success Criteria (Definition of Done)

**CSVBIConverter**:
- [ ] All unit tests pass
- [ ] Code coverage ≥ 95%
- [ ] Converts dev set in < 60s
- [ ] CSV_BI files parse correctly in nedc-bench
- [ ] Documentation complete

**NEDCScorer**:
- [ ] All unit + integration tests pass
- [ ] Code coverage ≥ 90%
- [ ] Scores 1000 pairs in < 30s
- [ ] All 5 algorithms work correctly
- [ ] Documentation complete

**ModelEvaluator**:
- [ ] All integration tests pass
- [ ] Full eval on dev set works
- [ ] CLI interface complete
- [ ] Generates publication tables
- [ ] Documentation complete

**End-to-End**:
- [ ] Baseline evaluated on eval/test set
- [ ] Official NEDC metrics obtained
- [ ] Results reproducible
- [ ] Publication-ready outputs
- [ ] Complete documentation

---

### Open Questions to Resolve

**Before starting implementation**:

1. **TUSZ eval/test set location**:
   - [ ] Where is the TUSZ eval/test split located?
   - [ ] Does it have ground truth .csv_bi files?
   - [ ] If not, how do we convert labels to CSV_BI?

2. **Metadata extraction**:
   - [ ] Do we have access to .edf files for duration extraction?
   - [ ] Can we cache metadata to avoid repeated EDF reads?
   - [ ] Should metadata be in cache manifest.json?

3. **Configuration**:
   - [ ] Should post-processing config be in main config or separate?
   - [ ] Should NEDC algorithm be configurable or hardcoded to "overlap"?
   - [ ] Should we support multiple algorithms simultaneously?

4. **Performance optimization**:
   - [ ] Can we parallelize CSV_BI conversion (multiprocessing)?
   - [ ] Should we cache CSV_BI files to avoid reconversion?
   - [ ] Can we optimize NEDC scoring (batch vs serial)?

**Answers needed before Phase 1 starts!**

---

### References

**NEDC-BENCH Documentation**:
- https://github.com/Clarity-Digital-Twin/nedc-bench
- https://github.com/Clarity-Digital-Twin/nedc-bench/blob/main/docs/algorithms/
- https://github.com/Clarity-Digital-Twin/nedc-bench/blob/main/docs/migration/data-formats.md

**TUSZ Dataset**:
- https://isip.piconepress.com/projects/tuh_eeg/
- File naming convention: `{patient}_{session}_{token}.edf`

**Existing Code**:
- `src/brain_brr/eval/metrics.py` (batch_probs_to_events function)
- `src/brain_brr/train/val_step.py` (validation loop, prediction saving)
- `src/brain_brr/config/schemas.py` (PostprocessingConfig)

---

## Notes

**Latest W&B screenshot**: `WB-Baseline-Training.png` (shows overfitting pattern clearly)

**Key insight from baseline**: The model CAN learn to detect seizures (28% is decent for first attempt), but it's overfitting the training data. Regularization and/or more training data likely needed.

**Patience required**: Exp1 won't show results for ~7 more epochs (need to reach epoch 9 for fair comparison). ETA: ~67 hours (2.8 days) from now.

**NEDC-BENCH**: Your maintained version at https://github.com/Clarity-Digital-Twin/nedc-bench provides official NEDC v6.0.0 scoring with modern Python API. This is THE standard for seizure detection evaluation in literature.

---

## Next Steps Summary (Priority Order) 📋

### Immediate Actions (While Exp1 Trains)

**1. Understand Current Data Formats** ✅ DONE!
- ✅ Found validation output code (`val_step.py`)
- ✅ Identified prediction format (`.npy` continuous timelines)
- ✅ Identified NEDC requirement (`.csv_bi` event lists)
- ✅ Imported nedc-bench to `reference_repos/nedc-bench/`

**2. Locate TUSZ Eval/Test Set** 🔍
- [ ] Find TUSZ eval/test data location
- [ ] Verify ground truth labels exist (CSV_BI format)
- [ ] Check if preprocessing needed (like train/dev)

**3. Build Evaluation Pipeline** 🔧

**Phase 1: Format Converter** (Build First!)
- [ ] Create `src/brain_brr/eval/format_converter.py`
- [ ] Implement `CSVBIConverter` class (reuse `batch_probs_to_events()`)
- [ ] Test conversion: `.npy` → `.csv_bi`
- [ ] Verify CSV_BI format correctness

**Phase 2: NEDC Wrapper** (Build Second!)
- [ ] Create `src/brain_brr/eval/nedc_wrapper.py`
- [ ] Add `sys.path.insert()` for nedc-bench import
- [ ] Implement `NEDCScorer` class (direct Python API)
- [ ] Test scoring with sample CSV_BI files

**Phase 3: End-to-End Evaluator** (Build Third!)
- [ ] Create `src/brain_brr/eval/evaluator.py`
- [ ] Implement `ModelEvaluator` class
- [ ] Wire up: load checkpoint → run inference → convert → score
- [ ] Add CLI interface (`python -m src.brain_brr.eval.evaluator`)

**Phase 4: Baseline Evaluation** (Do This Now!)
- [ ] Enable `save_predictions: true` in config (temporarily)
- [ ] Run inference on eval/test set with `best.pt`
- [ ] Convert predictions to CSV_BI
- [ ] Score with NEDC-BENCH
- [ ] Document official metrics
- [ ] Compare to dev set results (quantify overfitting)

### Medium-Term Actions

**5. Exp1 Comparison** (When Exp1 reaches epoch 9+)
- [ ] Run Exp1 best checkpoint on test set
- [ ] Score with NEDC-BENCH
- [ ] Compare: Baseline vs Exp1 (both dev AND test)
- [ ] Prove regularization improved generalization
- [ ] Update experiment tracking table

**6. Literature Comparison** (After baseline + Exp1 results)
- [ ] Create comparison table (our results vs papers)
- [ ] Identify performance gaps
- [ ] Plan next experiments (Exp2, Exp3, etc.)

### Long-Term Actions

**7. Next Experiments** (Based on baseline + Exp1 results)
- Exp2: Different regularization balance?
- Exp3: Data augmentation?
- Exp4: Architecture changes?
- Exp5: Ensemble methods?

**8. Publication Preparation**
- Write methods section (architecture + training)
- Create results tables (NEDC metrics)
- Generate figures (learning curves, predictions)
- Compare to SOTA (literature benchmarks)

---

## Key Decision Points 🎯

### Should we evaluate baseline despite overfitting?
**YES!** (See reasoning in "Should We Run Eval/Test Despite Overfitting?" section)
- Quantifies overfitting impact objectively
- Establishes baseline for all future work
- Required for scientific rigor
- Tests evaluation pipeline

### What evaluation approach should we use?
**Direct Python import of nedc-bench!** (See "TL;DR" section)
- NO Docker complexity
- NO data transfer overhead
- Fast, clean, simple
- Official NEDC v6.0.0 metrics

### When should we run baseline evaluation?
**NOW! While Exp1 trains!** (~67 hours until Exp1 reaches epoch 9)
- Build evaluation pipeline
- Run baseline inference on test set
- Get official metrics
- Have baseline numbers ready for Exp1 comparison

---

## Success Criteria ✅

**Evaluation Pipeline Success**:
- [ ] Can convert `.npy` predictions to `.csv_bi` format
- [ ] Can call nedc-bench via direct Python import
- [ ] Get official NEDC metrics (sensitivity@FA, TAES, F1, etc.)
- [ ] Results match internal validation metrics (sanity check)

**Baseline Evaluation Success**:
- [ ] Test set performance within expected range (23-27% sens@10FA)
- [ ] Dev-test gap quantified (measures overfitting)
- [ ] Beats random baseline (~8%)
- [ ] Proves architecture viability

**Exp1 Comparison Success**:
- [ ] Exp1 test performance ≥ baseline test performance
- [ ] Exp1 dev-test gap < baseline dev-test gap (better generalization)
- [ ] Proves regularization effectiveness

---

## What This Enables 🚀

**Once evaluation pipeline is built**:
- ✅ Official NEDC v6.0.0 metrics for ALL experiments
- ✅ Direct comparison to literature (apples-to-apples)
- ✅ Publication-ready results
- ✅ Clinical utility assessment (FA/24h is what clinicians care about!)
- ✅ Rapid experimentation (eval any checkpoint in minutes)
- ✅ Proof of generalization (test set validates dev set results)

**The complete story you'll tell**:
```
Table 1: Performance on TUSZ Eval Set (NEDC v6.0.0 Overlap Scoring)

Model              | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | Analysis
-------------------|---------------|----------------|--------------|------------------
Baseline (overfit) | 28.01%        | 24.3%          | 3.7%         | Overfitting observed
Exp1 (stronger reg)| 26.5%         | 25.8%          | 0.7%         | Better generalization!
Literature [Shah]  | -             | 89%            | -            | SOTA benchmark
Target (Clinical)  | -             | >75%           | -            | <1 FA/24h needed
```

**This is the path to publication!** 📄🎯
