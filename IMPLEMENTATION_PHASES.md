# IMPLEMENTATION_PHASES.md - Logical Build Order for Brain-Go-Brr v2

## 🎯 Mission
Build the world's first Bi-Mamba-2 + U-Net + ResCNN architecture for O(N) clinical EEG seizure detection.

## 📋 Prerequisites Check
- [ ] TUH EEG Seizure Corpus access
- [ ] CHB-MIT dataset access
- [ ] GPU with CUDA (A100 preferred)
- [ ] All reference repos cloned
- [ ] Dependencies installed (`uv sync -E gpu`)

## 🚀 PHASE 1: DATA PIPELINE (Week 1)

### 1.1 EDF Reader & Preprocessor
**File:** `src/experiment/data.py`
**Depends on:** MNE, scipy
```python
# Core functions to implement:
- load_edf_file(path) -> np.ndarray
- validate_channel_order(data, expected_channels)
- preprocess_recording(data, fs) -> processed_data
```

### 1.2 Window Extraction
**File:** `src/experiment/data.py`
```python
# Core functions:
- extract_windows(data, window_size=60, stride=10)
- create_window_dataset(edf_paths, labels)
```

### 1.3 Data Validation
**Deliverable:** Script that loads and validates 10 sample EDFs
**Success Criteria:**
- Loads TUH/CHB-MIT files
- Outputs shape (B, 19, 15360)
- Normalized correctly

## 🏗️ PHASE 2: MODEL COMPONENTS (Week 1-2)

### 2.1 U-Net Encoder/Decoder
**File:** `src/experiment/models.py`
**Reference:** SeizureTransformer `architecture.py:120-212`
```python
class UNetEncoder(nn.Module):
    # 4 stages, [64, 128, 256, 512] channels
    # Skip connections preserved

class UNetDecoder(nn.Module):
    # Mirror of encoder with skip fusion
```

### 2.2 ResCNN Stack
**File:** `src/experiment/models.py`
**Reference:** SeizureTransformer `architecture.py:215-260`
```python
class ResCNNBlock(nn.Module):
    # Kernel sizes [3, 5, 7]
    # Spatial dropout
```

### 2.3 Bidirectional Mamba-2
**File:** `src/experiment/models.py`
**Import:** `from mamba_ssm import Mamba2`
**Reference:** Vim patterns for bidirectional
```python
class BiMamba2(nn.Module):
    # Forward + backward Mamba2
    # State concatenation
```

### 2.4 Full Model Assembly
**File:** `src/experiment/models.py`
```python
class SeizureDetectorV2(nn.Module):
    def __init__(self):
        self.encoder = UNetEncoder()
        self.rescnn = ResCNNStack()
        self.mamba = BiMamba2()
        self.decoder = UNetDecoder()
        self.head = nn.Conv1d(64, 1, 1)
```

### 2.5 Model Validation
**Deliverable:** Test forward pass
**Success Criteria:**
- Input: (B, 19, 15360)
- Output: (B, 15360) probabilities

## 🔄 PHASE 3: TRAINING PIPELINE (Week 2)

### 3.1 Loss Functions
**File:** `src/experiment/losses.py`
```python
- BCE with boundary tolerance
- Dice loss component
- Combined weighted loss
```

### 3.2 Training Loop
**File:** `src/experiment/pipeline.py`
```python
- Balanced sampling (50% seizure, 50% background)
- Hard negative mining
- AMP (Automatic Mixed Precision)
- Gradient clipping
```

### 3.3 NEDC Scoring Integration
**Import:** `from nedc_bench.algorithms import taes`
```python
- Real-time TAES during validation
- FA/24h tracking
```

### 3.4 Initial Training Run
**Deliverable:** 1-epoch test on subset
**Success Criteria:**
- Loss decreases
- No NaN/inf
- GPU memory stable

## 🎯 PHASE 4: POST-PROCESSING (Week 2-3)

### 4.1 Hysteresis Thresholding
**File:** `src/experiment/postprocess.py`
**Import:** `from skimage.filters import apply_hysteresis_threshold`
```python
- τ_on = 0.86, τ_off = 0.78
- Apply to probability outputs
```

### 4.2 Morphological Operations
**Import:** `from scipy.ndimage import binary_opening, binary_closing`
```python
- Opening/closing with kernel=5
- Minimum duration filter (≥3s)
```

### 4.3 Window Stitching
**File:** `src/experiment/postprocess.py`
```python
- Overlap averaging for 50s regions
- Reconstruct full timeline
```

## 🏆 PHASE 5: EVALUATION (Week 3)

### 5.1 CSV_BI Export
**Import:** `from nedc_bench.models import annotations`
```python
- Format predictions as CSV_BI
- Match Temple evaluation format
```

### 5.2 Full Training
**Target:** 50-100 epochs on full TUH
**Hardware:** A100 GPU (2-3 days)
**Monitoring:**
- Wandb/Tensorboard logging
- Checkpoint best dev performance

### 5.3 Final Evaluation
**Datasets:**
- Dev: CHB-MIT
- Test: epilepsybenchmarks.com
**Metrics:**
- TAES @ 10/5/1 FA/24h
- AUROC (sanity check)

## 📊 SUCCESS CRITERIA

### Phase Gates (must pass to proceed):
1. **Data:** Can load and preprocess any TUH/CHB file
2. **Model:** Forward pass works, reasonable memory usage
3. **Training:** Loss converges, dev metrics improve
4. **Post:** Hysteresis improves raw predictions
5. **Eval:** Achieves >75% sens @ 10 FA/24h

### Target Performance:
- 10 FA/24h: >95% sensitivity ✅
- 5 FA/24h: >90% sensitivity ✅
- 1 FA/24h: >75% sensitivity ✅

## 🚨 RISK MITIGATION

### Common Issues & Solutions:
1. **OOM on long recordings:** Implement sliding window inference
2. **Training instability:** Reduce LR, increase gradient clipping
3. **Poor initial performance:** Verify preprocessing matches papers
4. **Slow training:** Enable AMP, reduce validation frequency

## 📝 DOCUMENTATION REQUIREMENTS

Each phase completion requires:
1. Working code with type hints
2. Unit tests (pytest markers)
3. `make q` passes (lint + format + mypy)
4. Brief results summary

---
**Next Step:** Start Phase 1.1 - Implement EDF reader with MNE