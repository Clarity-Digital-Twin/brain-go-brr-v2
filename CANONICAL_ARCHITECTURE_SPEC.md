# CANONICAL ARCHITECTURE SPECIFICATION
## Brain-Go-Brr v2: First Bi-Mamba-2 + U-Net + ResCNN for Clinical EEG Seizure Detection

This document serves as the single source of truth for the complete architecture specification, consolidating all design decisions and implementation requirements. Use this checklist to audit the codebase and ensure faithful implementation of the original vision.

---

## 🏗️ COMPLETE ARCHITECTURE CHECKLIST

### 1. DATA PIPELINE (Phase 1)
**Purpose**: Standardized EEG data loading, preprocessing, and windowing

#### 1.1 Input Specifications
- [ ] **File Format**: EDF/EDF+ support via MNE
  - Location: `src/experiment/data.py::load_edf_file()`
  - [ ] Handles malformed headers (TUSZ date separator fix)
  - [ ] Fallback header repair on temp copy if MNE fails

- [ ] **Channels**: 19-channel 10-20 montage in canonical order
  - Location: `src/experiment/constants.py::CHANNEL_NAMES_10_20`
  - Order: `["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]`
  - [ ] Channel synonym mapping (T7→T3, T8→T4, P7→T5, P8→T6)
  - [ ] Missing channel interpolation for Fz, Pz (automatic via MNE)
  - [ ] Fixed channel ordering with `pick_and_order_by_index()` utility

#### 1.2 Preprocessing Pipeline
- [ ] **Resampling**: Target 256 Hz
  - Location: `src/experiment/data.py::preprocess_recording()`
  - [ ] Uses `scipy.signal.resample()` for Phase 1 baseline

- [ ] **Filtering**:
  - [ ] Bandpass: 0.5-120 Hz (Butterworth order=3)
  - [ ] Notch: 60 Hz (US) or 50 Hz (EU) powerline
  - [ ] Uses `lfilter` (not `filtfilt`) for consistency

- [ ] **Normalization**: Per-channel z-score
  - [ ] Computed over full recording (not per-window)
  - [ ] NaN/Inf replaced with 0

#### 1.3 Window Extraction
- [ ] **Window Parameters**:
  - Location: `src/experiment/constants.py`
  - [ ] Size: 60 seconds (15,360 samples @ 256 Hz)
  - [ ] Stride: 10 seconds (2,560 samples)
  - [ ] Overlap: 50 seconds (83.3%)

- [ ] **Output Shape**: `(B, 19, 15360)` where B = batch size
  - Location: `src/experiment/data.py::extract_windows()`
  - [ ] Float32 dtype
  - [ ] Window metadata tracking (start_samples)

#### 1.4 Dataset & Caching
- [ ] **PyTorch Dataset**: `EEGWindowDataset`
  - Location: `src/experiment/data.py`
  - [ ] In-memory baseline (Phase 1)
  - [ ] NPZ cache support
  - [ ] File ID and timeline tracking for reconstruction

---

### 2. MODEL ARCHITECTURE (Phase 2)
**Purpose**: Novel Bi-Mamba-2 + U-Net + ResCNN for O(N) seizure detection

#### 2.1 U-Net Encoder
- Location: `src/experiment/models.py::UNetEncoder`

- [ ] **Structure**: 4 stages with progressive downsampling
  - [ ] Channel progression: [64, 128, 256, 512]
  - [ ] Downsample factor: ×2 per stage (total ×16)
  - [ ] Final bottleneck: (B, 512, 960)

- [ ] **Blocks**:
  - [ ] Initial projection: 19→64 channels (kernel=7)
  - [ ] Double convolution per stage (kernel=5, padding=2)
  - [ ] Skip connections saved AFTER block, BEFORE downsample
  - [ ] Skip shapes: [(64,15360), (128,7680), (256,3840), (512,1920)]

#### 2.2 ResCNN Stack
- Location: `src/experiment/models.py::ResCNNStack`

- [ ] **Multi-scale Feature Extraction**:
  - [ ] 3 ResidualCNN blocks
  - [ ] Multi-kernel branches: [3, 5, 7]
  - [ ] Channel split: [170, 170, 172] for 512 total
  - [ ] Residual connections per block

- [ ] **Shape Preservation**:
  - [ ] Input: (B, 512, 960)
  - [ ] Output: (B, 512, 960)
  - [ ] Spatial dropout: 0.1

#### 2.3 Bidirectional Mamba-2
- Location: `src/experiment/models.py::BiMamba2`

- [ ] **SSM Configuration**:
  - [ ] 6 bidirectional layers
  - [ ] d_model: 512
  - [ ] d_state: 16
  - [ ] d_conv: 5 (temporal conv kernel)
  - [ ] Expand factor: 2

- [ ] **Bidirectional Processing**:
  - [ ] Forward Mamba-2 branch
  - [ ] Backward Mamba-2 branch (flipped sequence)
  - [ ] Concatenate → Project (1024→512)
  - [ ] LayerNorm + Residual per layer

- [ ] **Fallback**: Conv1d for CPU testing
  - [ ] Automatic detection via MAMBA_AVAILABLE flag
  - [ ] Warning issued when using fallback
  - [ ] Shape-compatible but NOT functionally equivalent

#### 2.4 U-Net Decoder
- Location: `src/experiment/models.py::UNetDecoder`

- [ ] **Structure**: 4 stages with progressive upsampling
  - [ ] Channel progression: [512, 256, 128, 64]
  - [ ] Upsample factor: ×2 per stage (total ×16)
  - [ ] Skip fusion at each stage (concatenation)

- [ ] **Skip Connection Order**:
  - [ ] Stage 0 uses skip[3] (deepest)
  - [ ] Stage 1 uses skip[2]
  - [ ] Stage 2 uses skip[1]
  - [ ] Stage 3 uses skip[0] (shallowest)

- [ ] **Output**: (B, 19, 15360) - recovers input dimensions

#### 2.5 Detection Head
- Location: `src/experiment/models.py::SeizureDetector`

- [ ] **Final Layers**:
  - [ ] Conv1d: 19→1 channel (kernel=1)
  - [ ] Sigmoid activation
  - [ ] Output: (B, 15360) probabilities in [0, 1]

#### 2.6 Complete Model Assembly
- [ ] **SeizureDetector** class combines all components
- [ ] Parameter count: ~25M expected
- [ ] Weight initialization: Xavier/He
- [ ] `from_config()` method for schema compatibility

---

### 3. TRAINING PIPELINE (Phase 3)
**Purpose**: Robust training with clinical metrics and reproducibility

#### 3.1 Data Loading
- Location: `src/experiment/pipeline.py`

- [ ] **Balanced Sampling**:
  - [ ] WeightedRandomSampler at dataset level
  - [ ] 50% seizure, 50% background windows
  - [ ] pos_weight computed from training set statistics

- [ ] **DataLoader Config**:
  - [ ] Batch size from config (default 16)
  - [ ] num_workers from config
  - [ ] pin_memory=True when CUDA
  - [ ] Deterministic seeding

#### 3.2 Loss & Optimization
- [ ] **Loss Function**: Binary Cross-Entropy
  - [ ] Element-wise weighting for class imbalance
  - [ ] pos_weight = (1 - pos_ratio) / pos_ratio
  - [ ] Applied per-timestep over 15,360 samples

- [ ] **Optimizer**: AdamW
  - [ ] Learning rate: 3e-4 (from config)
  - [ ] Weight decay from config

- [ ] **Scheduler**: Cosine with warmup
  - [ ] Warmup ratio from config
  - [ ] Step per iteration (not epoch)

- [ ] **Regularization**:
  - [ ] Gradient clipping (global norm)
  - [ ] Mixed precision (AMP) when CUDA
  - [ ] Dropout: 0.1 throughout model

#### 3.3 Training Loop
- Location: `src/experiment/pipeline.py::train_epoch()`

- [ ] **Per Epoch**:
  - [ ] Forward pass with AMP autocast
  - [ ] Backward with gradient scaling
  - [ ] Optimizer step with clipping
  - [ ] Scheduler step per batch
  - [ ] Validation at epoch end

- [ ] **Monitoring**:
  - [ ] Train/val loss logging
  - [ ] Learning rate tracking
  - [ ] Gradient norms (optional)

#### 3.4 Validation & Metrics
- Location: `src/experiment/evaluate.py`

- [ ] **Clinical Metrics**:
  - [ ] TAES (Time-Aligned Event Scoring)
  - [ ] Sensitivity @ {10, 5, 2.5, 1} FA/24h
  - [ ] AUROC (sample-level)
  - [ ] FA curve generation

- [ ] **Early Stopping**:
  - [ ] Metric: sensitivity_at_10fa (default)
  - [ ] Patience from config
  - [ ] Best model checkpointing

---

### 4. POST-PROCESSING (Phase 4)
**Purpose**: Convert probabilities to clinical events

#### 4.1 Hysteresis Thresholding
- Location: `src/experiment/postprocess.py::apply_hysteresis()`

- [ ] **Dual-Tau System**:
  - [ ] τ_on: 0.86 (onset threshold)
  - [ ] τ_off: 0.78 (offset threshold)
  - [ ] Stability windows: min_onset=128, min_offset=256 samples

#### 4.2 Morphological Operations
- Location: `src/experiment/postprocess.py::apply_morphology()`

- [ ] **Sequence**: Opening → Closing
  - [ ] Opening kernel: 11 samples (~43ms)
  - [ ] Closing kernel: 31 samples (~121ms)
  - [ ] CPU: SciPy ndimage
  - [ ] GPU: MaxPool1d-based (optional)

#### 4.3 Duration Filtering
- [ ] **Constraints**:
  - [ ] Minimum: 3.0 seconds
  - [ ] Maximum: 600.0 seconds
  - [ ] Long events segmented into chunks

#### 4.4 Window Stitching
- Location: `src/experiment/postprocess.py::stitch_windows()`

- [ ] **Methods**:
  - [ ] overlap_add (uniform averaging)
  - [ ] overlap_add_weighted (triangular)
  - [ ] max (element-wise maximum)

#### 4.5 Event Generation
- [ ] **Event Merging**: tau_merge = 2.0s
- [ ] **Confidence Scoring**: mean/peak/percentile
- [ ] **Output Format**: SeizureEvent(start_s, end_s, confidence)

---

### 5. EVALUATION (Phase 5)
**Purpose**: Clinical evaluation and benchmarking

#### 5.1 Metrics Implementation
- Location: `src/experiment/evaluate.py`

- [ ] **TAES Calculation**:
  - [ ] Overlap-weighted scoring
  - [ ] False alarm penalty (α=0.15)
  - [ ] Output range: [0, 1]

- [ ] **FA/24h Computation**:
  - [ ] Event-level false alarms
  - [ ] Normalized by recording duration
  - [ ] Binary search for threshold selection

- [ ] **Sensitivity at FA Rates**:
  - [ ] Targets: {10, 5, 2.5, 1} FA/24h
  - [ ] Event-level overlap detection
  - [ ] Conservative threshold selection

#### 5.2 Export Formats
- Location: `src/experiment/export.py`

- [ ] **CSV_BI (Temple-compliant)**:
  - [ ] Header: version, bname, duration, montage
  - [ ] Columns: channel, start_time, stop_time, label, confidence
  - [ ] TERM channel for whole-record events

- [ ] **JSON Metrics**:
  - [ ] Complete metrics dictionary
  - [ ] Threshold table
  - [ ] Configuration hash

---

### 6. INFRASTRUCTURE & TOOLS

#### 6.1 Configuration System
- Location: `src/experiment/schemas.py`

- [ ] **Pydantic Models**:
  - [ ] ModelConfig (encoder, mamba, rescnn, decoder)
  - [ ] TrainingConfig (optimizer, scheduler, early_stopping)
  - [ ] DataConfig (paths, batch_size, num_workers)
  - [ ] PostprocessingConfig (hysteresis, morphology, duration)
  - [ ] ExperimentConfig (root config)

- [ ] **YAML Configs**:
  - [ ] configs/local.yaml (development)
  - [ ] configs/production.yaml (full training)
  - [ ] configs/smoke_test.yaml (CI testing)

#### 6.2 CLI Interface
- Location: `src/cli.py`

- [ ] **Commands**:
  - [ ] train: Full training pipeline
  - [ ] evaluate: Run evaluation on checkpoint
  - [ ] validate: Validate config files
  - [ ] info: Show environment info

#### 6.3 Testing Suite
- [ ] **Unit Tests**: All core functions
  - Location: `tests/test_*.py`
  - [ ] Data pipeline tests
  - [ ] Model component tests
  - [ ] Evaluation metric tests
  - [ ] Post-processing tests

- [ ] **Integration Tests**:
  - [ ] End-to-end training smoke test
  - [ ] Full evaluation pipeline test

- [ ] **Coverage**: Target >90% for core modules

#### 6.4 Development Tools
- [ ] **Makefile Commands**:
  - [ ] `make q`: Quality check (lint+format+type)
  - [ ] `make t`: Fast tests
  - [ ] `make train-local`: Local training
  - [ ] `make setup`: Initial setup

- [ ] **Pre-commit Hooks**:
  - [ ] Ruff formatting
  - [ ] Ruff linting
  - [ ] Type checking (mypy strict)

---

## 📊 PERFORMANCE TARGETS

### Clinical Metrics (TAES)
- [ ] 10 FA/24h: >95% sensitivity (current SOTA: ~90%)
- [ ] 5 FA/24h: >90% sensitivity (current SOTA: ~85%)
- [ ] 1 FA/24h: >75% sensitivity (current SOTA: ~70%)

### Model Performance
- [ ] Parameters: ~25M (actual: varies by config)
- [ ] Inference: <100ms per 60s window (GPU)
- [ ] Memory: <4GB for batch size 32
- [ ] Training: Convergence within 50 epochs

### Technical Specifications
- [ ] Sampling rate: 256 Hz (fixed)
- [ ] Window: 60s with 10s stride
- [ ] Channels: 19 (10-20 montage)
- [ ] Complexity: O(N) sequence modeling

---

## 🔍 VERIFICATION CHECKLIST

### Code Organization
- [ ] All model components in `src/experiment/models.py`
- [ ] Data pipeline in `src/experiment/data.py`
- [ ] Training in `src/experiment/pipeline.py`
- [ ] Evaluation in `src/experiment/evaluate.py`
- [ ] Post-processing in `src/experiment/postprocess.py`
- [ ] Configuration in `src/experiment/schemas.py`

### Dependencies
- [ ] PyTorch ≥2.5.0
- [ ] MNE ≥1.5.0
- [ ] mamba-ssm (GPU extra)
- [ ] scikit-image (post extra)
- [ ] pandas (eval extra)

### Critical Invariants
- [ ] Channel order ALWAYS: Fp1→F3→...→O2 (19 channels)
- [ ] Sampling rate ALWAYS: 256 Hz
- [ ] Window size ALWAYS: 60s (15,360 samples)
- [ ] Output ALWAYS: Per-timestep probabilities [0,1]
- [ ] Hysteresis ALWAYS: τ_on > τ_off

---

## ⚠️ KNOWN ISSUES & DEVIATIONS

1. **Mamba Conv Kernel**: CUDA supports only {2,3,4}, we coerce 5→4 internally
2. **Channel Interpolation**: Automatic for Fz, Pz via MNE
3. **CPU Fallback**: Conv1d replacement for Mamba (NOT equivalent)
4. **Header Fixes**: TUSZ date separator repair implemented

---

## ✅ AUDIT STATUS

**Last Audit Date**: [TO BE FILLED]
**Auditor**: [TO BE FILLED]

### Summary
- [ ] All core components implemented
- [ ] All tests passing (151+ tests)
- [ ] Documentation complete
- [ ] Performance targets met
- [ ] Ready for production

### Notes
[Add any deviations, concerns, or observations during audit]

---

**Mission**: Shock the world with O(N) clinical seizure detection 🚀