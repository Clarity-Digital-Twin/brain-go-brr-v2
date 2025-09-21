# SEIZURE DETECTION MVP: FINAL ARCHITECTURE SPECIFICATION

## Executive Summary

**Architecture:** U-Net (1D CNN) → ResCNN stack → Bi-Mamba-2 → U-Net decoder → sigmoid → Hysteresis → TAES

**Key Innovation:** Replace SeizureTransformer's attention mechanism with Bi-Mamba-2 for O(N) complexity on long recordings while maintaining state-of-the-art performance.

**Target Metric:** TAES sensitivity @ 10 FA/24h (primary), with reporting at 5, 2.5, 1 FA/24h

---

## 1. PREPROCESSING PIPELINE (SSOT)

### Input Requirements
- **Format:** Raw EDF files with multi-channel EEG data
- **Montage:** 19-channel 10-20 referential (fixed channel order - MUST assert)
- **Duration:** Variable length recordings (minutes to 24+ hours)

### Processing Steps
1. **Channel Selection:** Extract exactly 19 channels in predetermined order
2. **Resampling:** Downsample/upsample to 256 Hz
3. **Filtering:**
   - Bandpass: 0.5-120 Hz (3rd-order Butterworth)
   - Notch: 60 Hz (default) or 50 Hz where applicable (remove power line noise). Assert mains frequency ∈ {50, 60} per dataset.
4. **Normalization:** Per-channel z-score computed over full recording. If full-record statistics are impractical in-memory, compute via a streaming method (e.g., Welford’s algorithm) for numerically stable single-pass mean/variance.
5. **Metadata:** Preserve `start_sample` timestamps for stitching

### Window Extraction
- **Window size:** 60 seconds (15,360 samples @ 256 Hz)
- **Stride:** 10 seconds (50-second overlap between consecutive windows)
- **Output shape:** `(B, 19, 15360)` where B = batch size

---

## 2. MODEL ARCHITECTURE

### Component Overview
```
Input (B,19,15360) → U-Net Encoder → ResCNN → Bi-Mamba-2 → U-Net Decoder → Output (B,15360)
```

### 2.1 U-Net Encoder (Multi-scale Feature Extraction)
**Purpose:** Extract hierarchical features at multiple temporal scales

- **Stages:** 4 downsampling blocks
- **Downsampling factor:** ×2 per stage (total ×16)
- **Channel progression:** [64, 128, 256, 512]
- **Block structure:**
  - Double Conv1d per stage (kernel=5, padding=2)
  - BatchNorm1d
  - ReLU activation
- **Skip connections:** Save features AFTER block, BEFORE downsample

### 2.2 ResCNN Stack (Local Pattern Enhancement)
**Purpose:** Refine local temporal patterns before global modeling

- **Location:** Bottleneck (after encoder, before Bi-Mamba-2)
- **Structure:** 3 residual blocks
- **Kernel sizes:** [3, 5, 7] (multi-scale local receptive fields)
- **Width:** 512 channels maintained
- **Dropout:** 0.1
- **Shape preserved:** `(B, 512, 960)` where 960 = 15360/16

### 2.3 Bi-Mamba-2 (Temporal Context Modeling)
**Purpose:** Capture long-range temporal dependencies with linear complexity

- **Layers:** 6 bidirectional Mamba blocks
- **Model dimensions:**
  - d_model: 512
  - d_state: 16
  - conv_kernel: 5
- **Bidirectional processing:**
  1. Forward pass: process sequence left-to-right
  2. Backward pass: process sequence right-to-left
  3. Concatenate: merge to 1024 channels
  4. Project: 1×1 Conv back to 512 channels
- **Dropout:** 0.1
- **Output shape:** `(B, 512, 960)`
  - **Residual:** Add a residual connection from the pre-Mamba bottleneck features to the projected output.

### 2.4 U-Net Decoder (Multi-scale Reconstruction)
**Purpose:** Upsample to original temporal resolution with skip connections

- **Stages:** 4 upsampling blocks
- **Upsampling:** ConvTranspose1d (kernel=2, stride=2)
- **Skip fusion:**
  1. Concatenate encoder skip with upsampled features
  2. Double Conv1d (kernel=5, padding=2)
  3. BatchNorm1d → ReLU
- **Channel progression:** [512, 256, 128, 64] → 1

### 2.5 Output Head
- **Final layer:** Conv1d(64→1, kernel=1)
- **Activation:** Sigmoid
- **Output:** Per-timestep probabilities @ 256 Hz
- **Shape:** `(B, 15360)` probability values in [0,1]

---

## 3. POST-PROCESSING PIPELINE

### 3.1 Window Stitching
- **Method:** Uniform overlap-average over 50-second overlapping regions. For each time step, average predictions from all windows covering that time step. Edges are handled naturally by averaging over the available contributing windows only.
- **Result:** Continuous probability stream for entire recording

### 3.2 Hysteresis Thresholding
**Purpose:** Reduce false alarm chatter with dual thresholds

- **τ_on:** 0.86 (threshold to START seizure detection)
- **τ_off:** 0.78 (threshold to STOP seizure detection)
- **Algorithm:**
  ```python
  in_event = False
  for t, prob in enumerate(probabilities):
      if not in_event and prob >= 0.86:
          in_event = True
      if in_event:
          mask[t] = True
          if prob < 0.78:
              in_event = False
  ```
  Note: Hysteresis alone stops immediately upon crossing τ_off; the subsequent morphological closing (Section 3.3) can fill brief sub-threshold dips. We still apply opening first, then closing.

### 3.3 Morphological Operations
- **Opening:** Remove isolated spikes (kernel_size=11 ≈ 43 ms @ 256 Hz)
- **Closing:** Fill small gaps (kernel_size=31 ≈ 121 ms @ 256 Hz)
- **Order:** Open first, then close

All morphology kernel sizes are specified in samples unless otherwise stated.

### 3.4 Duration Filtering
- **Minimum duration:** 3.0 seconds
- **Action:** Remove all events shorter than threshold

### 3.5 Event Extraction
- **Input:** Binary mask
- **Output:** List of (start_time, end_time) tuples in seconds

---

## 4. TRAINING CONFIGURATION

### Loss Function
- **Components:** Binary cross-entropy with element-wise weighting
- **Class weights:** Computed from dataset statistics

### Data Sampling Strategy
- **Balance:** 50% seizure windows, 50% background
- **Hard negative mining:** Include top-K false positives mined on the dev set from the previous epoch; maintain 50/50 per batch via reweighting or controlled sampling.
- **Augmentation:** None (maintain signal integrity)

### Optimization
- **Optimizer:** AdamW
- **Learning rate:** 3e-4
- **Weight decay:** 0.05
- **Schedule:** Cosine annealing with 10% warmup
- **Gradient clipping:** 1.0
- **Mixed precision:** AMP enabled
- **Batch size:** 16

### Early Stopping
- **Metric:** Development set sensitivity @ 10 FA/24h
- **Patience:** 10 epochs
- **Checkpoint:** Save best model based on metric

---

## 5. EVALUATION METRICS

### Primary Metrics (TAES Framework)
- **Sensitivity @ 10 FA/24h** (primary target)
- **Sensitivity @ 5 FA/24h**
- **Sensitivity @ 2.5 FA/24h**
- **Sensitivity @ 1 FA/24h**

### Secondary Metrics
- **AUROC:** Overall discrimination capability
- **Event-based metrics:** Precision, recall, F1

---

## 6. IMPLEMENTATION CHECKLIST

### Phase 1: Infrastructure
- [ ] Implement SSOT preprocessing pipeline with unit tests
- [ ] Verify channel ordering and sampling rate assertions
- [ ] Test filtering and normalization correctness

### Phase 2: Model Components
- [ ] Implement U-Net encoder/decoder with skip connections
- [ ] Add ResCNN stack at bottleneck
- [ ] Integrate Bi-Mamba-2 layers
- [ ] Verify shape flow: `(B,19,15360) → (B,512,960) → (B,15360)`

### Phase 3: Post-processing
- [ ] Implement overlap-averaging stitcher
- [ ] Code hysteresis thresholding state machine
- [ ] Add morphological operations
- [ ] Create duration filter

### Phase 4: Training Pipeline
- [ ] Implement balanced sampler with hard negative mining
- [ ] Set up loss function with element-wise weighting
- [ ] Configure optimizer and training loop
- [ ] Add early stopping on dev sensitivity

### Phase 5: Evaluation
- [ ] Implement TAES scorer
- [ ] Verify parity with NEDC reference implementation
- [ ] Generate FA/24h curves
- [ ] Create evaluation reports

### Phase 6: Validation
- [ ] Determinism test (±0.1% variance across runs)
- [ ] Memory/latency profiling on 24-hour recordings
- [ ] False positive audit and categorization
- [ ] Final performance benchmarking

---

## 7. KEY PARAMETERS (LOCKED)

### Model Parameters
- **Input channels:** 19
- **Sampling rate:** 256 Hz
- **Window duration:** 60 seconds
- **Stride:** 10 seconds
- **Bottleneck compression:** ×16
- **Total parameters:** ~20-30M

### Post-processing Parameters
- **τ_on:** 0.86
- **τ_off:** 0.78
- **Morphology:** opening_kernel=11, closing_kernel=31 (samples)
- **Minimum duration:** 3.0 seconds

### Training Parameters
- **Learning rate:** 3e-4
- **Batch size:** 16
- **Gradient clip:** 1.0
- **Early stop metric:** Sensitivity @ 10 FA/24h

---

## 8. ARCHITECTURAL JUSTIFICATION

### Why This Stack?
1. **U-Net:** Proven in SeizureTransformer, EventNet - handles multi-scale morphology
2. **ResCNN:** SeizureTransformer's local pattern enhancement at bottleneck
3. **Bi-Mamba-2:** O(N) complexity enables 24-hour recordings (vs O(N²) Transformers)
4. **Hysteresis:** Reduces FA/24h by preventing threshold chatter

### Biological Alignment
- **U-Net encoder:** Detects spikes/sharp waves (20-70 Hz)
- **ResCNN:** Captures rhythmic buildup (3-30 Hz)
- **Bi-Mamba-2:** Models seizure state evolution and propagation
- **Hysteresis:** Mimics seizure threshold dynamics

### What We Exclude (and Why)
- **XGBoost/Classical ML:** Window-based, poor TAES alignment
- **Feature engineering:** Breaks end-to-end training
- **Ensembles:** Complicate calibration
- **Pure Transformers:** O(N²) complexity fails on long recordings
- **Gap merging:** Only for SzCORE comparison, not NEDC evaluation

---

## 9. PIPELINE VISUALIZATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEIZURE DETECTION MVP PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

RAW EDF ──▶ PREPROCESSING ──▶ WINDOWING ──▶ MODEL ──▶ STITCHING ──▶ POST-PROC ──▶ EVENTS

PREPROCESSING:                    MODEL ARCHITECTURE:
• 19-ch montage                   ┌─────────────────────────────────┐
• 256 Hz resample                 │ Input (B, 19, 15360)            │
• 0.5-120 Hz + notch              └────────────┬────────────────────┘
• Per-channel z-score                          ▼
                                  ┌─────────────────────────────────┐
WINDOWING:                        │ U-Net Encoder (↓16x)            │
• 60s windows                     │ • 4 stages                      │
• 10s stride                      │ • Skip connections saved        │
• (B, 19, 15360)                  └────────────┬────────────────────┘
                                               ▼
POST-PROCESSING:                  ┌─────────────────────────────────┐
• Hysteresis (0.86/0.78)          │ ResCNN Stack                    │
• Morphology (open/close)         │ • 3 blocks, k=[3,5,7]           │
• Min duration (3.0s)             │ • Bottleneck: (B, 512, 960)     │
                                  └────────────┬────────────────────┘
EVALUATION:                                    ▼
• TAES @ 10/5/2.5/1 FA/24h        ┌─────────────────────────────────┐
• AUROC (secondary)               │ Bi-Mamba-2 (6 layers)           │
                                  │ • Bidirectional                 │
                                  │ • Linear O(N) complexity        │
                                  └────────────┬────────────────────┘
                                               ▼
                                  ┌─────────────────────────────────┐
                                  │ U-Net Decoder (↑16x)            │
                                  │ • Skip fusion                   │
                                  │ • Output: (B, 15360)            │
                                  └─────────────────────────────────┘
```

---

## 10. DEFINITION OF DONE

All of the following must be satisfied to ship:

### Functional Requirements
- Architecture matches specification exactly
- Preprocessing SSOT enforced with assertions
- Model produces per-timestep probabilities @ 256 Hz
- Post-processing runs globally after stitching
- TAES evaluation implemented and verified

### Performance Requirements
- Sensitivity @ 10 FA/24h meets target
- O(N) scaling verified on 24-hour recordings
- Inference fits in 8-12GB VRAM
- Deterministic results (±0.1% variance)

### Quality Requirements
- Unit tests pass for all components
- NEDC scorer parity achieved
- False positive audit completed
- Documentation complete and accurate

---

## APPENDIX: Reference Papers

1. **SeizureTransformer (Wu et al., 2025):** U-Net + ResCNN architecture reference
2. **FEMBA/Bi-Mamba:** Bidirectional Mamba design and parameters
3. **NEDC/TAES (Picone/Shah):** Evaluation metrics and scoring methodology

---

**This document represents the complete, final specification. No variations or alternatives. Ship it.**
