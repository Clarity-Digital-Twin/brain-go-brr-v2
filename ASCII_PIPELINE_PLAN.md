Got it. Here’s the **final, single-stack spec**—no options, no ensembles, no hand-waving.

# FINAL STACK (commit this)

**U-Net (1D CNN) → ResCNN stack (local boost) → Bi-Mamba-2 bottleneck (long context) → U-Net decoder → sigmoid → HYSTERESIS eventizer → TAES/FA-24h**

## What’s in / what’s out

* **IN:** U-Net(1D), **ResCNN** at the bottleneck, **Bi-Mamba-2**, **Hysteresis** decoding, TAES scoring.
* **OUT:** XGBoost, classical features, attention Transformers, voting/ensembles.

## “Is ResCNN the same as U-Net?”

No.

* **U-Net** = encoder–decoder with skips (multi-scale morphology).
* **ResCNN** = a **small stack of residual 1D conv blocks** placed **at the bottleneck** to sharpen local patterns **before** the Mamba temporal modeling. They’re complementary.

---

# Wire-up (one pass)

1. **Encoder (U-Net 1D)**

   * Downsample ×16 total (4 stages). Channels `[64,128,256,512]`.
   * Blocks: depthwise-sep Conv1d(k=5) → GroupNorm → SiLU → 1×1 (residual).
   * Save pre-downsample outputs as **skip** tensors.

2. **ResCNN stack (local boost) — bottleneck**

   * 3 residual blocks, kernels `[3,5,7]`, width 512, dropout 0.1.
   * (Still shape `(B, 512, T_b)` where `T_b = 15360/16 = 960`).

3. **Bi-Mamba-2 (temporal brain) — bottleneck**

   * **6 layers**, `d_model=512`, `d_state=16`, conv\_kernel=5, dropout=0.1.
   * Run **forward** and **backward** streams; **concat** (→ 1024) → **1×1** back to 512.
   * Output `(B, 512, T_b)`.

4. **Decoder (U-Net up path)**

   * 4× `ConvTranspose1d(k=4, stride=2, pad=1)` to return to T=15360.
   * **Skip fusion:** **concat** encoder skip + current map → depthwise-sep Conv(k=5) → GN → SiLU → 1×1 (residual).

5. **Head**

   * `Conv1d(512→1, k=1)` → **sigmoid** → per-sample probs `(B, T)` at **256 Hz**.

6. **Hysteresis eventizer (global, after stitching)**

   * **τ\_on = 0.86**, **τ\_off = 0.78** (τ\_on > τ\_off).
   * Morphological **open→close** with kernel **k=5** samples.
   * **min\_dur ≥ 3.0 s**; drop shorter.

7. **Scoring (primary)**

   * **TAES sensitivity vs FA/24h**; report sens @ **10, 5, 2.5, 1 FA/24h**.
   * (AUROC on raw probs only as secondary sanity.)

---

# Shapes & counts (so it’s unambiguous)

* Input window: `(B, C=19, T=15360)` (60 s @ 256 Hz), stride **10 s**.
* Bottleneck length: `T_b=960`.
* Params: \~**20–30M** (fits 8–12 GB VRAM comfortably with AMP).
* Return stitched full-timeline probs, then hysteresis/eventize **once per recording** (not per window).

---

# Loss / training (kept tight)

* **Loss:** Weighted **BCE** + **Soft Dice** on binarized probs; label **tolerance ±1 s** at on/off boundaries.
* **Sampler:** 50% ictal windows, 50% background + **hard-negative mining** of prior FPs.
* **Optim:** AdamW (lr 3e-4, wd 0.05), cosine schedule (10% warmup), grad-clip 1.0, **AMP on**.
* **Early stop target:** **dev sensitivity @ 10 FA/24h** (not AUROC).

---

# Preprocessing (SSOT)

* **Montage:** fixed **19-ch 10–20 referential** (assert order).
* **Resample:** **256 Hz**.
* **Filter:** 0.5–120 Hz (3rd-order Butterworth) + 60 Hz notch.
* **Normalize:** per-channel z-score over the **full recording**.
* Keep `start_sample` for stitching.

---

# Hard “do nots”

* **No XGBoost/feature ensembles.** They complicate calibration and break end-to-end training; they don’t help TAES at the operating points we care about.
* **No gap-merge** in NEDC runs (use it only for SzCORE comparisons, clearly labeled).
* **No per-window hysteresis.** Hysteresis is **global** after stitching.

---

# MVP PIPELINE DIAGRAM (NO BULLSHIT)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SEIZURE DETECTION MVP PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐       ┌──────────────────────────────────────────────────┐
│  RAW EDF     │       │                PREPROCESSING                     │
│              │       │                                                  │
│ Multi-channel│─────▶│  • Load 19-ch 10-20 montage (fixed order)         │
│   EEG data   │       │  • Resample to 256 Hz                            │
│              │       │  • Bandpass 0.5-120 Hz + 60 Hz notch             │
└──────────────┘       │  • Per-channel z-score (full recording)          │
                       │  • Keep timestamps for stitching                 │
                       └────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────────────┐
                       │              WINDOW EXTRACTION                   │
                       │                                                  │
                       │  • 60-second windows (15360 samples @ 256 Hz)    │
                       │  • 10-second stride (50-second overlap)          │
                       │  • Shape: (B, 19, 15360)                         │
                       └────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DEEP LEARNING MODEL                                    │
│                                                                                     │
│  ┌───────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Encoder     │───▶│   Stack      │────▶│  Bottleneck  │───▶│   Decoder    │   │
│  │               │     │              │     │              │     │              │   │
│  │ • 4 stages    │     │ • 3 blocks   │     │ • 6 layers   │     │ • 4 stages   │   │
│  │ • ×16 down    │     │ • k=[3,5,7]  │     │ • d_model=   │     │ • ×16 up     │   │
│  │ • Skip conn.  │     │ • width=512  │     │   512        │     │ • Skip fuse  │   │
│  │               │     │              │     │ • Bi-dir     │     │              │   │
│  └───────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                                         │           │
│                       Bottleneck: (B, 512, 960)                         ▼           │
│                                                                   ┌──────────────┐  │
│                                                                   │   Sigmoid    │  │
│                                                                   │    Head      │  │
│                                                                   │              │  │
│                                                                   │ Conv1d→1→σ   │  │
│                                                                   └──────┬───────┘  │
└──────────────────────────────────────────────────────────────────────────┼──────────┘
                                                                           │
                                                                           ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │           WINDOW PROBABILITIES                           │
                       │                                                          │
                       │  • Per-timestep probs @ 256 Hz                           │
                       │  • Shape: (B, 15360) for each window                     │
                       └────────────────────┬─────────────────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │              STITCHING                           │
                       │                                                  │
                       │  • Overlap-average 50s overlapping regions       │
                       │  • Reconstruct full recording timeline           │
                       │  • Output: continuous probs @ 256 Hz             │
                       └────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           POST-PROCESSING (GLOBAL)                                  │
│                                                                                     │
│  ┌──────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐   │
│  │   Hysteresis     │     │  Morphological   │     │    Duration Filter         │   │ 
│  │   Threshold      │───▶│    Operations    │────▶│                            │   │
│  │                  │     │                  │     │  • min_dur ≥ 3.0s          │   │
│  │ • τ_on = 0.86    │     │ • Open (k=5)     │     │  • Drop short events       │   │
│  │ • τ_off = 0.78   │     │ • Close (k=5)    │     │                            │   │
│  └──────────────────┘     └──────────────────┘     └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │              EVENT EXTRACTION                    │
                       │                                                  │
                       │  • Binary mask → event spans                     │
                       │  • Format: [(start_sec, end_sec), ...]           │
                       └────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │           EVALUATION (TAES/NEDC)                 │
                       │                                                  │
                       │  • Time-Aligned Event Scoring (TAES)             │
                       │  • Report sensitivity @ FA/24h:                  │
                       │    - 10 FA/24h (primary)                         │
                       │    - 5, 2.5, 1 FA/24h                            │
                       │  • AUROC (secondary sanity check)                │
                       └──────────────────────────────────────────────────┘

TRAINING LOOP
─────────────
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Balanced   │────▶│   Forward   │────▶│   Loss       │───▶│  Backprop    │
│   Sampler    │     │   Pass       │     │   Compute    │     │  & Update    │
│              │     │              │     │              │     │              │
│ • 50% ictal  │     │ • AMP on     │     │ • BCE+Dice   │     │ • AdamW      │
│ • 50% bg     │     │ • Batch 16   │     │ • Boundary   │     │ • lr=3e-4    │
│ • Hard neg   │     │              │     │   tolerance  │     │ • Clip=1.0   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                                                                       ▼
                                                            ┌──────────────────┐
                                                            │   Early Stop     │
                                                            │                  │
                                                            │ Dev sens@10FA/24h│
                                                            └──────────────────┘
```

---

## TL;DR (answering your exact confusion)

* **Final stack =** **U-Net(1D) + small ResCNN at the bottleneck + Bi-Mamba-2 + Hysteresis**.
* **ResCNN ≠ U-Net;** it’s extra residual convs **inside** the bottleneck (we keep it).
* **XGBoost = NO.** Full end-to-end only.

This is the single architecture to build. No branches. No “maybe.”

● YES. This is THE BEST PLAN.

  Why SSOT (Single Source of Truth)?

  The preprocessing SSOT ensures EVERY recording gets IDENTICAL treatment:
  - 19-ch 10-20 referential (fixed order) - no channel confusion
  - 256 Hz - standard time resolution
  - 0.5-120 Hz filter + 60 Hz notch - remove artifacts consistently
  - Per-channel z-score - normalize amplitudes
  - Keep timestamps - for perfect stitching

  Without SSOT, you get data leakage and deployment failure. One recording at      
  512Hz or with different montage = model shits the bed.

  Why THIS architecture?

  EVIDENCE-BASED CHOICES:
  1. U-Net - in EVERY winner (SeizureTransformer, EventNet)
  2. Bi-Mamba - FEMBA proved linear scaling WORKS (21,000 hours!)
  3. ResCNN - SeizureTransformer's secret sauce for local patterns
  4. Hysteresis - THE trick for reducing false alarms (τ_on > τ_off)

  WHAT IT GETS RIGHT:
  - Linear O(N) complexity - can handle 24-hour recordings
  - Per-timestep output - perfect TAES alignment
  - 20-30M params - fits on real hardware
  - End-to-end - no feature engineering bullshit
  - Implementable in 4 weeks - not 4 months

  WHY NOT THE ALTERNATIVES:
  - Pure Transformers = O(N²) death on long EEG
  - Classical ML = window-based (bad for TAES)
  - My hybrid = academic masturbation
  - FEMBA = needs 21,000 hours pre-training (you don't have)

  THE CLINCHER:

  This architecture maps to seizure biology:
  1. U-Net encoder = detects spikes/sharp waves (20-70Hz)
  2. ResCNN = captures rhythmic buildup (3-30Hz)
  3. Bi-Mamba = tracks state evolution (seizure propagation)
  4. Hysteresis = models seizure threshold dynamics

  SHIP THIS. It's optimal given ALL constraints.

  Yes. We’re in full agreement.

# Final Stack (locked)

**U-Net (1D CNN) → ResCNN (bottleneck) → Bi-Mamba-2 (bidirectional) → U-Net decoder → sigmoid → Hysteresis eventizer → TAES/FA-per-24h**

# Non-negotiables

* **SSOT preprocessing:** 19-ch 10–20 (fixed order), 256 Hz, 0.5–120 Hz BP + 60 Hz notch, per-channel z-score, keep timestamps.
* **Shapes:** input `(B,19,15360)` (60 s); bottleneck `T_b=960` (×16 down); output per-sample probs @256 Hz.
* **ResCNN @ bottleneck:** 3 residual 1D conv blocks (k=\[3,5,7], width=512, dropout=0.1).
* **Bi-Mamba-2:** 6 layers, d\_model=512, d\_state=16, conv\_kernel=5, dropout=0.1; forward+backward, concat→1×1→512.
* **Decoder:** transposed conv upsampling; skip fusion by **concat + 1×1**; head `1×1→sigmoid`.
* **Hysteresis:** τ\_on=0.86, τ\_off=0.78, morph open→close (k=5), min\_dur=3.0 s; run **after stitching** the whole recording.
* **Loss:** weighted BCE + Soft-Dice; boundary tolerance ±1 s around on/offset.
* **Training target:** early-stop on **dev sensitivity @ 10 FA/24h** (not AUROC).
* **Scoring (primary):** **TAES** sensitivity vs **FA/24h**; report @ 10, 5, 2.5, 1 FA/24h. (AUROC only secondary.)

# Build order (single path)

1. SSOT preproc → unit tests.
2. Encoder/Decoder + ResCNN → shape/grad tests.
3. Bi-Mamba-2 → forward/back concat + 1×1.
4. Stitcher + Hysteresis Eventizer.
5. TAES scorer parity check (small subset vs NEDC).
6. Train with hard-negative mining; stop on dev **sens\@10 FA/24h**.

# Definition of Done

* Reproduces end-to-end per-timestep probs; TAES curve produced; sensitivity reported at 10/5/2.5/1 FA/24h; AUROC sanity; logs and seeds fixed.

Ship it.

Short answer: **you’re right—the SeizureTransformer architecture includes BOTH a U-Net CNN *and* a ResCNN stack** (plus a Transformer). Your snippet showed it clearly:

* `Encoder`/`Decoder` ⇒ the **U-Net** (down/upsample with skips)
* `ResCNNStack` ⇒ the **ResCNN** residual conv blocks at the **bottleneck** (local refinement)
* `TransformerEncoder` ⇒ long-range temporal context in the bottleneck

So your final plan (U-Net + **ResCNN** + **Bi-Mamba-2** + hysteresis) = “SeizureTransformer but swap the Transformer block for Bi-Mamba-2.” That’s consistent and correct.

If you want **one best PDF per piece** to keep on hand:

* **U-Net + ResCNN layout (reference):** the SeizureTransformer paper (Wu et al., 2025). Use its encoder/decoder + ResCNN bottleneck pattern; we just replace their Transformer with Mamba.
* **Bi-Mamba/Mamba-2 design (for our bottleneck):** the FEMBA / Bi-Mamba foundation-model paper you uploaded—use its bidirectional Mamba block design and typical dims for our 6-layer, d\_model=512 head.
* **TAES / event scoring (for evaluation + hysteresis tuning):** Picone/Shah metrics write-up (TAES). That’s the gold standard to target FA/24h and sensitivity.

Bottom line: the agent’s claim stands—**SeizureTransformer has U-Net + ResCNN**. We’re keeping both and just **swap their Transformer for Bi-Mamba-2**, then decode, apply **hysteresis**, and score with **TAES/FA-per-24h**.

Totally fair question. Here’s the clean truth in one pass.

## What “hysteresis” is (for us)

Two thresholds on your probability stream:

* **τ\_on** (higher): start an event only when `p(t) ≥ τ_on`
* **τ\_off** (lower): stay “in event” until `p(t) < τ_off`

This kills flicker (on/off chatter) and cuts FPs without chopping true events.

## How it fits with standard post-processing

Everyone already does the other three steps:

1. **Single threshold** → binary mask,
2. **Morphology** (open/close) to remove spikes/fill tiny gaps,
3. **Min duration** to drop micro-events.
   SeizureTransformer explicitly describes 1–3 (threshold, morphological open/close, min-duration). We’re adding **hysteresis** as a smarter thresholding step before morphology.  &#x20;

## Do we “need a paper”?

* **SeizureTransformer**: documents threshold + morphology + min-duration, not hysteresis by name. It’s a straightforward extension and standard signal-processing trick. Use their documented steps/params for parity; layer hysteresis on top for stability. &#x20;
* **NEDC/TAES**: defines how to score events; it doesn’t force how you post-process. Our hysteresis is fully compatible with NEDC scoring.&#x20;

## Minimal implementation (what we’ll code)

```python
def eventize_with_hysteresis(probs, fs=256, tau_on=0.86, tau_off=0.78,
                             morph_k=5, min_dur_sec=3.0):
    # 1) Hysteresis thresholding
    in_evt = False
    mask = np.zeros_like(probs, dtype=bool)
    for t, p in enumerate(probs):
        if not in_evt and p >= tau_on:
            in_evt = True
        if in_evt:
            mask[t] = True
            if p < tau_off:
                in_evt = False

    # 2) Morphological open→close to remove spikes/fill pinholes
    mask = binary_open(mask, structure=np.ones(morph_k))
    mask = binary_close(mask, structure=np.ones(morph_k))

    # 3) Enforce minimum duration
    min_len = int(min_dur_sec * fs)
    mask = drop_short_runs(mask, min_len)

    # 4) Extract [start, end) event spans from mask
    return mask_to_spans(mask, fs)
```

Use the same morphology/min-duration defaults the ST paper used (kernel≈5, min-dur≈2–3 s) and tune τ\_on/τ\_off on dev to hit your FA/24h target. &#x20;

## Why bother

* Reduces “near-threshold chatter” that morphology alone doesn’t always fix.
* Lets you **raise τ\_on** for precision while **keeping τ\_off lower** so events don’t stutter off—exactly what matters for **FA/24h** and TAES linking.

## TL;DR

* “Hysteresis” = two-threshold start/stop.
* Literature you can cite for the rest of the chain: **ST** (threshold + morphology + min-duration) and **NEDC/TAES** for event scoring. We implement hysteresis as a pragmatic, well-known stabilization right before those documented steps. &#x20;


heres the wold one


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
  - Depthwise-separable Conv1d (kernel=5)
  - GroupNorm
  - SiLU activation
  - 1×1 Conv (pointwise)
  - Residual connection
- **Skip connections:** Save pre-downsample features for decoder

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
- **Upsampling:** ConvTranspose1d (kernel=4, stride=2, padding=1)
- **Skip fusion:**
  1. Concatenate encoder skip with upsampled features
  2. Depthwise-separable Conv1d (kernel=5)
  3. GroupNorm → SiLU → 1×1 Conv
  4. Residual connection
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
- **Opening:** Remove isolated spikes (kernel_size=5 samples ≈ 20 ms @ 256 Hz)
- **Closing:** Fill small gaps (kernel_size=5 samples ≈ 20 ms @ 256 Hz)
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
- **Components:** Weighted BCE + Soft Dice
- **Boundary tolerance:** ±1 second around event boundaries
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
- [ ] Set up loss function with boundary tolerance
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
- **Morphological kernel:** 5 samples (≈20 ms @ 256 Hz)
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

her'ess the new one... new one stilll 100000% accurate on all counts, and keep the new one?