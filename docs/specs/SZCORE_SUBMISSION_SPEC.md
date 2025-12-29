# SzCORE/epilepsybenchmarks.com Submission Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Status:** DRAFT - AWAITING SENIOR REVIEW
**Author:** Claude Code (automated research)

---

## Executive Summary

This spec documents the requirements for submitting our Brain-Go-Brr seizure detection model to [epilepsybenchmarks.com](https://epilepsybenchmarks.com/) (SzCORE benchmark).

### Critical Finding: Channel Order Mismatch

**Our model CANNOT be directly submitted without retraining or inference-time remapping.**

| Position | SzCORE Order | Our Training Order | Match? |
|----------|--------------|-------------------|--------|
| 1 | Fp1-Avg | Fp1 | ✅ |
| 2 | F3-Avg | F3 | ✅ |
| 3 | C3-Avg | C3 | ✅ |
| 4 | P3-Avg | P3 | ✅ |
| 5 | **O1-Avg** | **F7** | ❌ |
| 6 | **F7-Avg** | **T3** | ❌ |
| 7 | **T3-Avg** | **T5** | ❌ |
| 8 | **T5-Avg** | **O1** | ❌ |
| 9 | Fz-Avg | Fz | ✅ |
| 10 | Cz-Avg | Cz | ✅ |
| 11 | Pz-Avg | Pz | ✅ |
| 12 | Fp2-Avg | Fp2 | ✅ |
| 13 | F4-Avg | F4 | ✅ |
| 14 | C4-Avg | C4 | ✅ |
| 15 | P4-Avg | P4 | ✅ |
| 16 | **O2-Avg** | **F8** | ❌ |
| 17 | **F8-Avg** | **T4** | ❌ |
| 18 | **T4-Avg** | **T6** | ❌ |
| 19 | **T6-Avg** | **O2** | ❌ |

**Impact:** 8 out of 19 channels are in wrong positions. Our model learned spatial relationships (TCN convolutions, GNN adjacency) based on our order. Using SzCORE's order without remapping would cause the model to interpret occipital activity as temporal, etc.

---

## Table of Contents

1. [SzCORE Platform Overview](#1-szcore-platform-overview)
2. [Submission Requirements](#2-submission-requirements)
3. [Channel Mapping Strategy](#3-channel-mapping-strategy)
4. [Docker Image Specification](#4-docker-image-specification)
5. [Input Format: EDF Files](#5-input-format-edf-files)
6. [Output Format: TSV Files](#6-output-format-tsv-files)
7. [YAML Submission File](#7-yaml-submission-file)
8. [Scoring Methodology](#8-scoring-methodology)
9. [Implementation Plan](#9-implementation-plan)
10. [Risk Analysis](#10-risk-analysis)
11. [References](#11-references)

---

## 1. SzCORE Platform Overview

SzCORE (Seizure Community Open-source Research Evaluation) is an open-source benchmarking platform hosted at [epilepsybenchmarks.com](https://epilepsybenchmarks.com/).

### Key Features

- **Submission via PR**: Submit a YAML file to [github.com/esl-epfl/szcore](https://github.com/esl-epfl/szcore)
- **Automated Evaluation**: CI/CD runs your Docker container on test datasets
- **Multiple Datasets**: TUH Sz, Siena, SeizeIT1, Dianalund, CHB-MIT
- **Leaderboard**: Public benchmark results at epilepsybenchmarks.com

### 2025 Challenge (COMPLETED)

- **Timeline**: Dec 2024 - Feb 16, 2025 (we missed this)
- **Evaluation Metric**: Event-based F1 score
- **Prize**: $10,000 ($7K first, $3K second)
- **Status**: Results announced March 2025

We can still submit to the **permanent benchmark** (non-challenge).

---

## 2. Submission Requirements

### 2.1 Docker Image Requirements

| Requirement | Specification |
|------------|---------------|
| **Registry** | Public (Docker Hub, GHCR, etc.) |
| **Volumes** | `/data` (read-only), `/output` (read-write) |
| **Environment** | `INPUT` (EDF path), `OUTPUT` (TSV path) |
| **Entrypoint** | `CMD python3 -m algorithm "/data/$INPUT" "/output/$OUTPUT"` |
| **Network** | OFFLINE (no internet access during inference) |
| **Resources** | 10 CPU cores, 40GB RAM, 1 V100 GPU, 15min max per 1hr EEG |

### 2.2 YAML Submission File

Required fields:
```yaml
title: "Brain-Go-Brr: TCN+SSM+GNN Seizure Detection"
short_title: "BrainGoBrr"  # Max 20 chars
image: "ghcr.io/clarity-digital-twin/brain-go-brr:v4.3.0"
version: "4.3.0"
date: "2025-12-28"
authors:
  - given_name: "Author"
    family_name: "Name"
    institution: "Institution"
    email: "email@example.com"  # Optional
license: "Apache-2.0"  # Or MIT, GPL-3.0, etc.
repository: "https://github.com/Clarity-Digital-Twin/brain-go-brr-v2"
abstract: |
  # Algorithm overview
  [Description of architecture]

  # Input
  19-channel 10-20 EEG at 256Hz, common average reference

  # Training data
  TUH EEG Seizure Corpus v2.0.3 (train split)

  # Preprocessing
  [Description]

  # Performance analysis
  [Validation metrics]

  # Complexity
  31M parameters, ~5s inference per 1hr EEG on GPU
datasets:
  - "tuh_sz"
```

---

## 3. Channel Mapping Strategy

### 3.1 The Problem

SzCORE provides EDF files with this **exact channel order**:
```
Fp1-Avg, F3-Avg, C3-Avg, P3-Avg, O1-Avg, F7-Avg, T3-Avg, T5-Avg,
Fz-Avg, Cz-Avg, Pz-Avg,
Fp2-Avg, F4-Avg, C4-Avg, P4-Avg, O2-Avg, F8-Avg, T4-Avg, T6-Avg
```

Our model was trained with:
```
Fp1, F3, C3, P3, F7, T3, T5, O1,
Fz, Cz, Pz,
Fp2, F4, C4, P4, F8, T4, T6, O2
```

### 3.2 Solution Options

#### Option A: Inference-Time Remapping (RECOMMENDED)

**Pros:**
- No retraining required
- Uses existing best checkpoint (epoch 63)
- Minimal code changes

**Cons:**
- Slight inference overhead (negligible)
- Must verify remapping is correct

**Implementation:**
```python
# SzCORE channel indices -> Our training indices
SZCORE_TO_OURS = {
    0: 0,   # Fp1 -> Fp1
    1: 1,   # F3 -> F3
    2: 2,   # C3 -> C3
    3: 3,   # P3 -> P3
    4: 7,   # O1 -> O1 (was at index 7 in our order)
    5: 4,   # F7 -> F7 (was at index 4)
    6: 5,   # T3 -> T3 (was at index 5)
    7: 6,   # T5 -> T5 (was at index 6)
    8: 8,   # Fz -> Fz
    9: 9,   # Cz -> Cz
    10: 10, # Pz -> Pz
    11: 11, # Fp2 -> Fp2
    12: 12, # F4 -> F4
    13: 13, # C4 -> C4
    14: 14, # P4 -> P4
    15: 18, # O2 -> O2 (was at index 18)
    16: 15, # F8 -> F8 (was at index 15)
    17: 16, # T4 -> T4 (was at index 16)
    18: 17, # T6 -> T6 (was at index 17)
}

def remap_szcore_to_ours(eeg_data: np.ndarray) -> np.ndarray:
    """Remap SzCORE channel order to our training order.

    Args:
        eeg_data: (19, T) array in SzCORE channel order

    Returns:
        (19, T) array in our training channel order
    """
    remapped = np.zeros_like(eeg_data)
    for szcore_idx, our_idx in SZCORE_TO_OURS.items():
        remapped[our_idx] = eeg_data[szcore_idx]
    return remapped
```

#### Option B: Retrain with SzCORE Order

**Pros:**
- Native compatibility
- No inference-time conversion

**Cons:**
- ~40 days retraining on RTX 4090
- May not match current best performance
- Significant compute cost

**NOT RECOMMENDED** for initial submission.

#### Option C: Submit Both Orders

Train a second model with SzCORE order for future submissions, but use Option A for immediate submission.

### 3.3 Channel Name Handling

SzCORE uses `-Avg` suffix for common average reference. We need to strip this:

```python
def normalize_szcore_channel(name: str) -> str:
    """Convert SzCORE channel name to our canonical format.

    'Fp1-Avg' -> 'Fp1'
    """
    return name.replace("-Avg", "").strip()
```

---

## 4. Docker Image Specification

### 4.1 Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# SzCORE environment variables
ENV INPUT=""
ENV OUTPUT=""

# Create non-root user (security best practice)
RUN useradd -m -u 10001 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy model code and checkpoint
COPY src/ ./src/
COPY configs/ ./configs/
COPY checkpoints/best.pt ./checkpoints/

# Copy inference entrypoint
COPY docker/szcore_inference.py .

# Create volume mount points
VOLUME ["/data", "/output"]

# Switch to non-root user
USER appuser

# Entrypoint for SzCORE
CMD ["python3", "-m", "szcore_inference", "/data/${INPUT}", "/output/${OUTPUT}"]
```

### 4.2 Inference Entrypoint (`docker/szcore_inference.py`)

```python
#!/usr/bin/env python3
"""SzCORE-compliant inference entrypoint for Brain-Go-Brr."""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import mne

# Import our modules
from src.brain_brr.models import SeizureDetector
from src.brain_brr.config.schemas import Config
from src.brain_brr.streaming import StreamingPostProcessor
from src.brain_brr.constants import SAMPLING_RATE

# Channel mapping (see Section 3.2)
SZCORE_ORDER = [
    "Fp1-Avg", "F3-Avg", "C3-Avg", "P3-Avg", "O1-Avg",
    "F7-Avg", "T3-Avg", "T5-Avg", "Fz-Avg", "Cz-Avg", "Pz-Avg",
    "Fp2-Avg", "F4-Avg", "C4-Avg", "P4-Avg", "O2-Avg",
    "F8-Avg", "T4-Avg", "T6-Avg"
]

OUR_ORDER = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"
]

# Mapping: szcore_idx -> our_idx
SZCORE_TO_OURS = [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 15, 16, 17]


def load_edf_szcore(edf_path: Path) -> tuple[np.ndarray, float, float]:
    """Load SzCORE-formatted EDF and remap channels.

    Returns:
        data: (19, T) float32 in OUR channel order
        fs: sampling rate
        duration_sec: recording duration
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Verify channel order matches SzCORE spec
    for i, expected in enumerate(SZCORE_ORDER):
        if raw.ch_names[i] != expected:
            raise ValueError(
                f"Channel mismatch at index {i}: "
                f"expected '{expected}', got '{raw.ch_names[i]}'"
            )

    # Extract data and remap to our order
    data = raw.get_data()[:19]  # First 19 channels only
    remapped = data[SZCORE_TO_OURS]

    # Resample if needed (SzCORE guarantees 256Hz but verify)
    fs = raw.info["sfreq"]
    if fs != SAMPLING_RATE:
        from scipy.signal import resample
        n_samples_new = int(len(data[0]) * SAMPLING_RATE / fs)
        remapped = np.array([resample(ch, n_samples_new) for ch in remapped])
        fs = SAMPLING_RATE

    duration_sec = raw.times[-1]
    return remapped.astype(np.float32), fs, duration_sec


def run_inference(edf_path: Path, output_path: Path) -> None:
    """Run seizure detection and write HED-SCORE compliant TSV."""

    # Load and remap EDF
    eeg_data, fs, duration_sec = load_edf_szcore(edf_path)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.from_yaml(Path("/app/configs/inference.yaml"))
    model = SeizureDetector.from_config(config.model).to(device)

    checkpoint = torch.load("/app/checkpoints/best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Windowed inference
    window_samples = 60 * int(fs)  # 60 seconds
    stride_samples = 10 * int(fs)  # 10 seconds

    all_probs = []
    with torch.no_grad():
        for start in range(0, eeg_data.shape[1] - window_samples + 1, stride_samples):
            window = eeg_data[:, start:start + window_samples]
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    # Concatenate and post-process
    probs = np.concatenate(all_probs, axis=0)

    # Apply hysteresis and morphology (use streaming post-processor)
    processor = StreamingPostProcessor(config.postprocessing, sampling_rate=int(fs))

    events = []
    # ... (event detection logic)

    # Write HED-SCORE compliant TSV
    write_szcore_tsv(output_path, events, duration_sec, edf_path.stem)


def write_szcore_tsv(
    output_path: Path,
    events: list[tuple[float, float, float]],  # (onset, duration, confidence)
    recording_duration: float,
    filename: str,
) -> None:
    """Write HED-SCORE compliant TSV output.

    Format:
        onset	duration	eventType	confidence	channels	dateTime	recordingDuration
    """
    # Get current datetime for dateTime field
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w") as f:
        # Header
        f.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n")

        if len(events) == 0:
            # No seizures: write single "bckg" row
            f.write(f"0.0\t{recording_duration:.1f}\tbckg\tn/a\tall\t{dt}\t{recording_duration:.1f}\n")
        else:
            # Write each seizure event
            for onset, duration, confidence in events:
                f.write(
                    f"{onset:.3f}\t{duration:.3f}\tsz\t{confidence:.3f}\t"
                    f"all\t{dt}\t{recording_duration:.1f}\n"
                )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m szcore_inference <input.edf> <output.tsv>")
        sys.exit(1)

    edf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    run_inference(edf_path, output_path)
```

---

## 5. Input Format: EDF Files

### 5.1 SzCORE EDF Specification

| Property | Requirement |
|----------|-------------|
| **Format** | European Data Format (EDF) |
| **Channels** | Exactly 19 electrodes, 10-20 system |
| **Montage** | Unipolar, common average reference |
| **Channel Names** | `{electrode}-Avg` suffix |
| **Sampling Rate** | 256 Hz |
| **Duration** | Minimum 10 minutes, typically ~1 hour |
| **File Size** | < 1 GB |

### 5.2 Channel Order (CRITICAL)

SzCORE guarantees this exact order in EDF channels 0-18:
```
Index 0:  Fp1-Avg
Index 1:  F3-Avg
Index 2:  C3-Avg
Index 3:  P3-Avg
Index 4:  O1-Avg    <- DIFFERENT from ours
Index 5:  F7-Avg    <- DIFFERENT from ours
Index 6:  T3-Avg    <- DIFFERENT from ours
Index 7:  T5-Avg    <- DIFFERENT from ours
Index 8:  Fz-Avg
Index 9:  Cz-Avg
Index 10: Pz-Avg
Index 11: Fp2-Avg
Index 12: F4-Avg
Index 13: C4-Avg
Index 14: P4-Avg
Index 15: O2-Avg    <- DIFFERENT from ours
Index 16: F8-Avg    <- DIFFERENT from ours
Index 17: T4-Avg    <- DIFFERENT from ours
Index 18: T6-Avg    <- DIFFERENT from ours
```

Additional channels (ECG, EMG, etc.) may follow but are NOT used for common average.

---

## 6. Output Format: TSV Files

### 6.1 HED-SCORE Compliant TSV

Tab-separated values with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `onset` | float | Start time in seconds from recording beginning |
| `duration` | float | Event length in seconds |
| `eventType` | string | `sz` for seizure, `bckg` for no-seizure recordings |
| `confidence` | float/string | [0-1] or `n/a` |
| `channels` | string | `all` or comma-separated list |
| `dateTime` | string | POSIX format: `%Y-%m-%d %H:%M:%S` |
| `recordingDuration` | float | Total recording length in seconds |

### 6.2 Example Output

**With seizures:**
```tsv
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
125.5	45.2	sz	0.92	all	2025-12-28 10:30:00	3600.0
890.1	32.8	sz	0.87	all	2025-12-28 10:30:00	3600.0
```

**No seizures:**
```tsv
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
0.0	3600.0	bckg	n/a	all	2025-12-28 10:30:00	3600.0
```

---

## 7. YAML Submission File

### 7.1 Complete Template

```yaml
# Brain-Go-Brr submission to SzCORE benchmark
# File: algorithms/brain_go_brr.yaml

title: "Brain-Go-Brr v4.3: Dual-Stream TCN+SSM+GNN with Dynamic LPE for Seizure Detection"
short_title: "BrainGoBrr"
image: "ghcr.io/clarity-digital-twin/brain-go-brr:v4.3.0"
version: "4.3.0"
date: "2025-12-28"

authors:
  - given_name: "[First Name]"
    family_name: "[Last Name]"
    institution: "[Institution]"
    email: "[email]"  # Optional

license: "Apache-2.0"
repository: "https://github.com/Clarity-Digital-Twin/brain-go-brr-v2"

abstract: |
  # Algorithm Overview

  Brain-Go-Brr is a 31M parameter neural network combining Temporal Convolutional
  Networks (TCN), State Space Models (SSM via Flash Linear Attention), Graph Neural
  Networks (GNN), and Dynamic Laplacian Positional Encoding (LPE) for EEG seizure
  detection. The architecture achieves O(N) complexity through linear attention
  mechanisms.

  Key components:
  - TCN: 8 layers, [64,128,256,512] channels, stride_down=16
  - SSM: 6-layer Gated DeltaNet with bidirectional fusion
  - GNN: SSGConv (alpha=0.05), 2 layers for spatial relationships
  - LPE: 16 Laplacian eigenvectors computed dynamically per timestep

  # Input Specification

  - 19-channel EEG (10-20 international system)
  - Common average reference montage
  - 256 Hz sampling rate
  - 60-second windows with 10-second stride

  # Training Data

  - TUH EEG Seizure Corpus v2.0.3 (train split)
  - 4,667 recordings
  - Balanced sampling (8% -> 30% seizure windows)
  - Focal loss (alpha=0.5, gamma=2.0)

  # Preprocessing

  - Bandpass filter: 0.5-120 Hz
  - Notch filter: 60 Hz (US power line)
  - Per-channel z-score normalization
  - Amplitude clipping: +/- 10 sigma

  # Post-processing

  - Hysteresis thresholding (tau_on=0.86, tau_off=0.78)
  - Morphological operations (opening=11, closing=31 samples)
  - Event duration filtering (3s - 600s)
  - Event merging (gap < 2s)

  # Performance (TUSZ eval set, 836 recordings)

  - AUROC: 0.8654
  - Sensitivity @ 10 FA/24h: 35.9%
  - F1 (NEDC OVERLAP): Pending official evaluation

  # Computational Complexity

  - Parameters: 31M
  - Inference time: ~5s per 1-hour recording (GPU)
  - Memory: ~4GB GPU RAM

datasets:
  - "tuh_sz"
```

---

## 8. Scoring Methodology

### 8.1 Event-Based Scoring Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pre-ictal tolerance | 30 seconds | Detection window before marked onset |
| Post-ictal tolerance | 60 seconds | Detection window after marked offset |
| Event merge threshold | 90 seconds | Events closer are merged |
| Max event duration | 5 minutes | Longer events are split |
| Minimum overlap | Any | Any overlap counts as TP |

### 8.2 Metrics Computed

- **Sensitivity**: TP / (TP + FN)
- **Precision**: TP / (TP + FP)
- **F1-score**: 2 × sensitivity × precision / (sensitivity + precision)
- **False Alarms per 24h**: FP / total_hours × 24

### 8.3 Our Current Performance

From FLA Exp4 evaluation on TUSZ eval set:

| Metric | Value |
|--------|-------|
| Recordings | 836 |
| Duration | 127.8 hours |
| AUROC | 0.8654 |
| Sensitivity @ 10 FA/24h | 35.9% |
| NEDC OVERLAP F1 | 0.357 |

---

## 9. Implementation Plan

### Phase 1: Docker Infrastructure (1-2 days)

1. [ ] Create `docker/` directory structure
2. [ ] Write `Dockerfile` following SzCORE template
3. [ ] Create `requirements-docker.txt` (minimal deps)
4. [ ] Implement `szcore_inference.py` with channel remapping
5. [ ] Create `configs/inference.yaml` (frozen inference config)

### Phase 2: Local Testing (1 day)

1. [ ] Build Docker image locally
2. [ ] Download SzCORE test data (if available)
3. [ ] Run inference on sample EDF files
4. [ ] Validate TSV output format
5. [ ] Test channel remapping correctness

### Phase 3: Registry & Submission (1 day)

1. [ ] Push image to GitHub Container Registry
2. [ ] Create `algorithms/brain_go_brr.yaml`
3. [ ] Fork esl-epfl/szcore repository
4. [ ] Submit PR with YAML file
5. [ ] Monitor CI validation

### Phase 4: Iteration

1. [ ] Address any CI failures
2. [ ] Review benchmark results
3. [ ] Iterate on thresholds if needed

---

## 10. Risk Analysis

### 10.1 High Risk: Channel Remapping

**Risk:** Incorrect channel mapping causes severe performance degradation.

**Mitigation:**
- Triple-verify the mapping indices
- Create unit tests with known EDF samples
- Compare model activations with/without remapping on same data

### 10.2 Medium Risk: Preprocessing Differences

**Risk:** SzCORE preprocessing differs from our training preprocessing.

**Analysis:**
- SzCORE guarantees: 256 Hz, common average reference
- We trained on: 256 Hz, various references (TUSZ uses linked-ear)
- Common average ≠ linked-ear reference

**Mitigation:**
- Accept potential performance drop
- Consider re-reference layer in future versions
- Document in abstract

### 10.3 Medium Risk: Docker Dependencies

**Risk:** Missing CUDA/Mamba/FLA dependencies in Docker.

**Mitigation:**
- Use CPU-only inference initially (slower but simpler)
- Include pre-compiled wheels in image
- Test thoroughly before submission

### 10.4 Low Risk: TSV Format

**Risk:** Output format doesn't match HED-SCORE spec.

**Mitigation:**
- Use szcore-evaluation library for validation
- Copy exact format from gotman.yaml reference

---

## 11. References

### Primary Sources

1. [SzCORE Benchmark](https://epilepsybenchmarks.com/) - Main platform
2. [SzCORE GitHub](https://github.com/esl-epfl/szcore) - Submission repository
3. [SzCORE Framework Paper](https://arxiv.org/abs/2402.13005) - Technical details
4. [Contribution Guide](https://epilepsybenchmarks.com/contribute/) - Submission requirements
5. [szcore-evaluation](https://github.com/esl-epfl/szcore-evaluation) - Evaluation library

### Tools

- [epilepsy2bids](https://github.com/esl-epfl/epilepsy2bids) - Dataset conversion
- [timescoring](https://github.com/esl-epfl/timescoring) - Scoring metrics

---

## Appendix A: Full Channel Mapping Code

```python
"""Channel mapping between SzCORE and Brain-Go-Brr training order."""

# SzCORE standard order (as received in EDF)
SZCORE_CHANNELS = [
    "Fp1-Avg", "F3-Avg", "C3-Avg", "P3-Avg", "O1-Avg",
    "F7-Avg", "T3-Avg", "T5-Avg", "Fz-Avg", "Cz-Avg", "Pz-Avg",
    "Fp2-Avg", "F4-Avg", "C4-Avg", "P4-Avg", "O2-Avg",
    "F8-Avg", "T4-Avg", "T6-Avg"
]

# Our training order (from constants.py)
OUR_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"
]

def build_remap_indices() -> list[int]:
    """Build index mapping from SzCORE order to our order.

    Returns:
        List where remap[i] = j means SzCORE channel i maps to our channel j
    """
    # Strip -Avg suffix for comparison
    szcore_clean = [ch.replace("-Avg", "") for ch in SZCORE_CHANNELS]

    remap = []
    for sz_ch in szcore_clean:
        our_idx = OUR_CHANNELS.index(sz_ch)
        remap.append(our_idx)

    return remap

# Pre-computed: SZCORE_TO_OURS[szcore_idx] = our_idx
SZCORE_TO_OURS = [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 15, 16, 17]

def remap_eeg(data: "np.ndarray") -> "np.ndarray":
    """Remap SzCORE channel order to our training order.

    Args:
        data: (19, T) or (B, 19, T) array in SzCORE order

    Returns:
        Same shape array in our training order
    """
    import numpy as np

    if data.ndim == 2:
        return data[SZCORE_TO_OURS]
    elif data.ndim == 3:
        return data[:, SZCORE_TO_OURS, :]
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

# Verification
if __name__ == "__main__":
    # Verify mapping is correct
    computed = build_remap_indices()
    assert computed == SZCORE_TO_OURS, f"Mismatch: {computed} vs {SZCORE_TO_OURS}"
    print("Channel mapping verified!")

    # Print human-readable mapping
    szcore_clean = [ch.replace("-Avg", "") for ch in SZCORE_CHANNELS]
    print("\nChannel Mapping (SzCORE -> Ours):")
    for i, (sz, our_idx) in enumerate(zip(szcore_clean, SZCORE_TO_OURS)):
        print(f"  SzCORE[{i:2d}] {sz:4s} -> Ours[{our_idx:2d}] {OUR_CHANNELS[our_idx]:4s}")
```

---

## Appendix B: Decision Matrix

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| Channel handling | Remap at inference | Retrain | **Option A** (faster) |
| Docker base | python:3.11-slim | pytorch/pytorch:2.5.0 | slim (smaller image) |
| GPU support | Include CUDA | CPU only | CPU initially, add GPU later |
| Output format | HED-SCORE TSV | CSV_BI | **HED-SCORE TSV** (required) |
| Initial submission | FLA Exp4 checkpoint | Retrain | **FLA Exp4** (proven) |

---

**END OF SPECIFICATION**

**Next Step:** Senior review of this specification before implementation.
