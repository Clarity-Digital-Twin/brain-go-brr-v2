# SzCORE Benchmark Submission Specification

**Version**: 1.0.0
**Date**: 2025-12-07
**Status**: PLANNING

---

## Executive Summary

This document specifies requirements for submitting Brain-Go-Brr to the SzCORE benchmark at [epilepsybenchmarks.com](https://epilepsybenchmarks.com). The submission is **feasible with moderate effort** and requires a **separate inference-only Docker image** (not modifying the existing training Dockerfile).

**Key Finding**: The channel order difference is **NOT a blocker** - it's a trivial index permutation at inference time.

---

## 1. Channel Mapping Analysis

### 1.1 The Two Channel Orders

**Your Model (TUSZ/Temple convention)**:
```
Index:  0    1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18
Name:  Fp1  F3  C3  P3  F7  T3  T5  O1  Fz  Cz  Pz  Fp2 F4  C4  P4  F8  T4  T6  O2
```

**SzCORE Format (10-20 referential common avg)**:
```
Index:  0    1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18
Name:  Fp1  F3  C3  P3  O1  F7  T3  T5  Fz  Cz  Pz  Fp2 F4  C4  P4  O2  F8  T4  T6
                       ^       ^           ^                   ^       ^
                      diff    diff        same                diff    diff
```

### 1.2 Exact Differences

| Position | Your Model | SzCORE | Match? |
|----------|-----------|--------|--------|
| 0-3      | Fp1,F3,C3,P3 | Fp1,F3,C3,P3 | ✓ Same |
| 4        | **F7** | **O1** | ✗ |
| 5        | **T3** | **F7** | ✗ |
| 6        | **T5** | **T3** | ✗ |
| 7        | **O1** | **T5** | ✗ |
| 8-10     | Fz,Cz,Pz | Fz,Cz,Pz | ✓ Same |
| 11-14    | Fp2,F4,C4,P4 | Fp2,F4,C4,P4 | ✓ Same |
| 15       | **F8** | **O2** | ✗ |
| 16       | **T4** | **F8** | ✗ |
| 17       | **T6** | **T4** | ✗ |
| 18       | **O2** | **T6** | ✗ |

**Summary**: 11/19 positions match, 8 positions differ (O1/O2 and surrounding temporal electrodes are swapped).

### 1.3 Channel Remapping Solution

To convert SzCORE input to your model's expected order:

```python
# Maps: SzCORE index -> Your model index
# Given data shape (19, T), apply: data_model = data_szcore[SZCORE_TO_TUSZ, :]

SZCORE_TO_TUSZ = [
    0,   # Fp1 -> Fp1 (position 0)
    1,   # F3  -> F3  (position 1)
    2,   # C3  -> C3  (position 2)
    3,   # P3  -> P3  (position 3)
    5,   # O1  -> O1  (SzCORE[4] -> position 7, but we need SzCORE[5]=F7 -> position 4)
    6,   # F7  -> F7  (SzCORE[5] -> position 4)
    7,   # T3  -> T3  (SzCORE[6] -> position 5)
    4,   # T5  -> T5  (SzCORE[7] -> position 6) ... wait let me recalculate
    8,   # Fz  -> Fz
    9,   # Cz  -> Cz
    10,  # Pz  -> Pz
    11,  # Fp2 -> Fp2
    12,  # F4  -> F4
    13,  # C4  -> C4
    14,  # P4  -> P4
    16,  # O2  -> O2  (SzCORE[15] -> position 18)
    17,  # F8  -> F8  (SzCORE[16] -> position 15)
    18,  # T4  -> T4  (SzCORE[17] -> position 16)
    15,  # T6  -> T6  (SzCORE[18] -> position 17)
]
```

**Corrected mapping** (SzCORE channel at index i should go to which TUSZ position):

```python
# To reorder: model_input[tusz_idx] = szcore_data[szcore_idx]
# We need: model_input = szcore_data[REORDER_INDICES]

SZCORE_TO_TUSZ_ORDER = [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 15]
#                                    F7 T3 T5 O1                           F8 T4 T6 O2
```

This is a single line: `data = data[SZCORE_TO_TUSZ_ORDER, :]`

---

## 2. Does Channel Order Affect the Model?

**Yes, but it's easily fixable.**

### 2.1 Model Components That Care About Channel Order

| Component | How It Uses Channels | Impact |
|-----------|---------------------|--------|
| **TCN Encoder** | First Conv1d layer has learned weights per channel position | High - each position expects specific electrode's frequency content |
| **Node Stream** | Reshapes to (B, 19, 64, T) - position = electrode identity | High - electrode 4 expects F7's data, not O1's |
| **Edge Stream** | Computes pairwise similarities between positions 0-18 | High - edge (4,5) means F7↔T3 in your model, not O1↔F7 |
| **GNN** | Adjacency computed from edge similarities | High - spatial relationships are position-based |

### 2.2 Why This Isn't a Blocker

The model learns **position-to-electrode mappings** during training. At inference:
- Position 4 always expects F7's signal (low frequency, temporal activity)
- Position 7 always expects O1's signal (alpha rhythm, occipital)

As long as we **permute the input channels** before inference, the model sees exactly what it expects. No retraining needed.

---

## 3. Separate Docker Image Strategy

**Recommendation**: Create `Dockerfile.szcore` (separate from main `Dockerfile`)

### 3.1 Why Separate?

| Aspect | Main Dockerfile | SzCORE Dockerfile |
|--------|----------------|-------------------|
| **Purpose** | Training, development, full stack | Inference only |
| **Size** | ~15GB (full CUDA devel, build tools) | ~5GB (CUDA runtime only) |
| **Dependencies** | All training deps (wandb, tensorboard, etc) | Minimal inference deps |
| **Secrets** | May need WANDB_API_KEY | No secrets (air-gapped evaluation) |
| **Model Weights** | Downloaded/mounted at runtime | **Baked into image** |
| **Entrypoint** | `train` command | `inference` command with ENV vars |

### 3.2 SzCORE Docker Requirements

```dockerfile
# Environment variables (set by SzCORE evaluation harness)
ENV INPUT=""   # Relative path to EDF file under /data
ENV OUTPUT=""  # Relative path to TSV file under /output

# Volumes (mounted by evaluation harness)
# /data   - read-only, contains EDF file
# /output - read-write, for TSV output

# Command must process: /data/$INPUT -> /output/$OUTPUT
CMD ["python3", "-m", "brain_brr.inference.szcore", "/data/${INPUT}", "/output/${OUTPUT}"]
```

### 3.3 What Goes in the SzCORE Image

```
/app/
├── src/brain_brr/
│   ├── inference/
│   │   └── szcore.py       # NEW: Single-file inference entrypoint
│   ├── models/             # Model definition only
│   ├── post/               # Post-processing
│   ├── events/             # Event detection
│   └── constants.py        # Channel order, etc.
├── weights/
│   └── best.pt             # BAKED IN: Your trained checkpoint
└── configs/
    └── inference.yaml      # Inference-only config
```

---

## 4. Implementation Checklist

### 4.1 New Code Required

| File | Purpose | Complexity |
|------|---------|------------|
| `src/brain_brr/inference/szcore.py` | Main entrypoint for SzCORE inference | Medium |
| `src/brain_brr/inference/channel_remap.py` | Channel reordering logic | Trivial |
| `src/brain_brr/events/export_tsv.py` | TSV export in HED-SCORE format | Easy |
| `Dockerfile.szcore` | Inference-only Docker image | Medium |
| `deploy/szcore/brain-go-brr.yaml` | Algorithm submission YAML | Easy |

### 4.2 TSV Output Format

SzCORE requires tab-separated values with specific columns:

```tsv
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
120.5	45.3	sz	0.87	all	2025-01-15 14:32:00	3600.0
```

**Columns**:
- `onset`: Seconds from recording start
- `duration`: Event duration in seconds
- `eventType`: "sz" for seizure, "bckg" if no seizures
- `confidence`: 0-1 (optional, can be "n/a")
- `channels`: "all" or comma-separated list
- `dateTime`: POSIX format from EDF header
- `recordingDuration`: Total recording length in seconds

### 4.3 Scoring Parameters (for reference)

```yaml
# These are what SzCORE uses - don't need to implement, just understand
pre_ictal_tolerance: 30   # seconds before seizure onset
post_ictal_tolerance: 60  # seconds after seizure end
merge_threshold: 90       # seconds - merge events closer than this
max_event_duration: 300   # 5 minutes - split longer events
```

---

## 5. Blockers and Considerations

### 5.1 True Blockers (Must Resolve)

| Blocker | Status | Resolution |
|---------|--------|------------|
| **No checkpoint in repo** | BLOCKING | Copy best.pt from NVIDIA box |
| **Missing TSV exporter** | BLOCKING | Implement `export_tsv.py` |
| **No inference CLI** | BLOCKING | Implement `szcore.py` entrypoint |

### 5.2 Minor Issues (Easy to Fix)

| Issue | Resolution |
|-------|------------|
| Channel order mismatch | Single-line permutation at inference |
| Different output format | Implement TSV export (straightforward) |
| EDF reading without labels | Already supported (pyedflib) |

### 5.3 Non-Issues

| Concern | Why It's Not a Problem |
|---------|------------------------|
| "Different channel mapping" | Trivial index permutation |
| "Separate Docker image" | This is the RIGHT approach |
| "Need to modify main code" | No - all new code in `inference/` |
| "Model trained on different order" | Permutation handles this |

---

## 6. Does Pydantic Help Here?

**Short answer**: Marginally, not essential.

### 6.1 Where Pydantic Could Help

| Use Case | Benefit |
|----------|---------|
| Validating inference config | Type safety, defaults |
| TSV output schema | Ensure correct columns |
| CLI argument parsing | Could use, but Click is fine |
| Algorithm YAML validation | Nice-to-have before submission |

### 6.2 Where Pydantic Doesn't Help

| Area | Why |
|------|-----|
| Channel remapping | Just array indexing |
| EDF reading | pyedflib handles this |
| Model loading | PyTorch checkpoint loading |
| Docker setup | Shell scripting |

**Recommendation**: Don't add Pydantic complexity for the SzCORE submission. Your existing code patterns (Click CLI, dataclasses for events) are sufficient.

---

## 7. Professional Recommendation

### 7.1 Architecture Decision

```
┌─────────────────────────────────────────────────────────────┐
│                    EXISTING (unchanged)                      │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │   Dockerfile     │     │  src/brain_brr/  │             │
│  │   (training)     │     │   train/         │             │
│  │                  │     │   eval/          │             │
│  │                  │     │   cli/           │             │
│  └──────────────────┘     └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      NEW (additive)                          │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ Dockerfile.szcore│     │  src/brain_brr/  │             │
│  │   (inference)    │     │   inference/     │ ← NEW       │
│  │                  │     │     szcore.py    │             │
│  │                  │     │     remap.py     │             │
│  └──────────────────┘     │   events/        │             │
│                           │     export_tsv.py│ ← NEW       │
│  ┌──────────────────┐     └──────────────────┘             │
│  │ deploy/szcore/   │                                       │
│  │   brain-go-brr   │                                       │
│  │   .yaml          │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Your Channel Order Is Fine

Your TUSZ channel order follows the **Temple University Hospital convention**, which is:
- Standard in US clinical practice
- Used by the largest public EEG seizure dataset
- Logical grouping: Left hemisphere → Midline → Right hemisphere

SzCORE uses a different convention (O1/O2 between P3/P4 and F7/F8). Neither is "better" - they're just different conventions.

### 7.3 Effort Estimate

| Task | Time | Can Be Parallelized? |
|------|------|---------------------|
| Implement `export_tsv.py` | 1 hour | Yes |
| Implement `szcore.py` entrypoint | 2-3 hours | Yes |
| Implement channel remapping | 30 min | Yes |
| Create `Dockerfile.szcore` | 2 hours | After above |
| Create algorithm YAML | 30 min | Anytime |
| Copy checkpoint from NVIDIA box | Manual | You only |
| Build and test Docker image | 1-2 hours | After all above |
| Submit PR to SzCORE repo | 30 min | After all above |

**Total**: ~8-10 hours of development + manual checkpoint copy

---

## 8. Submission Process Summary

1. **Copy checkpoint** from NVIDIA box to repo (e.g., `weights/best.pt`)
2. **Implement inference code** in `src/brain_brr/inference/`
3. **Build Docker image**: `docker build -f Dockerfile.szcore -t ghcr.io/clarity-digital-twin/brain-go-brr:szcore .`
4. **Push to GHCR**: `docker push ghcr.io/clarity-digital-twin/brain-go-brr:szcore`
5. **Create algorithm YAML** with metadata
6. **Open PR** to [esl-epfl/szcore](https://github.com/esl-epfl/szcore)
7. **Wait for CI** to validate your submission
8. **Merge** and appear on leaderboard

---

## Appendix A: Algorithm YAML Template

```yaml
title: "Brain-Go-Brr: TCN + BiMamba2 + GNN Seizure Detection"
short_title: "Brain-Go-Brr v4"
image: "ghcr.io/clarity-digital-twin/brain-go-brr:szcore"

authors:
  - given_names: Your
    family_names: Name
    email: your.email@example.com
    affiliation: Your Institution

version: "4.1.0"
date_released: "2025-XX-XX"

abstract: >
  Brain-Go-Brr is a seizure detection system combining Temporal Convolutional
  Networks (TCN) for multi-scale feature extraction, Bidirectional Mamba-2
  (BiMamba2) for O(N) global temporal modeling, and Graph Neural Networks
  (GNN) with dynamic Laplacian positional encoding for spatial electrode
  relationships. Trained on TUH EEG Corpus (TUSZ v2.0.3) with balanced
  sampling and focal loss for class imbalance handling.

license: "MIT"  # Or your license
repository: "https://github.com/Clarity-Digital-Twin/brain-go-brr-v2"

datasets:
  - "TUSZ v2.0.3"
```

---

## Appendix B: Quick Reference - Channel Remapping

```python
# In src/brain_brr/inference/channel_remap.py

# SzCORE channel names in their order
SZCORE_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6"
]

# Your model's expected channel names (from constants.py)
TUSZ_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"
]

def get_szcore_to_tusz_indices():
    """Return index mapping: data_tusz = data_szcore[indices]"""
    return [SZCORE_CHANNELS.index(ch) for ch in TUSZ_CHANNELS]

# Result: [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 15]

def remap_channels(eeg_data):
    """Remap SzCORE channel order to TUSZ order.

    Args:
        eeg_data: numpy array shape (19, T) in SzCORE order

    Returns:
        numpy array shape (19, T) in TUSZ order
    """
    indices = get_szcore_to_tusz_indices()
    return eeg_data[indices, :]
```
