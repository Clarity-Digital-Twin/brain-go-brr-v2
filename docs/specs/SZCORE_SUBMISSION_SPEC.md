# SzCORE/epilepsybenchmarks.com Submission Specification

**Version:** 1.1.0
**Date:** 2025-12-29
**Status:** IMPLEMENTED - READY FOR SUBMISSION
**Author:** Claude Code (initial draft) + Brain-Go-Brr team (verification + implementation)
**Last Verified Against (SSOT):** epilepsybenchmarks.com/framework, epilepsybenchmarks.com/contribute, and `esl-epfl/szcore/.github/workflows/1-pr-check.yml` (2025-12-29)

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
9. [Implementation Status](#9-implementation-status)
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
| **Entrypoint** | `CMD python3 -m <module> "/data/${INPUT}" "/output/${OUTPUT}"` |
| **Network** | OFFLINE (no internet access during inference) |
| **CI Validation** | GitHub Actions (`ubuntu-latest`, CPU-only) - quick format check on `tests/data/unipolar.edf` |
| **Evaluation Resources** | Official docs do not specify; the 2025 challenge evaluation ran on the EPFL Research Computing Platform (CaaS) with **400+ GPUs** and executed GPU workloads on A100-SXM4-80GB nodes (arXiv:2505.18191) |

### 2.2 YAML Submission File

SzCORE validates this YAML against `esl-epfl/szcore/config/schema.json` in PR CI.

- Required keys (schema): `title`, `short_title`, `image`, `authors`, `license`
- SSOT template: `deploy/szcore/brain_go_brr.yaml.template` (copy into `esl-epfl/szcore/algorithms/brain_go_brr.yaml`)

Minimal required example (spaces only; no tabs):

```yaml
title: "Brain-Go-Brr: TCN+SSM+GNN Seizure Detection"
short_title: "BrainGoBrr"
image: "ghcr.io/clarity-digital-twin/brain-go-brr-szcore:v4.4.0"
authors:
  - given_names: "John H."
    family_names: "Jung"
    affiliation: "Clarity Digital Twin"
license: "Apache-2.0"
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

SSOT Docker image definition: `deploy/szcore/Dockerfile`.

Key invariants (must match SzCORE PR CI + framework/contribute docs):
- Runs as non-root user
- Reads `/data/${INPUT}` and writes `/output/${OUTPUT}`
- Produces HED-SCORE TSV header exactly (tabs, not spaces)
- Runs offline at inference time (no network calls)
- Bundles the Exp4 checkpoint at `/app/checkpoints/best.pt`

Notes:
- The official SzCORE template uses `python:X-slim`; our SSOT Dockerfile uses a multi-stage CUDA base to compile and ship CUDA extensions (`mamba-ssm`, `causal-conv1d`) and align with the project CUDA 12.4 lock.
- `.dockerignore` must allow the checkpoint path used by the Docker build (`results/local_fla_exp4_cyclic/checkpoints/best.pt`).

### 4.2 Inference Entrypoint

SSOT entrypoint module: `src/brain_brr/szcore/__main__.py` (invoked as `python3 -m src.brain_brr.szcore`).

Behavior:
- Loads EDF and enforces/picks SzCORE channels (`src/brain_brr/szcore/loader.py`).
- Remaps SzCORE channel order into our `CHANNEL_NAMES_10_20` order (`src/brain_brr/szcore/channels.py`).
- If `torch.cuda.is_available()` is true: loads `configs/szcore_inference.yaml` + `best.pt` and runs windowed GPU inference (`src/brain_brr/szcore/infer.py`).
- If no GPU is available: uses a conservative CPU heuristic fallback (real detector; no dummy TSV).
- Writes HED-SCORE TSV (`src/brain_brr/szcore/hed_score.py`).

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

Note: In the current SzCORE benchmark, the `confidence` and `channels` fields are not used for evaluation (epilepsybenchmarks.com/contribute). Keep the columns but using `n/a` is acceptable.

### 6.2 Example Output

**With seizures:**
```tsv
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
125.5	45.2	sz	n/a	n/a	2025-12-28 10:30:00	3600.0
890.1	32.8	sz	n/a	n/a	2025-12-28 10:30:00	3600.0
```

**No seizures:**
```tsv
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
0.0	3600.0	bckg	n/a	n/a	2025-12-28 10:30:00	3600.0
```

---

## 7. YAML Submission File

### 7.1 Complete Template

SSOT template: `deploy/szcore/brain_go_brr.yaml.template` (kept schema-valid against `esl-epfl/szcore/config/schema.json`).

```yaml
---
# SzCORE algorithm submission file (copy into `esl-epfl/szcore/algorithms/`).
# Must validate against: https://github.com/esl-epfl/szcore/blob/main/config/schema.json

title: "Brain-Go-Brr v4.4.0: Dual-Stream TCN + Gated DeltaNet + GNN for Seizure Detection"
short_title: "BrainGoBrr"
image: "ghcr.io/clarity-digital-twin/brain-go-brr-szcore:v4.4.0"
version: "4.4.0"
date_released: "2025-12-29"

authors:
  - given_names: "John H."
    family_names: "Jung"
    affiliation: "Clarity Digital Twin"

repository: "https://github.com/Clarity-Digital-Twin/brain-go-brr-v2"
license: "Apache-2.0"

abstract: >-
  Brain-Go-Brr is a seizure detection algorithm for long-term scalp EEG.
  It expects 19-channel 10-20 EEG at 256 Hz and produces HED-SCORE TSV event
  annotations. This submission uses Brain-Go-Brr's V3 architecture and the
  FLA (Gated DeltaNet) variant trained on TUH EEG Seizure Corpus.

datasets:
  - "TUH EEG Sz Corpus v2"
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

### 8.3 Internal Reference Metrics (Chosen Checkpoint)

These metrics are from our held-out TUSZ eval run for the exact checkpoint bundled in the SzCORE image:
- Results JSON: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- Eval log: `results/local_fla_exp4_cyclic/eval_v2.log`

Note: SzCORE leaderboard scoring is event-based F1 with pre/post-ictal tolerances; these internal TUSZ numbers are not directly comparable.

| Metric | Value |
|--------|-------|
| EDF files in eval dir | 865 |
| Recordings included in metrics | 836 |
| Duration | 127.8139 hours |
| AUROC | 0.8654 |
| PR-AUC | 0.5409 |
| ECE | 0.0289 |
| Sensitivity @ 10 FA/24h | 35.9% |
| Sensitivity @ 5 FA/24h | 27.1% |
| Sensitivity @ 2.5 FA/24h | 18.6% |
| Sensitivity @ 1 FA/24h | 5.8% |
| Val Loss | 0.0901 |

---

## 9. Implementation Status

Implementation is complete in this repo:
- SzCORE inference package: `src/brain_brr/szcore/`
- Frozen inference config: `configs/szcore_inference.yaml`
- Buildable Docker image: `deploy/szcore/Dockerfile` + `deploy/szcore/requirements.txt`
- Submission YAML template: `deploy/szcore/brain_go_brr.yaml.template`
- Unit tests (mapping + TSV header): `tests/unit/szcore/test_szcore_submission.py`

Next actions to submit:
1. Build + push the Docker image to a public registry (GHCR recommended).
2. Copy and fill `deploy/szcore/brain_go_brr.yaml.template` into `esl-epfl/szcore/algorithms/` as `brain_go_brr.yaml`.
3. Open a PR; SzCORE CI validates schema + header via `.github/workflows/1-pr-check.yml`.

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

**Reality:** Challenge evaluation ran on the EPFL Research Computing Platform (CaaS) and used A100 GPUs for GPU workloads (arXiv:2505.18191). The public PR CI check remains CPU-only and only validates TSV format.

**Mitigation:**
- Use the SSOT multi-stage build in `deploy/szcore/Dockerfile`:
  - builder: `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel` (build CUDA wheels)
  - runtime: `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime` (stable Python + CUDA runtime)
- Base image pins `torch==2.5.0+cu124`; `deploy/szcore/requirements.txt` pins PyG wheels + runtime deps.
- Keep PR CI (CPU-only) safe: the container must still run and emit a valid TSV header (use a real CPU fallback, not a dummy TSV).
- Runtime must include a C compiler (`build-essential`) for Triton to initialize its CUDA backend (required by `fla`).
- Before submission: run a GPU smoke test locally (SzCORE PR CI does not validate GPU kernels).

### 10.4 Low Risk: TSV Format

**Risk:** Output format doesn't match HED-SCORE spec.

**Mitigation:**
- Use szcore-evaluation library for validation
- Copy exact format from gotman.yaml reference

---

## 11. References

### Primary Sources

1. [Framework Docs](https://epilepsybenchmarks.com/framework/) - Channel order, HED-SCORE TSV, scoring parameters
2. [Contribution Guide](https://epilepsybenchmarks.com/contribute/) - Docker requirements + offline execution
3. [SzCORE GitHub](https://github.com/esl-epfl/szcore) - Submission repository
4. [PR CI Workflow](https://raw.githubusercontent.com/esl-epfl/szcore/main/.github/workflows/1-pr-check.yml) - Schema + header validation
5. [YAML Schema](https://raw.githubusercontent.com/esl-epfl/szcore/main/config/schema.json) - Required YAML keys/format
6. [Dockerfile Template](https://raw.githubusercontent.com/esl-epfl/szcore/main/config/template.Dockerfile) - Official Docker conventions
7. [SzCORE Framework Paper](https://arxiv.org/abs/2402.13005) - Technical details
8. [SzCORE Challenge Report](https://arxiv.org/abs/2505.18191) - Evaluation compute platform (CaaS, GPU workloads)

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
        out = np.empty_like(data)
        out[SZCORE_TO_OURS] = data
        return out
    elif data.ndim == 3:
        out = np.empty_like(data)
        out[:, SZCORE_TO_OURS, :] = data
        return out
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
| Docker base | python:*slim* (SzCORE template) | pytorch/pytorch:2.5.0-cuda12.4-*runtime* (multi-stage build) | **pytorch/pytorch:2.5.0-cuda12.4-*runtime*** (stable Python + CUDA runtime; builds CUDA wheels) |
| GPU support | CUDA wheels + GPU inference | CPU-only | **GPU inference** for benchmark + CPU heuristic fallback for PR CI |
| Output format | HED-SCORE TSV | CSV_BI | **HED-SCORE TSV** (required) |
| Initial submission | FLA Exp4 checkpoint | Retrain | **FLA Exp4** (proven) |

---

**END OF SPECIFICATION**

**Next Step:** Build + push `deploy/szcore/Dockerfile`, then submit `brain_go_brr.yaml` PR to `esl-epfl/szcore` (see Section 9).
