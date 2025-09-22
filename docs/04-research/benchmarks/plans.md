# Research Benchmark & Ablation Plan — Brain‑Go‑Brr v2

Last updated: 2025‑09‑20 (Revised with optimal dataset strategy)

Status: Planning document for rigorous, publishable benchmarking of a Bi‑Mamba‑2 + U‑Net + ResCNN seizure detector. Targets SzCORE/NEDC‑style event metrics on TUSZ and cross‑dataset generalization.

## Purpose

- Establish a clinically relevant, reproducible benchmark suite for event‑based seizure detection with O(N) sequence modeling.
- Provide an ablation and reporting protocol suitable for EMBC/JBHI/ML4H or NeurIPS Datasets & Benchmarks (D&B).
- Separate goals (pending) from claims (current code functionality).

## Datasets & Splits

### Primary Strategy: Clean Baseline First

**Training Set:**
- TUSZ train only (no mixing)
- **Rationale**: Establishes strongest baseline, optimizes for primary benchmark

**Validation Set:**
- TUSZ dev only (threshold tuning, early stopping, hyperparameter selection)

**Test Sets (Zero-Shot Evaluation):**
- TUSZ eval: Primary benchmark (patient-disjoint, one-shot)
- CHB-MIT: Cross-dataset generalization (all 23 patients, zero-shot)
  - **Key Result**: No CHB-MIT in training → strongest generalization claim
  - Most compelling evidence for clinical deployment readiness

### Dataset Characteristics

| Dataset | Patient-Disjoint | N Patients/Subjects | Role | Rationale |
|---------|-----------------|---------------------|------|----------|
| TUSZ | ✅ YES | ~600 train, 50 dev, 50 eval | Primary train/test | Gold standard, proper splits |
| CHB-MIT | ✅ YES | 23 patients | Zero-shot test only | Cross-site generalization |
| Siena | ❌ NO | 14 subjects (PN00-PN13) | Future ablation only | Not used in primary experiments |

### Paths
- TUSZ: `data_ext4/tusz/edf/{train,dev,eval}/`
- CHB-MIT: `data_ext4/chb-mit/`
- Siena: `data_ext4/siena-scalp-eeg-database-1.0.0/`

### Critical Notes
- Channel synonyms enforced (T7→T3, T8→T4, P7→T5, P8→T6)
- Canonical 10-20 order preserved (see `src/brain_brr/constants.py`)
- Siena reserved for future ablation studies after baseline established
- Document exact dataset versions and any repairs applied

## Metrics (Event‑Level + Operating Curves)

### Primary Metrics (Publication Focus)

- NEDC/TAES‑style event metrics:
  - Sensitivity/Recall@
    - 10 false alarms (FA)/24 h
    - 5 FA/24 h
    - 1 FA/24 h
  - Event F1, mean temporal alignment error
- Curves:
  - Sensitivity vs FA/24 h
  - Precision‑Recall; ROC (for sanity, not primary)
- System metrics:
  - Streaming latency per 60s window (target <100 ms)
  - Peak memory and throughput (windows/s) on CPU/GPU
- Export:
  - CSV_BI event files per TUH convention; JSON summaries
- **Cross-Dataset Zero-Shot**:
  - Report TUSZ→CHB-MIT transfer without any target training
  - Key differentiator for publication impact

## Preprocessing (SSOT)

- EDF read via MNE; canonical 10‑20 montage
- Bandpass 0.5–120 Hz; notch 60 Hz
- Resample to 256 Hz
- Window 60 s, stride 10 s (83% overlap)
- Per‑channel z‑score (train statistics only)
- Channel ordering invariant across pipeline

## Model Suite & Ablations

Baseline and variants trained/evaluated with identical preprocessing and post‑processing:

- Architectures
  - A0: CNN‑only (U‑Net + ResCNN; no Mamba)
  - A1: U‑Net + ResCNN + Transformer (SeizureTransformer baseline)
  - A2: U‑Net + ResCNN + Mamba (unidirectional)
  - A3: U‑Net + ResCNN + Bi‑Mamba‑2 (canonical)
- Bidirectionality
  - Uni vs Bi; report ΔFA/24 h for fixed sensitivity
- Mamba internals
  - d_model ∈ {256, 512}, n_layers ∈ {2, 4, 6}, d_state ∈ {8, 16}
  - conv kernel on CUDA coerced to {2,3,4}; evaluate fallback Conv1d path via `SEIZURE_MAMBA_FORCE_FALLBACK=1`
- ResCNN
  - n_blocks ∈ {0, 1, 3}; kernel sets {[3], [3,5,7]}
- Decoder
  - With/without skip attention (if present); upsampling modes

## Post‑Processing Ablations

- Hysteresis thresholds: τ_on ∈ [0.7, 0.95], τ_off ∈ [0.6, 0.9]
- Morphology: dilation/erosion kernel sizes ∈ {0, 1, 3}
- Min/Max duration constraints
- Stitching across window/chunk boundaries on/off
- Threshold selection via FA target on dev set; lock before eval

## Training Protocol

### Training Strategy: TUSZ-Only Baseline

**Primary Training (Clean Baseline):**
- Train exclusively on TUSZ train
- No dataset mixing or complex sampling
- Optimize purely for TUSZ dev performance
- Most defensible and standard approach

**Why this is optimal for initial work:**
1. **Maximizes TUSZ eval scores** (primary benchmark everyone compares)
2. **Cleanest methodology** (no sampling decisions to defend)
3. **Establishes strong baseline** (required before exploring variations)
4. **Standard practice** (what most papers do)

**Future Ablations (After Baseline):**
- If CHB-MIT transfer is weak → experiment with Siena augmentation
- If TUSZ performance plateaus → try mixed training strategies
- Report as "supplementary experiments" if improvements found

- Hardware profiles
  - Local WSL2 (WSL‑safe): `configs/tusz_train_wsl2.yaml` (num_workers=0)
  - Modal A100‑80GB: `configs/modal/train.yaml`
- Optimizer/schedule
  - AdamW, cosine decay with warmup; gradient clipping
- Regularization
  - Label smoothing (optional), dropout grids
- Class balancing
  - Balanced sampler or per‑epoch reweighting; report class prior
- Early stopping on dev metric; patience K
- Seeds and determinism flags pinned; 3‑seed repeats for main rows

## Reproducibility

- Configs checked into `configs/` with frozen hyperparams
- Environment lock via `uv.lock`; PyTorch/cuDNN versions logged
- Exact commit hash recorded in metrics JSON and event exports
- Checkpoints published with config + metrics
- Modal runs log:
  - Docker/image digest, hardware, runtime, volumes
  - Command line and environment (excluding secrets)

## Evaluation & Export

- CLI
  - Train: `python -m src train <config.yaml>`
  - Evaluate: `python -m src evaluate <checkpoint.pt> <data_dir> --config <config.yaml> --output-json <metrics.json> [--output-csv-bi <dir_or_file>]`
- Duration source
  - Use dataset/window metadata for total duration; do not approximate as len(windows)*60 s
- Guardrails
  - Warn and short‑circuit AUROC/PR if labels are single‑class or missing; still export events

## Reporting Package (Per Model)

### Core Results Matrix

| Model | TUSZ eval Sens@10FA | CHB-MIT Sens@10FA | Δ Transfer |
|-------|---------------------|-------------------|------------|
| A0: CNN-only | Target | Zero-shot | Gap % |
| A1: Transformer | Target | Zero-shot | Gap % |
| A2: Uni-Mamba | Target | Zero-shot | Gap % |
| A3: Bi-Mamba (ours) | Target | Zero-shot | Gap % |

**Key Claims**:
1. Bi-Mamba achieves smallest transfer gap (best generalization)
2. Zero-shot CHB-MIT performance competitive with supervised baselines

- Tables
  - Sensitivity at 10/5/1 FA/24 h (dev, eval)
  - Event F1, TAES summary
  - Latency (p50/p95), memory footprint, throughput
- Plots
  - Sensitivity vs FA/24 h curves
  - PR/ROC (appendix)
- Error analysis
  - Per‑patient breakdown; event length histograms; false‑alarm taxonomy
- Interpretability (optional but recommended)
  - Channel‑wise attribution or per‑channel head; FP gating experiment

## Publication Targets & Framing

### Unique Contributions
1. **First Bi-Mamba for EEG**: O(N) bidirectional sequence modeling
2. **Zero-shot cross-dataset**: TUSZ→CHB-MIT without target training
3. **Rigorous benchmark**: Patient-disjoint, event-level TAES metrics
4. **Open artifacts**: Code, configs, checkpoints, event files

- EMBC / JBHI / ML4H
  - Applied/clinical framing; strong if cross‑dataset results and practical FP‑reduction components included
- NeurIPS Datasets & Benchmarks
  - Benchmark as contribution (protocol + code + ablations + artifacts)
- NeurIPS main track
  - Unlikely without new modeling ideas; not primary target

## Milestones & Timeline (Indicative)

1. Week 1–2: Stabilize pipeline, complete CLI evaluate duration fix, write export tests
2. Week 3–4: Baselines (A0, A1) trained on TUSZ; dev tuning + curves
3. Week 5–6: Canonical (A3) + variants; cross‑dataset CHB‑MIT
4. Week 7: Error taxonomy; interpretability/gating experiment
5. Week 8: Draft report; artifact release (configs, checkpoints, events, metrics)

### Training Runs Priority
1. A3 Bi-Mamba on TUSZ-only → eval TUSZ & CHB-MIT (primary result)
2. A0 CNN-only on TUSZ-only (baseline comparison)
3. A2 Uni-Mamba on TUSZ-only (bidirectionality ablation)
4. A1 Transformer on TUSZ-only if compute allows
5. (Optional) A3 with Siena augmentation if time permits

## Risks & Mitigations

- Data leakage
  - Enforce patient‑level disjoint splits; CI check on path roots
- Unlabeled sets
  - Metrics guard + clear CLI messaging; events still exported
- CUDA/Mamba kernel constraints
  - Coerce conv kernel on CUDA; document fallback flag
- WSL2 instability
  - num_workers=0; pin_memory=false; persistent_workers=false

## Test Plan Alignment

- Add CLI/evaluate integration tests (tuple batches, sigmoid, CSV_BI write)
- Export invariants tests (headers, monotonic times, stop ≥ start, confidence ∈ [0,1])
- Post‑processing edge cases (all‑zeros/ones probs; total_hours=0; stitching)
- Config sanity tests (correct data splits; batch_size under training)

## Command Cookbook

### Primary Training Commands
- TUSZ-only training (main experiments):
  ```bash
  python -m src train configs/tusz_only_a100.yaml
  ```

### Zero-Shot Evaluation
- CHB-MIT zero-shot test:
  ```bash
  python -m src evaluate checkpoints/best_tusz_siena.pt \
    data_ext4/chb-mit/ --config configs/chb_mit_zero_shot.yaml \
    --output-json results/chb_mit_zero_shot.json
  ```

- Local smoke: `python -m src train configs/smoke_test.yaml`
- Local full (WSL2‑safe): `python -m src train configs/tusz_train_wsl2.yaml`
- Modal smoke: `modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml`
- Modal A100: `modal run --detach deploy/modal/app.py -- --action train --config configs/modal/train.yaml`
- Dev tuning (no training): `python -m src evaluate <best.pt> data_ext4/tusz/edf/dev --config configs/tusz_dev_tuning.yaml --output-json results/dev_metrics.json`
- Final eval (one‑shot): `python -m src evaluate <best.pt> data_ext4/tusz/edf/eval --config configs/tusz_eval_final.yaml --output-json results/final_metrics.json --output-csv-bi results/final_events.csv`

---

This document is a planning SSOT for benchmarking work. Claims remain pending until results and artifacts are released.

## Why This Strategy?

1. **Maximum Rigor**: Clean TUSZ-only baseline is unquestionable
2. **Optimized Performance**: Focused training on primary benchmark
3. **Strong Generalization Claims**: TUSZ→CHB-MIT zero-shot transfer
4. **Publication Ready**: Standard approach with novel architecture
5. **Future Work Clear**: Siena experiments become natural follow-up
