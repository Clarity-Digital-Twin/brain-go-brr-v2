# Research Benchmark & Ablation Plan — Brain‑Go‑Brr v2

Last updated: 2025‑09‑20

Status: Planning document for rigorous, publishable benchmarking of a Bi‑Mamba‑2 + U‑Net + ResCNN seizure detector. Targets SzCORE/NEDC‑style event metrics on TUSZ and cross‑dataset generalization.

## Purpose

- Establish a clinically relevant, reproducible benchmark suite for event‑based seizure detection with O(N) sequence modeling.
- Provide an ablation and reporting protocol suitable for EMBC/JBHI/ML4H or NeurIPS Datasets & Benchmarks (D&B).
- Separate goals (pending) from claims (current code functionality).

## Datasets & Splits

- TUSZ EEG Seizure Corpus (v2.x)
  - train/: model training only
  - dev/: hyperparameter/threshold tuning
  - eval/: final one‑shot test; never used in training/tuning
  - Pathing: `data_ext4/tusz/edf/{train,dev,eval}` (ext4‑mounted volume recommended)
- CHB‑MIT: cross‑site generalization (patient‑level disjoint split)
- Optional: TUAB/TUAR for pretraining or auxiliary validation
- Notes:
  - Channel synonyms mapping enforced (T7→T3, T8→T4, P7→T5, P8→T6) and canonical 10‑20 order preserved (see `src/brain_brr/constants.py`).
  - Record and document dataset versions and any repairs applied.

## Metrics (Event‑Level + Operating Curves)

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

- Hardware profiles
  - Local WSL2 (WSL‑safe): `configs/tusz_train_wsl2.yaml` (num_workers=0)
  - Modal A100‑80GB: `configs/tusz_train_a100.yaml`
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

- Local smoke: `python -m src train configs/smoke_test.yaml`
- Local full (WSL2‑safe): `python -m src train configs/tusz_train_wsl2.yaml`
- Modal smoke: `modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml`
- Modal A100: `modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml`
- Dev tuning (no training): `python -m src evaluate <best.pt> data_ext4/tusz/edf/dev --config configs/tusz_dev_tuning.yaml --output-json results/dev_metrics.json`
- Final eval (one‑shot): `python -m src evaluate <best.pt> data_ext4/tusz/edf/eval --config configs/tusz_eval_final.yaml --output-json results/final_metrics.json --output-csv-bi results/final_events.csv`

---

This document is a planning SSOT for benchmarking work. Claims remain pending until results and artifacts are released.

