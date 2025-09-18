# PHASE 5 — Evaluation, Scoring, and Benchmarking (Iron‑Clad, TDD)

## 🎯 Phase Goal
Establish a clinically grounded, reproducible evaluation pipeline that turns per‑timestep probabilities into events and reports TAES and sensitivity at target FA/24h operating points on TUH/CHB‑MIT and external benchmarks.

Success means: a deterministic, well‑tested evaluation stack with clear CLI, CI integration, and artifacts suitable for epilepsybenchmarks.com submission.

## 🔭 Scope & Status
- Current interim code lives in `src/experiment/evaluate.py` and basic tests in `tests/test_evaluate.py`.
- Phase 4 specification (post‑processing) is documented in `PHASE4_POSTPROCESSING.md`. Phase 5 consumes those APIs and evaluates outputs.
- Mamba CUDA/CPU dispatch is documented in README/AGENTS; evaluation is CPU‑safe and does not require CUDA.

This document defines the target end‑state APIs, metrics, datasets/splits, and a TDD plan to complete Phase 5 with high assurance.

## 📏 Metrics (Clinical)
- TAES (Time‑Aligned Event Scoring): primary temporal alignment score in [0,1].
- Sensitivity@FA: event‑level sensitivity at FA/24h ∈ {10, 5, 2.5, 1}.
- FA curve: sensitivity vs FA/24h curve for multiple targets.
- AUROC (sample‑level): for sanity and regression tracking.

Optional (stretch):
- Calibration (ECE), PR‑AUC, per‑patient TAES distribution.

## 🧩 Eventization Semantics
Eventization must follow Phase 4: hysteresis (τ_on=0.86, τ_off=0.78), morphology (opening→closing), min/max duration, and optional stitching (for full‑record inference). Conventions:
- Thresholding: strictly > τ_on to enter; strictly < τ_off to exit; stable windows min_onset/min_offset when added.
- Morphology kernels are odd; defaults documented in Phase 4.
- Min duration default 3.0s; max duration default 600.0s with segmentation.
- Convert masks to intervals by diff on zero‑padded mask.

## 🎛️ Threshold Search & FA Targets
- Purpose: choose hysteresis τ_on such that FA/24h ≈ target (conservative bisection). Use τ_off = max(0, τ_on − Δ) with Δ≈0.08.
- Procedure:
  1) For a candidate τ_on, set τ_off accordingly and post‑process probs → predicted events.
  2) Compute FA/24h across the corpus using per‑record durations (stitching if needed).
  3) Binary search on τ_on ∈ [Δ, 1] until tolerance (1e‑4) or max_iters.
- Sensitivity@FA: once τ_on found, compute event‑level sensitivity vs references.

## 📚 Datasets & Splits
- Train: TUH EEG Seizure Corpus
- Dev/Validation: CHB‑MIT (pediatric generalization)
- Final: epilepsybenchmarks.com

Split discipline:
- Patient‑level splits where applicable.
- Record duration tracked to compute FA/24h accurately.
- Store split manifests for reproducibility (CSV/JSON under `configs/` or `results/manifests/`).

## 🔗 Pipeline Overview (Eval Time)
1) Load model outputs (B×T probs) and labels (B×T masks) or run model inference.
2) Post‑process to masks → events (Phase 4 APIs).
3) Threshold search to match FA targets.
4) Compute TAES, sensitivity@FA, FA curve, AUROC.
5) Export artifacts: JSON metrics, CSV_BI events, plots, and threshold table.

## 🧪 TDD Build‑Out Plan

Unit tests (small, deterministic):
- `tests/test_evaluate.py` (extend):
  - TAES correctness on synthetic cases (single/multiple overlaps, FP penalty).
  - FA/24h computation with controlled durations and gaps.
  - Binary search threshold converges (monotone FA vs θ) with mock traces.
  - Sensitivity@FA matches ground truth under simple scenarios.

Integration tests:
- `tests/test_integration_eval.py`:
  - End‑to‑end: probs → postprocess (Phase 4) → events → metrics (satisfy invariants; e.g., θ_high ≥ θ_low, FA monotonicity).
  - Window stitching path: reconstruct full timeline and verify boundary behavior.

Export/Compliance tests:
- `tests/test_export.py`:
  - CSV_BI header format, column schema, sorting, duration consistency.

Performance determinism:
- Seed test to assert reproducible metrics across invocations with same inputs.

## 🧩 Target APIs (Python)

Already present (to standardize):
- `evaluate.batch_masks_to_events(masks, fs)`
- `evaluate.batch_probs_to_events(probs, post_cfg, fs, threshold)`
- `evaluate.fa_per_24h(pred_events, ref_events, total_hours)`
- `evaluate.calculate_taes(pred_events, ref_events, alpha=0.15)`
- `evaluate.find_threshold_for_fa_eventized(probs, post_cfg, ref_events, fa_target, total_hours, fs)`
- `evaluate.sensitivity_at_fa_rates(probs, labels, fa_targets, post_cfg, sampling_rate)`
- `evaluate.evaluate_predictions(probs, labels, fa_rates, post_cfg, sampling_rate)`

Phase 4 dependencies (implemented):
- `postprocess.apply_hysteresis`, `apply_morphology`, `filter_duration`, `stitch_windows`
- `events.mask_to_events`, `events.merge_events`, `events.calculate_event_confidence`
- `export.export_csv_bi`

During Phase 5, swap in the Phase 4 APIs internally (keep current functions as adapters for test stability until migration is complete).

## 🧵 CLI & Make Targets

CLI (suggested):
- `python -m src.cli evaluate --config configs/production.yaml --split dev --out results/metrics/dev.json`
- `python -m src.cli export --pred results/preds/*.npy --out results/exports/dev_csv_bi/`

Make (optional additions):
- `make eval-dev` → run evaluation on validation split; write metrics JSON + plots.
- `make export-dev` → export CSV_BI for dev split.

## 📦 Artifacts & Logging
- Metrics JSON: TAES, AUROC, sensitivity@{10,5,2.5,1}fa, FA curve.
- Threshold table: FA target → θ.
- CSV_BI folder with per‑record events.
- Plots: FA curve, ROC, calibration (optional).
- Manifest: data split listing, duration totals.
- Metadata: config hash, git commit, timestamp, env (CUDA, torch, mamba_ssm present/absent).

## 🔁 Reproducibility & Determinism
- Set torch/np/python seeds and deterministic flags.
- Cache keys based on config hash (noted in AGENTS.md critical notes).
- For CI CPU runs where gpu extra may be installed, set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to avoid GPU path.

## ✅ Definition of Done (Phase 5)
- Unit + integration tests passing; coverage ≥ 90% for evaluation modules.
- `make q` passes (ruff + mypy strict) after final refactor.
- CLI evaluates dev split and writes complete artifacts.
- CSV_BI exporter matches Temple format; spot‑checked in tests.
- Documentation (this file + README link) up‑to‑date and concise.

## ⚠️ Risks & Mitigations
- Non‑monotone FA vs θ under rare post‑proc combos → enforce monotonicity by increasing θ when binary search oscillates; log warning.
- Stitching boundary artifacts → tests with partial windows and off‑by‑one checks.
- Dataset header quirks (TUH EDF) → addressed in `TUSZ_EDF_HEADER_FIX.md`; ensure eval respects fixed channel order.
- CPU/GPU divergence → Evaluation runs CPU‑deterministic; GPU only affects model inference (Phase 2/3).

## 📌 References
- NEDC TAES specifications and scoring methodology.
- epilepsybenchmarks.com submission formats.
- Internal Phase docs: PHASE2.* (model), PHASE3_TRAINING_PIPELINE.md, PHASE4_POSTPROCESSING.md.
