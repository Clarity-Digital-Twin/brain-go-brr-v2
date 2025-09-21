# PHASE 5 — Evaluation, Scoring, and Benchmarking (Iron‑Clad, TDD) ✅ COMPLETE

## 🎯 Phase Goal
Establish a clinically grounded, reproducible evaluation pipeline that turns per‑timestep probabilities into events and reports TAES and sensitivity at target FA/24h operating points on TUH/CHB‑MIT and external benchmarks.

Success means: a deterministic, well‑tested evaluation stack with clear CLI, CI integration, and artifacts suitable for epilepsybenchmarks.com submission.

## 🔭 Scope & Status ✅ COMPLETE
- ✅ Implementation lives under `src/brain_brr/` (older `src/experiment/*` names are historical)
- ✅ Phase 4 APIs integrated (hysteresis, morphology, stitching)
- ✅ Export functionality in `src/brain_brr/events/export.py` (CSV_BI, JSON)
- ✅ CLI evaluation command in `src/brain_brr/cli/cli.py` (runnable via `python -m src`) with export options
- ✅ Evaluation is CPU‑safe and does not require CUDA

All target APIs defined in this document have been implemented and tested.

## 📏 Metrics (Clinical)
- TAES (Time‑Aligned Event Scoring): primary temporal alignment score in [0,1].
- Sensitivity@FA: event‑level sensitivity at FA/24h ∈ {10, 5, 2.5, 1}.
- FA curve: sensitivity vs FA/24h curve for multiple targets.
- AUROC (sample‑level): for sanity and regression tracking.

Additional metrics (implemented):
- ✅ Calibration (ECE) - Expected Calibration Error
- ✅ PR‑AUC - Precision-Recall Area Under Curve
- ⏳ Per‑patient TAES distribution (future work)

## 🧩 Eventization Semantics
Eventization must follow Phase 4: hysteresis (τ_on=0.86, τ_off=0.78), morphology (opening→closing), min/max duration, and optional stitching (for full‑record inference). Conventions:
- Thresholding: ≥ τ_on to enter; < τ_off to exit; stable windows min_onset/min_offset when added.
- Morphology kernels are odd; defaults documented in Phase 4.
- Min duration default 3.0s; max duration default 600.0s with segmentation.
- Convert masks to intervals by diff on zero‑padded mask.

Note: Evaluation currently uses the corrected duration formula for overlapped windows
to compute FA/24h time (for N windows: duration = (N−1)×stride + window_size). Full
cross‑window event stitching (merging events crossing window boundaries) is available
in Phase 4 and can be threaded into evaluation as needed.

## 🎛️ Threshold Search & FA Targets
- Purpose: choose hysteresis τ_on such that FA/24h ≈ target (conservative bisection). Use τ_off = max(0, τ_on − Δ) with Δ≈0.08.
- Procedure:
  1) For a candidate τ_on, set τ_off accordingly and post‑process probs → predicted events.
  2) Compute FA/24h across the corpus using per‑record durations (stitching if needed).
  3) Binary search on τ_on ∈ [Δ, 1] until tolerance (1e‑4) or max_iters.
- Sensitivity@FA: once τ_on found, compute event‑level sensitivity vs references.

Implementation details:
- The legacy `threshold` parameter in `evaluate.batch_probs_to_events(...)` is deprecated.
  Evaluation selects τ_on via binary search to meet the FA/24h target (conservative),
  with τ_off = max(0, τ_on − Δ). Hysteresis values in the config act as defaults/initial
  seeds; evaluation is not limited to fixed config thresholds.
- TAES false‑alarm penalty weight defaults to α = 0.15 in code; this balances temporal
  alignment against spurious predictions.

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
  - Binary search on τ_on converges (monotone FA vs τ_on) with mock traces.
  - Sensitivity@FA matches ground truth under simple scenarios.

Integration tests:
- `tests/test_integration_eval.py`:
  - End‑to‑end: probs → postprocess (Phase 4) → events → metrics (satisfy invariants; e.g., τ_on,high ≥ τ_on,low; FA monotonicity).
  - Window stitching path: reconstruct full timeline and verify boundary behavior.

Export/Compliance tests:
- `tests/test_export.py`:
  - CSV_BI header format, column schema, sorting, duration consistency.

Performance determinism:
- Seed test to assert reproducible metrics across invocations with same inputs.

## 🧩 Target APIs (Python)

Already present (to standardize):
- `evaluate.batch_masks_to_events(masks, fs)`
- `evaluate.batch_probs_to_events(probs, post_cfg, fs, threshold)` [Deprecated: `threshold` ignored]
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

Outputs enhancement:
- `evaluate.evaluate_predictions(...)` returns a `thresholds` table mapping FA target → τ_on
  used for sensitivity evaluation (e.g., `{ "10": 0.8725, "5": 0.9031, ... }`).

## 🧵 CLI & Make Targets

CLI examples:
- Evaluate (dev tuning):
  `python -m src evaluate /path/to/checkpoint.pt data_ext4/tusz/edf/dev --config configs/tusz_dev_tuning.yaml --output-json results/dev_metrics.json`
- Final eval:
  `python -m src evaluate /path/to/checkpoint.pt data_ext4/tusz/edf/eval --config configs/tusz_eval_final.yaml --output-json results/final_metrics.json`

Exports: use library functions in `src/brain_brr/events/export.py` (CSV_BI/JSON).

Make (optional additions):
- `make eval-dev` → run evaluation on validation split; write metrics JSON + plots.
- `make export-dev` → export CSV_BI for dev split.

## 📦 Artifacts & Logging
- Metrics JSON: TAES, AUROC, sensitivity@{10,5,2.5,1}fa, FA curve.
- Threshold table: FA target → τ_on.
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
- Non‑monotone FA vs τ_on under rare post‑proc combos → enforce monotonicity by increasing τ_on when binary search oscillates; log warning.
- Stitching boundary artifacts → tests with partial windows and off‑by‑one checks.
- Dataset header quirks (TUH EDF) → addressed in `TUSZ_EDF_HEADER_FIX.md`; ensure eval respects fixed channel order.
- CPU/GPU divergence → Evaluation runs CPU‑deterministic; GPU only affects model inference (Phase 2/3).

## 📌 References
- NEDC TAES specifications and scoring methodology.
- epilepsybenchmarks.com submission formats.
- Internal Phase docs: PHASE2.* (model), PHASE3_TRAINING_PIPELINE.md, PHASE4_POSTPROCESSING.md.
> Note: This Phase doc is being replaced by component‑oriented docs. See components/evaluation.md for the canonical, code‑aligned reference.
