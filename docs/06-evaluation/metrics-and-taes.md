# Evaluation and TAES

Targets (clinical operating points)

- 10 FA/24h → >95% sensitivity
- 5 FA/24h → >90% sensitivity
- 1 FA/24h → >75% sensitivity

Pipeline for evaluation

- Convert per-sample probabilities `(N,T)` to events via post-processing (hysteresis + morphology + duration + merging).
- Stitch windows per recording using `file_id` and `window_start_s` emitted by the datasets; overlapping regions are averaged before eventization.
- Compute metrics using event overlaps and FA counts normalized to 24 hours.
- Select thresholds (`tau_on`) per FA target by binary search over hysteresis settings.

Core functions

- Post-process to events: `src/brain_brr/post/postprocess.py`
- Metrics: `src/brain_brr/eval/metrics.py`
  - `batch_probs_to_events` — applies post-processing and returns per-record events
  - `find_threshold_for_fa_eventized` — binary search for `tau_on` to meet FA target
  - `fa_per_24h` — false alarms per 24 hours (pred events without overlap)
  - `sensitivity_at_fa_rates` — computes event-level sensitivity at FA targets; supports window stitching
  - `calculate_taes` — TAES scoring (overlap reward minus FA penalty)
  - `calculate_ece` — calibration error (ECE)
  - AUROC computed via sklearn's `roc_auc_score` (no wrapper function)

Timeline metadata and stitching

- `EEGWindowDataset`, `BalancedSeizureDataset`, and `ValidationDataset` return dictionaries with `window`, `label`, `file_id`, and `window_start_s`.
- `evaluate_predictions` groups windows by `file_id`, sorts by `window_start_s`, rebuilds continuous traces, and averages overlapping samples before post-processing.
- The per-record view supplies reference/predicted events along with the total monitored hours, ensuring FA/24 h and TAES honour true recording durations.

Streaming evaluation

- Validation aggregates metrics per recording instead of retaining the entire dev split in memory (22 GB → ~5 GB peak usage on A100).
- Local hosts (64 GB RAM) no longer risk OOM during metric calculation.
- Metrics are emitted after each recording batch, keeping progress logs responsive.

CLI evaluate

- `python -m src evaluate <checkpoint> <edf_dir> --config <config.yaml> [--device cuda] [--output-json out.json] [--output-csv-bi out.csv] [--dry-run]`
- Config resolution order: `--config` takes precedence; otherwise uses checkpoint-embedded config if present and not `None`; else exits with an error.
- Exits with an error if no EDF files are found under `<edf_dir>`.
- Computes metrics and can export events in CSV_BI format.
- Source: `src/brain_brr/cli/cli.py` (evaluate command)

Outputs

- Metrics: AUROC, sensitivity/specificity, TAES, sensitivity_at_{10|5|2.5|1}fa, and thresholds for each FA target.
- Events (CSV_BI export): Uses the best threshold (10 FA/24h by default). Current implementation emits a single CSV for the evaluation run with stride-aware timing (60s windows with 10s stride). It does not yet group outputs per recording file.

Mathematical definitions

```text
TAES = base_score - penalty

base_score = (1/|R|) * Σᵣ min(1, overlap_duration(r, P) / duration(r))
penalty    = α * (fp_duration / total_pred_duration)
```

- `R` = reference events, `P` = predicted events, `α = 0.15`.
- Range: `[0, 1]`; rewards good overlap while penalising false-alarm duration.

False alarms per 24 hours:

```text
FA/24h = (FA_count / total_hours) * 24

FA_count = |{p ∈ P : overlap(p, r) = 0 for all r ∈ R}|
```

Expected calibration error (ECE):

```text
ECE = Σᵢ | accuracy(Bᵢ) - confidence(Bᵢ) | * P(Bᵢ)
```

- Bins `Bᵢ` partition probability space (default `n = 15`).
- `accuracy(Bᵢ)` = mean label for samples in `Bᵢ`.
- `confidence(Bᵢ)` = mean probability for samples in `Bᵢ`.
- `P(Bᵢ)` = fraction of samples in that bin.

Notes and caveats

- Hysteresis thresholds: `tau_off` is derived as `max(0, tau_on - 0.08)` during threshold search.
- When no negatives, specificity is defined as 1.0; when no predicted positives, precision is 0.0 (stability in tests).
- TAES includes a false-alarm duration penalty (alpha=0.15) in addition to overlap reward.
- Threshold path correctness: FA‑curve search sets `tau_on/off` on a cloned post config before eventization, avoiding deprecated threshold arguments.
