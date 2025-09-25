# Evaluation and TAES

Targets (clinical operating points)

- 10 FA/24h → >95% sensitivity
- 5 FA/24h → >90% sensitivity
- 1 FA/24h → >75% sensitivity

Pipeline for evaluation

- Convert per-sample probabilities `(N,T)` to events via post-processing (hysteresis + morphology + duration + merging).
- Compute metrics using event overlaps and FA counts normalized to 24 hours.
- Select thresholds (tau_on) per FA target by binary search over hysteresis settings.

Core functions

- Post-process to events: `src/brain_brr/post/postprocess.py`
- Metrics: `src/brain_brr/eval/metrics.py`
  - `batch_probs_to_events` — applies post-processing and returns per-record events
  - `find_threshold_for_fa_eventized` — binary search for `tau_on` to meet FA target
  - `fa_per_24h` — false alarms per 24 hours (pred events without overlap)
  - `sensitivity_at_fa_rates` — computes event-level sensitivity at FA targets; supports window stitching
  - `calculate_taes` — TAES scoring (overlap reward minus FA penalty)
  - `calculate_roc_auc` — AUROC; `calculate_ece` — calibration error (ECE)

Window stitching

- For record-level evaluation, overlapping windows can be stitched (`overlap_add`) to compute a continuous trace before eventization.
- Controlled by `stitch_windows` in `sensitivity_at_fa_rates`; uses `post.stitch_windows` implementation.

CLI evaluate

- `python -m src evaluate <checkpoint> <edf_dir> --config <config.yaml> [--device cuda] [--output-json out.json] [--output-csv-bi out.csv]`
- Uses `Config` from checkpoint unless `--config` is given.
- Computes metrics and can export events in CSV_BI format.
- Source: `src/brain_brr/cli/cli.py` (evaluate command)

Outputs

- Metrics: AUROC, sensitivity/specificity, TAES, sensitivity_at_{10|5|2.5|1}fa, and thresholds for each FA target.
- Events: Optional CSV_BI export with per-record events using the best threshold (10 FA/24h by default).

Notes and caveats

- Hysteresis thresholds: `tau_off` is derived as `max(0, tau_on - 0.08)` during threshold search.
- When no negatives, specificity is defined as 1.0; when no predicted positives, precision is 0.0 (stability in tests).
- TAES includes a false-alarm duration penalty (alpha=0.15) in addition to overlap reward.
 - Threshold path correctness: FA‑curve search sets `tau_on/off` on a cloned post config before eventization, avoiding deprecated threshold arguments.
