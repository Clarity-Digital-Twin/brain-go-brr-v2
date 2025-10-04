# Additional Config & Env Wiring Gaps (Beyond Phase 1)

**Updated:** 2025-10-04 (post Phase-1/6 wiring). This file is the single source of truth for
configuration or environment flags that still diverge from runtime behaviour.

Priority legend:
- **P0** – Blocks production configs (local/modal train.yaml depend on it); implement immediately.
- **P1** – Professional best practice; implement before production training.
- **P2** – Quality-of-life improvement; triage after P1.
- **P3** – Phantom feature with no implementation; REMOVE from schema to prevent user confusion.

All paths below point to the current `fix/cleanup-debt` branch. When a gap is
closed, remove the entry (and update docs/tests accordingly).

---

## Preprocessing Controls

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |

## Dataset Debug Limits

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `data.max_samples`, `data.max_hours`
<br>`schemas.py:70-75` | **P1 – Implement** | Enforce limits inside `EEGWindowDataset` and `BalancedSeizureDataset` before window extraction. Respect whichever constraint triggers first and surface a warning for transparency. | Docs promise quick-debug limits; nothing happens today. Implementing keeps dev ergonomics on par with Meta/DeepMind pipelines. |

## Logging & Experiment Telemetry

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `logging.log_gradients`, `logging.log_weights`
<br>`schemas.py:587-588` | **P1 – Implement** | Add gradient histogram hooks in `train_step.py` when enabled. Both configs set `false` but framework should respect `true`. | Professional ML teams need gradient/weight monitoring for debugging. Currently no-op (grep: 0 matches). |
| `experiment.log_level`
<br>`schemas.py:569-571` | **P1 – Implement** | Feed into `setup_logging` (or `utils/logging_config.py`) so config drives root logger level. Honour env override (`BGB_LOG_LEVEL`). | Both configs set `INFO`. Production teams expect logging level in config artifact, not just env vars. Currently no-op. |

## Evaluation Surface

(All items completed)

## Post-processing Controls

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `postprocessing.stitching.{method, window_size, stride}`
<br>`schemas.py:385-391` | **P2 – Implement** | Thread into `sensitivity_at_fa_rates`/`batch_probs_to_events` so experiments can compare `overlap_add` vs `max` stitching. Validate with regression tests (TAES, FA curves). | `stitch_windows` already supports multiple methods; we just need to stop hardcoding `overlap_add`. |
| `postprocessing.morphology.use_gpu`
<br>`schemas.py:333-341` | **P2 – Implement** | Honour the flag by moving tensors to CUDA when available (and fall back gracefully). Document expected VRAM hit. | `apply_morphology` always runs on CPU today. GPU acceleration is optional but helps long recordings. |


---

## Retired Items (Handled in Phase 1/6)

- `experiment.save_model`, `experiment.save_best_only` – now honoured in `train/loop.py` (2025-10-04).
- `training.gradient_accumulation_steps`, `training.mid_checkpoint_interval_s`, `training.mid_epoch_keep`, `preprocessing.bandpass`, `preprocessing.notch_freq` – fully wired as of Phase 1–6.

Remove this section as more knobs graduate from the gap list.
