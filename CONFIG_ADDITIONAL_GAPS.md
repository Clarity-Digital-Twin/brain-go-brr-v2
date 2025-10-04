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
| `preprocessing.normalize`<br>`src/brain_brr/config/schemas.py:98` | **P1 – Implement** | Thread `normalize` into `preprocess_recording` so we can skip z-score when users provide pre-normalised data. Default remains `True`; add regression tests for both paths. | `preprocess_recording` ( `src/brain_brr/data/preprocess.py` ) always z-scores today. Trainers at Google DeepMind and Modal often need to compare with vendor-normalised inputs, so this flag must actually toggle behaviour. |
| `preprocessing.montage` (`"10-20"`/`"standard_1020"`)
<br>`schemas.py:95` → `load_edf_file` | **P1 – Implement** | Pass the flag through `EEGWindowDataset` ➝ `load_edf_file(... apply_montage=...)`. Default stays `True`, but disabling should skip `_apply_montage_best_effort`. | `load_edf_file` already accepts `apply_montage`; the datasets just ignore the config. Implementing unlocks pre-montaged corpora (e.g., hospital exports). |
| `preprocessing.use_mne`
<br>`schemas.py:105` | **P3 – Remove** | Delete from schema + docs. Pipeline hard-depends on MNE; no alternative loader exists. Pretending this is optional misleads users. | Phantom feature. Grep: zero references in `src/brain_brr/` beyond schema. One blessed loader = simpler support. |

## Dataset Debug Limits

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `data.max_samples`, `data.max_hours`
<br>`schemas.py:70-75` | **P1 – Implement** | Enforce limits inside `EEGWindowDataset` and `BalancedSeizureDataset` before window extraction. Respect whichever constraint triggers first and surface a warning for transparency. | Docs promise quick-debug limits; nothing happens today. Implementing keeps dev ergonomics on par with Meta/DeepMind pipelines. |

## Logging & Experiment Telemetry

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `logging.log_every_n_steps`
<br>`schemas.py:586` | **P0 – Implement** | Wire into `train_step.py` batch logging. Both local/modal configs set this (50/25). Env var `BGB_LOG_EVERY_N_STEPS` should override. | **BLOCKS PRODUCTION**: `configs/local/train.yaml:214` sets 50, `configs/modal/train.yaml:184` sets 25. Currently only env var takes effect (grep: 0 config references in train code). |
| `logging.log_gradients`, `logging.log_weights`
<br>`schemas.py:587-588` | **P1 – Implement** | Add gradient histogram hooks in `train_step.py` when enabled. Both configs set `false` but framework should respect `true`. | Professional ML teams need gradient/weight monitoring for debugging. Currently no-op (grep: 0 matches). |
| `experiment.log_level`
<br>`schemas.py:569-571` | **P1 – Implement** | Feed into `setup_logging` (or `utils/logging_config.py`) so config drives root logger level. Honour env override (`BGB_LOG_LEVEL`). | Both configs set `INFO`. Production teams expect logging level in config artifact, not just env vars. Currently no-op. |

## Evaluation Surface

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `evaluation.save_predictions`, `evaluation.save_plots`
<br>`schemas.py:545-546` | **P0 – Implement** | Wire into `train/loop.py` validation and `evaluate` CLI. Save `.npy` predictions and diagnostic plots to `experiment.output_dir` when enabled. | **BLOCKS PRODUCTION**: `configs/modal/train.yaml:164-165` sets both to `true`. Currently NO implementation exists (grep: 0 matches in `src/brain_brr/train/` or `src/brain_brr/eval/`). |
| `evaluation.metrics`
<br>`schemas.py:538-541` | **P3 – Remove** | Delete from schema. We always compute full TAES/AUROC suite; early-stopping depends on known metrics. User-filtering would break training logic. | Phantom feature with no implementation. Deterministic metric computation is correct design. |

## Warmup Schedule Extras

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `warmup_schedule.residual_scale_*`
<br>`schemas.py:470-478` | **P3 – Remove** | Delete `residual_scale_enabled`, `residual_scale_blocks`, `residual_scale_factor` from schema. No module consumes them; V3 residual graph doesn't support partial scaling. | Phantom feature (grep: 0 matches beyond schema). Configs set `residual_scale_enabled: false` but field shouldn't exist. Remove to prevent user confusion. |

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
