# Additional Config & Env Wiring Gaps (Beyond Phase 1)

**Updated:** 2025-10-04 (post Phase-1/6 wiring). This file is the single source of truth for
configuration or environment flags that still diverge from runtime behaviour.

Priority legend:
- **P0** – Blocks production; fix before next train run.
- **P1** – High leverage; schedule for the next engineering sprint.
- **P2** – Valuable but not urgent; triage once P1 items ship.
- **P3** – Prefer to deprecate/remove unless a strong product requirement appears.

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
<br>`schemas.py:105` | **P3 – Deprecate** | Remove from schema + docs unless we add a non-MNE loader (e.g., pyEDFlib fallback). Today the pipeline hard-depends on MNE; pretending this is optional misleads users. | No references in code beyond the schema/documentation. Eliminating the flag simplifies support (one blessed loader). |

## Dataset Debug Limits

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `data.max_samples`, `data.max_hours`
<br>`schemas.py:70-75` | **P1 – Implement** | Enforce limits inside `EEGWindowDataset` and `BalancedSeizureDataset` before window extraction. Respect whichever constraint triggers first and surface a warning for transparency. | Docs promise quick-debug limits; nothing happens today. Implementing keeps dev ergonomics on par with Meta/DeepMind pipelines. |

## Logging & Experiment Telemetry

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `logging.log_every_n_steps`, `logging.log_gradients`, `logging.log_weights`
<br>`schemas.py:583-588` | **P1 – Implement** | Plumb these into `constants.LOG_EVERY_N_STEPS` / `train_step` instrumentation and gradient histogram hooks. Treat env vars as overrides (config → env → default). | Currently only env vars (`BGB_LOG_EVERY_N_STEPS`, etc.) take effect. Hooking configs restores truthfulness and allows per-run tuning in YAML. |
| `experiment.log_level`
<br>`schemas.py:569-571` | **P1 – Implement** | Feed into `setup_logging` so CLI/config drives root logger level. Honour env override precedence (`BGB_LOG_LEVEL`). | Production teams expect logging level to travel with the config artifact; today it silently ignores the field. |

## Evaluation Surface

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `evaluation.metrics`
<br>`schemas.py:538-545` | **P3 – Deprecate** | Remove from schema/docs (or reintroduce later as a reporting filter). We always compute the full TAES/AUROC suite, and early-stopping depends on a known metric set. | Allowing users to drop metrics would desynchronise training logic. Prefer deterministic outputs. |
| `evaluation.save_predictions`, `evaluation.save_plots`
<br>`schemas.py:545-547` | **P2 – Implement** | Wire into `train loop`/`evaluate` CLI so that, when enabled, we persist `.npy` predictions and diagnostic plots to `experiment.output_dir`. | These toggles are visible in configs but no-op. Persisting artifacts aligns with industry eval workflows. |

## Warmup Schedule Extras

| Field(s) | Priority | Recommendation | Notes |
| --- | --- | --- | --- |
| `warmup_schedule.residual_scale_*`
<br>`schemas.py:469-478` | **P3 – Deprecate** | Drop the residual scaling flags for now. No module consumes them and adding partial scaling on the V3 residual graph requires careful stability analysis. | Keeping dormant flags invites confusion; reintroduce if/when we implement residual warmups. |

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
