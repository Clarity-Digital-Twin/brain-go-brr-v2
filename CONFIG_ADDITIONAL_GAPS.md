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

## All Config Wiring Complete! 🎉

All configuration fields now have implementation paths from schema → runtime behavior.
Zero phantom features, zero ignored flags. Config-to-runtime wiring is 100% complete.


---

## Retired Items

### Phase 1/6 (2025-10-04)
- `experiment.save_model`, `experiment.save_best_only` – now honoured in `train/loop.py`.
- `training.gradient_accumulation_steps`, `training.mid_checkpoint_interval_s`, `training.mid_epoch_keep`, `preprocessing.bandpass`, `preprocessing.notch_freq` – fully wired.

### Additional Gaps - Completed (2025-10-04)
**P0 (Production Blockers)**:
- `logging.log_every_n_steps` – wired through train_step.py with config → env var → default precedence (train_step.py:167, loop.py:199-202)
- `evaluation.save_predictions`, `evaluation.save_plots` – implemented in val_step.py with defensive guards, wired through loop.py and evaluation.py

**P1 (Professional Best Practice)**:
- `preprocessing.normalize` – threaded into preprocess_recording (preprocess.py:17,63-70), datasets.py:41,50-51,151
- `preprocessing.montage` – apply_montage parameter added to datasets (datasets.py:42,52,147), wired through loop.py
- `data.max_samples`, `data.max_hours` – enforced in EEGWindowDataset (datasets.py:43-44,55-56,130-157), wired through loop.py (4 call sites)
- `logging.log_gradients`, `logging.log_weights` – histogram hooks in train_step.py (lines 78-143,356-395), wired with W&B integration
- `experiment.log_level` – wired into setup_logging with env > config > default precedence (loop.py:403-411)

**P2 (Quality of Life)**:
- `postprocessing.stitching.method` – threaded through metrics.py:348, honors config.stitching.method
- `postprocessing.morphology.use_gpu` – implemented with device handling (postprocess.py:153-156,181-182, streaming.py:166)

**P3 (Phantom Features - Removed)**:
- `preprocessing.use_mne` – deleted from schema (schemas.py) and both configs
- `evaluation.metrics` – deleted from schema (schemas.py:537-540) and both configs
- `warmup_schedule.residual_scale_*` – deleted all 3 fields (schemas.py:468-477) and both configs

**All wiring gaps closed. Config schema is now fully implemented.**
