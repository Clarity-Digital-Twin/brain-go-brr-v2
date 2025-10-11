# Bug Tracker (Prioritized)

Severity legend
- P0: Blocks correct training/eval or corrupts results
- P1: High risk of silent degradation; not a hard block
- P2: Medium risk/confusion; suboptimal defaults
- P3: Low risk; polish

P0 — Fixed
- Patient leakage via file‑level split
  - Fix: Loader enforces official train/dev splits on read; legacy `split_policy` / `validation_split` config fields were removed in V4.
  - Docs: `docs/tusz/tusz-splits.md`, `docs/05-training/modal.md`.
- Config rename fallout after BiMamba2/FLA split
  - Fix: Tests now reference `train_bimamba.yaml` / `train_fla.yaml` pairs instead of deleted monolithic configs (`tests/unit/models/test_nan_robustness.py`).
  - Modal CLI defaults and README examples updated to `smoke_bimamba.yaml` / `train_bimamba.yaml`; Docker `CMD` was aligned to the same baseline (`deploy/modal/app.py`, `deploy/modal/README.md`, `Dockerfile`).
  - Status: Commands tutorial, tooling defaults, and CI runs no longer raise `FileNotFoundError` when configs are omitted.
- FA‑curve threshold path inconsistent
  - Fix: set `tau_on/off` on cloned post config before eventization.
  - Docs: `docs/06-evaluation/metrics-and-taes.md#notes-and-caveats`.
- V3 NaN explosion during training
  - Root causes: dynamic PE on poorly initialized adjacency; optimizer hygiene (weight decay on norms); edge‑stream amplification
  - Fix: optimizer parameter groups (no decay on norms/bias), edge clamping (env‑gated), PE safeguards (degree/diag regularization, fp32 eigens, cached fallback)
  - Fallback: on RTX 4090, set `use_dynamic_pe: false` if NaNs persist
  - Docs: `docs/08-operations/incidents/v3-nan-explosion-resolution.md`, `docs/08-operations/troubleshooting.md`

P1 — Fixed / Hardened
- TensorBoard optional import
  - Fix: guarded import; training runs without TB installed.
- Manifest strictness for unlabeled NPZ
  - Fix: unlabeled files excluded; warning emitted.
- CLI threshold key robustness
  - Fix: tolerate keys "10", 10, 10.0 on read.
- Constants centralization (v3.5.0)
  - Fix: All clinical constants (event durations, morphology kernels, threshold search bounds, merge gaps) moved to `src/brain_brr/constants.py`; schema defaults import and use these constants.
  - Affected: `schemas.py` (DurationConfig, EventsConfig, MorphologyConfig), `eval/helpers/false_alarm.py` (threshold search bounds)
- Mid-epoch resume continuity
  - Fix: `StatefulDataLoader` state and `global_step`/`batch_idx` offsets are stored in checkpoints (`src/brain_brr/train/loop.py`, `train_step.py`, `checkpoint.py`) so resumes pick up the exact batch with correct warmup and W&B step counters.
  - Impact: No more 1–2 h of replay on Modal timeout; warmup schedules stay on track after resumes.

P2 — Fixed / Cleanup
- Pydantic warning sweep
  - Fix: All optional config fields now use `Annotated[..., Field(...)]` so Pydantic v2.12 no longer emits `UnsupportedFieldAttributeWarning` (`src/brain_brr/config/schemas.py`).
  - Impact: `make q` runs warning-free; config validation stays strict with zero cosmetic noise.

P1 — Open
- **Validation loss weighting under imbalance** (ACTIVE)
  - Issue: Training uses `pos_weight` (train_step.py:141), validation doesn't (val_step.py:239)
  - Impact: Train/val loss not directly comparable due to different class weighting
  - TODO: Mirror `pos_weight` in validation OR report both weighted+unweighted val loss
  - Severity: P1 (affects interpretability, not correctness)

P2 — Open/Polish
- Edge adjacency sparsification ordering
  - TODO: zero diagonal pre‑topk; threshold before top‑k; keep strictly positive edges.
- Config/docs drift (legacy fields)
  - Action: remove stale examples referencing removed schema fields in future edits.
- Dataset batch contract typing
  - TODO: add `TypedDict` (or `Protocol`) for the dataset batch dictionary to prevent regressions to tuple outputs.

Notes
- Local smoke: keep `batch_size ≥ 4` to avoid tiny‑batch NaNs on RTX 4090.
- Modal: run `clean-cache` once to purge pre‑fix caches; app verifies patient disjointness.

Audit trail
- Historical analyses and fixes were moved from `docs/archive/` into canonical 0X docs for long‑term maintenance.
