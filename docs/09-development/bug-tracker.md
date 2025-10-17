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
- GNN positional encoding buffer incompatibility
  - Fix: `gnn.last_valid_pe` now registers with a placeholder tensor so checkpoints always contain the buffer; loader skips legacy mismatches before calling `load_state_dict`.
  - Impact: Mid-epoch resumes no longer crash with “Unexpected key gnn.last_valid_pe” when a fresh model is initialized.
- Config rename fallout after BiMamba2/FLA split
  - Fix: Tests now reference `train_bimamba.yaml` / `train_fla.yaml` pairs instead of deleted monolithic configs (`tests/unit/models/test_nan_robustness.py`).
  - Modal CLI defaults and README examples updated to `smoke_bimamba.yaml` / `train_bimamba.yaml`; Docker `CMD` was aligned to the same baseline (`deploy/modal/app.py`, `deploy/modal/README.md`, `Dockerfile`).
  - Status: Commands tutorial, tooling defaults, and CI runs no longer raise `FileNotFoundError` when configs are omitted.
- WSL2 SIGBUS crash (FLA stack, batch ~2900)
  - Root causes: (1) Windows host driver 572.xx (Ada instability) and (2) mmap cache on `/mnt/d` via 9P invalidating pages under AVX2 copy.
  - Fix: Upgrade to NVIDIA driver 581.42 **and** migrate `cache/tusz_mmap` to a native ext4 volume. Guide: `docs/08-operations/wsl2-sigbus-fix.md`.
  - Impact: Local FLA training now progresses past batch 5,400 with zero SIGBUS events; dual-stack production (Modal BiMamba2 + local FLA) confirmed.
- RNG state device mismatch
  - Fix: Checkpoint loads leave RNG tensors on CPU before restoring; added regression coverage in `tests/unit/train/test_checkpoint_rng_device.py`.
  - Impact: CUDA resumes no longer raise “RNG state must be a torch.ByteTensor”; deterministic restart is preserved.
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

P1 — Fixed
- **Validation loss weighting consistency** (RESOLVED v3.8.0+)
  - Historical issue: Training/validation used different loss weighting schemes
  - Fix: Both now use identical focal loss (focal_alpha/gamma) - no pos_weight anywhere
  - Code: train_step.py:319-328 and val_step.py:474-481 use same focal loss formula
  - Impact: Train/val losses are now directly comparable (same weighting)

P2 — Open/Polish
- Edge adjacency sparsification ordering
  - TODO: zero diagonal pre‑topk; threshold before top‑k; keep strictly positive edges.
- Config/docs drift (legacy fields)
  - Action: remove stale examples referencing removed schema fields in future edits.
- Dataset batch contract typing
  - TODO: add `TypedDict` (or `Protocol`) for the dataset batch dictionary to prevent regressions to tuple outputs.

Notes
- Local: Production batch_size=8 (optimized for RTX 4090), smoke tests use batch_size=4 minimum to avoid tiny-batch NaNs.
- Modal: run `clean-cache` once to purge pre‑fix caches; app verifies patient disjointness.
- Documented non-issues (tracked for completeness):
  - pytorch-tcn CPU hang scares (P2.2) — production uses the internal MinimalTCN backend; external pytorch-tcn stays optional (`docs/archive/REMAINING_ISSUES_INVESTIGATION.md`).
  - Migration regression tests (P2.3) — coverage gap only; enable when bandwidth allows.

Audit trail
- Historical analyses and fixes were moved from `docs/archive/` into canonical 0X docs for long‑term maintenance.
