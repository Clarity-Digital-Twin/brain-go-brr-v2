# Bug Tracker (Prioritized)

Severity legend
- P0: Blocks correct training/eval or corrupts results
- P1: High risk of silent degradation; not a hard block
- P2: Medium risk/confusion; suboptimal defaults
- P3: Low risk; polish

P0 — Fixed
- Patient leakage via file‑level split
  - Fix: `split_policy: official_tusz` + patient disjointness asserts; train←train/, val←dev/.
  - Docs: `docs/tusz/tusz-splits.md`, `docs/05-training/modal.md`.
- FA‑curve threshold path inconsistent
  - Fix: set `tau_on/off` on cloned post config before eventization.
  - Docs: `docs/06-evaluation/metrics-and-taes.md#notes-and-caveats`.
- V3 NaN explosion during training
  - Root cause: Dynamic PE eigendecomposition on uninitialized adjacency
  - Fix: Disabled dynamic PE, added edge clamping, fixed optimizer parameter groups
  - Docs: `docs/08-operations/incidents/v3-nan-explosion-resolution.md`

P1 — Fixed / Hardened
- TensorBoard optional import
  - Fix: guarded import; training runs without TB installed.
- Manifest strictness for unlabeled NPZ
  - Fix: unlabeled files excluded; warning emitted.
- CLI threshold key robustness
  - Fix: tolerate keys "10", 10, 10.0 on read.

P1 — Open
- Validation loss weighting under imbalance
  - TODO: mirror `pos_weight` used in train, or report weighted+unweighted val loss.

P2 — Open/Polish
- Edge adjacency sparsification ordering
  - TODO: zero diagonal pre‑topk; threshold before top‑k; keep strictly positive edges.
- Config/docs drift (legacy fields)
  - Action: remove stale examples referencing removed schema fields in future edits.

Notes
- Local smoke: keep `batch_size ≥ 4` to avoid tiny‑batch NaNs on RTX 4090.
- Modal: run `clean-cache` once to purge pre‑fix caches; app verifies patient disjointness.

Audit trail
- Historical analyses and fixes were moved from `docs/archive/` into canonical 0X docs for long‑term maintenance.
