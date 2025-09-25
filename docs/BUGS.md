Brain-Go-Brr v2.6/V3 — Risk Audit and Bug List

Status legend
- Fixed: Verified fix merged and exercised in smoke
- Open: Not yet addressed
- Next: Planned/low-risk, queued after training restarts

Severity legend
- P0: Blocks correct training/evaluation or corrupts results
- P1: High risk of silent degradation or wrong metrics; not a hard block
- P2: Medium risk, confusing or suboptimal defaults; can mislead users
- P3: Low risk, polish and ergonomics

P0 blockers
- Train/val split risks leakage and bias — Fixed
  - Problem: Previous file-level alphabetical split mixed patients across splits.
  - Fix: Added official TUSZ split resolver and enforced patient-disjointness.
  - Where: src/brain_brr/data/tusz_splits.py; src/brain_brr/train/loop.py (split_policy="official_tusz"); configs/* set to parent edf dir
  - Notes: Prints patient/file counts and asserts zero overlap. Local and Modal configs now point to parent `.../edf` with `split_policy: official_tusz`.

- Evaluation FA curve uses wrong thresholds — Fixed
  - Fix: Clone post_cfg, set tau_on/off per FA target, and pass that through; no reliance on deprecated arg.
  - Where: src/brain_brr/eval/metrics.py (sensitivity_at_fa_rates, find_threshold_for_fa_eventized)

P1 issues
- Evaluate CLI defaults to best threshold lookup by string key; brittle if types change
  - Status: Fixed (robust key coercion)
  - Problem: It expects thresholds["10"] (string key). The thresholds dict is currently stringified, but future changes to keep numeric keys will silently fall back to 0.86.
  - Where: src/brain_brr/cli/cli.py: around thresholds export
  - Fix: Coerce keys; accept "10", 10, or 10.0.
  - Impact: Wrong export threshold if key types differ; inconsistent event exports.

- Validation loss unweighted wrt extreme imbalance
  - Status: Open
  - Problem: validate_epoch() uses plain BCEWithLogitsLoss without pos_weight. For highly imbalanced data this can make val_loss misleading.
  - Where: src/brain_brr/train/loop.py:812–828 (criterion construction)
  - Impact: Val loss comparisons and LR/scheduler heuristics based on it are biased; can hide collapse.
  - Fix: Mirror class weighting used in train (pos_weight) or report both weighted and unweighted metrics.

 - Non-random split order can encode confounds even with fixed seed — Mitigated
  - Problem: Sorting filenames and then slicing induces a fixed bias if filenames encode site/date/patient.
  - Where: Custom split path in src/brain_brr/train/loop.py (seeded shuffle now applied)
  - Impact: Distribution shift between splits; over/underestimation of generalization.
  - Fix: Official policy avoids file-level splitting entirely; custom mode now shuffles deterministically before splitting. Prefer `split_policy: official_tusz`.

P2 issues
- TCN channels in config are ignored by implementation
  - Status: Fixed (schema cleaned)
  - Problem: ModelConfig.tcn.channels is accepted and shown in summaries but TCNEncoder constructs channels internally and never reads that field.
  - Where: src/brain_brr/config/schemas.py (TCNConfig) — removed unused channels field; YAMLs updated
  - Impact: Config ≠ behavior; wasted tuning time; confusion for reviewers.
  - Fix: Remove misleading field to avoid bait-and-switch.

- TensorBoard likely not installed in base deps, but imported unconditionally
  - Status: Fixed (lazy import)
  - Problem: from torch.utils.tensorboard import SummaryWriter at import time; tensorboard pip isn’t in [project] deps.
  - Where: src/brain_brr/train/loop.py (optional import with HAS_TENSORBOARD)
  - Impact: Runtime import error on fresh envs unless dev deps are installed; training fails early.
  - Fix: Guarded import; prints install hint if missing.

- Manifest-based balancing silently assumes label presence in NPZ
  - Status: Fixed (strict mode)
  - Problem: scan_existing_cache() treats any NPZ without labels as fully background and floods manifest with no-seizure entries.
  - Where: src/brain_brr/data/cache_utils.py
  - Impact: If labels failed to save for a subset, sampling skews heavily, reducing effective positives.
  - Fix: Warn and exclude unlabeled NPZs from manifest; avoids flooding negatives.

- Edge top-k can select zeros/self-edges after thresholding
  - Status: Open
  - Problem: assemble_adjacency() top-k is applied before threshold; with many small weights this may keep near-zeros; diagonal is not explicitly excluded before top-k.
  - Where: src/brain_brr/models/edge_features.py
  - Impact: Graph contains weak/uninformative edges; unnecessary compute and reduced SNR.
  - Fix: Zero the diagonal pre-topk; apply threshold before top-k; ensure top-k is taken over strictly positive weights.

- Heavy sampler probing on uncached datasets
  - Status: Next
  - Problem: create_balanced_sampler() can probe up to 20k windows; if cache isn’t built yet, this triggers expensive I/O and preprocessing.
  - Where: src/brain_brr/train/loop.py (sampler creation path)
  - Impact: Extremely slow startup, potential timeouts in cloud environments.
  - Fix: Gate probing on cache presence; or require manifest first and use BalancedSeizureDataset by default when balancing is true.

P3 issues (polish)
- Deprecated morphology options in tests/configs
  - Problem: Some tests pass morphology={"kernel_size": ..., "operation": ...}, but MorphologyConfig uses opening/closing kernels; the legacy fields are ignored.
  - Impact: Confusing for readers; doesn’t affect correctness in current tests.
  - Fix: Remove legacy fields from tests/config examples or shim them in config parsing.

- Mixed naming of constants and config
  - Problem: constants.py defines sampling/window/stride; DataConfig duplicates them. Multiple sources of truth can drift.
  - Fix: Centralize in config and import into constants (or vice versa) to ensure consistency.

Verification and next steps
- Splits: Run smoke with `split_policy: official_tusz` and verify patient overlap = 0 (printed at startup).
- Modal configs: Updated to parent `/data/edf` + `official_tusz` (smoke/train). Ensure Modal volume has dev/ present.
- FA curve: Visual checks now align with evaluate_predictions; thresholds table returned as strings; CLI coercion added.
- Validation loss weighting: Decide whether to report weighted val_loss (Open).
- Edge adjacency ordering: Consider threshold-before-topk and zeroing diag (Open).
