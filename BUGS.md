Brain-Go-Brr v2.6/V3 — Risk Audit and Bug List

Severity legend
- P0: Blocks correct training/evaluation or corrupts results
- P1: High risk of silent degradation or wrong metrics; not a hard block
- P2: Medium risk, confusing or suboptimal defaults; can mislead users
- P3: Low risk, polish and ergonomics

P0 blockers
- Train/val split risks leakage and bias
  - Problem: Splitting uses a deterministic alphabetical split across all EDFs with no patient-level grouping or shuffling. This can mix windows from the same patient across splits and/or induce strong dataset-order bias.
  - Where: src/brain_brr/train/loop.py:1364 and surrounding record discovery/split logic
  - Impact: Invalid validation metrics; optimistic estimates; overfitting/generalization issues.
  - Fix: Build a patient-level manifest and split by patient ID (stable hash + seed) or accept a predefined split. Shuffle with a fixed seed; expose split seeds in config for reproducibility.

- Evaluation FA curve uses wrong thresholds (ignored override)
  - Problem: sensitivity_at_fa_rates() computes a per-FA tau_on but passes it via the deprecated threshold arg to batch_probs_to_events(), which currently ignores the threshold parameter.
  - Where: src/brain_brr/eval/metrics.py: sensitivity_at_fa_rates → batch_probs_to_events; batch_probs_to_events ignores its threshold arg.
  - Impact: FA curve values are inconsistent with the threshold search; downstream analyses relying on fa_curve are wrong. Training’s validate_epoch() uses a correct code path, but this still risks misuse.
  - Fix: Either (A) update sensitivity_at_fa_rates() to clone post_cfg, set hysteresis.tau_on/off, and pass that (as evaluate_predictions does), or (B) implement threshold override inside batch_probs_to_events() when threshold is not None.

P1 issues
- Evaluate CLI defaults to best threshold lookup by string key; brittle if types change
  - Problem: It expects thresholds["10"] (string key). The thresholds dict is currently stringified, but future changes to keep numeric keys will silently fall back to 0.86.
  - Where: src/brain_brr/cli/cli.py:460–468
  - Impact: Wrong export threshold if key types differ; inconsistent event exports.
  - Fix: Normalize keys to str/float at creation or coerce when reading (e.g., map keys to str once).

- Validation loss unweighted wrt extreme imbalance
  - Problem: validate_epoch() uses plain BCEWithLogitsLoss without pos_weight. For highly imbalanced data this can make val_loss misleading.
  - Where: src/brain_brr/train/loop.py:812–828 (criterion construction)
  - Impact: Val loss comparisons and LR/scheduler heuristics based on it are biased; can hide collapse.
  - Fix: Mirror class weighting used in train (pos_weight) or report both weighted and unweighted metrics.

- Non-random split order can encode confounds even with fixed seed
  - Problem: Sorting filenames and then slicing induces a fixed bias if filenames encode site/date/patient.
  - Where: src/brain_brr/train/loop.py:1359–1368
  - Impact: Distribution shift between splits; over/underestimation of generalization.
  - Fix: Shuffle file list deterministically (seeded) before splitting and prefer patient-level split.

P2 issues
- TCN channels in config are ignored by implementation
  - Problem: ModelConfig.tcn.channels is accepted and shown in summaries but TCNEncoder constructs channels internally and never reads that field.
  - Where: src/brain_brr/models/tcn.py (TCNEncoder.__init__) and src/brain_brr/config/schemas.py (TCNConfig)
  - Impact: Config ≠ behavior; wasted tuning time; confusion for reviewers.
  - Fix: Wire channels list from config into TCNEncoder or remove it from config to avoid bait-and-switch.

- TensorBoard likely not installed in base deps, but imported unconditionally
  - Problem: from torch.utils.tensorboard import SummaryWriter at import time; tensorboard pip isn’t in [project] deps.
  - Where: src/brain_brr/train/loop.py (top-level import)
  - Impact: Runtime import error on fresh envs unless dev deps are installed; training fails early.
  - Fix: Make TB optional (lazy import guarded by try/except) or add tensorboard to base dependencies.

- Manifest-based balancing silently assumes label presence in NPZ
  - Problem: scan_existing_cache() treats any NPZ without labels as fully background and floods manifest with no-seizure entries.
  - Where: src/brain_brr/data/cache_utils.py
  - Impact: If labels failed to save for a subset, sampling skews heavily, reducing effective positives.
  - Fix: Warn loudly and optionally exclude unlabeled files from balancing unless explicitly allowed.

- Edge top-k can select zeros/self-edges after thresholding
  - Problem: assemble_adjacency() top-k is applied before threshold; with many small weights this may keep near-zeros; diagonal is not explicitly excluded before top-k.
  - Where: src/brain_brr/models/edge_features.py
  - Impact: Graph contains weak/uninformative edges; unnecessary compute and reduced SNR.
  - Fix: Zero the diagonal pre-topk; apply threshold before top-k; ensure top-k is taken over strictly positive weights.

- Heavy sampler probing on uncached datasets
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

Suggested next steps
- Implement patient-level, seeded splits and document the exact policy.
- Fix FA curve path by honoring tau_on either in sensitivity_at_fa_rates() or batch_probs_to_events().
- Wire TCN channels from config or delete the unused field to avoid reviewer confusion.
- Make TensorBoard optional or part of base deps; guard its import.
- Add CI check to catch unlabeled NPZ files polluting manifest.

