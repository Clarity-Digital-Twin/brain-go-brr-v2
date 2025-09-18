# Known Issues (P0 Focus)

This file tracks critical (P0) issues discovered so far. Do not fix here to avoid collisions with parallel workstreams; use this for triage and coordination.

- Issue: Pipeline CLI requires labels but datasets created without labels
  - Severity: P0 (crash at runtime)
  - Location: src/experiment/pipeline.py: train() and train_epoch(); src/experiment/data.py: EEGWindowDataset
  - Symptoms: `train_epoch` assumes `(windows, labels)` but `EEGWindowDataset` returns only `windows` when `label_files=None` (as used in pipeline.main). Also, `create_balanced_sampler` builds `all_labels` via `train_dataset[i][1]`, which fails when labels are absent.
  - Repro:
    - `python -m src.experiment.pipeline --config configs/production.yaml` (with no label files provisioned)
  - Impact: Training CLI unusable on unlabeled data; sampler creation crashes; first-batch label extraction crashes.
  - Workaround: Provide label files and adapt dataset instantiation; or disable balanced sampling and avoid sampler path (not recommended for production).
  - Owner: Pipeline/Data
  - Notes: Decide contract for `EEGWindowDataset`: always return `(window, label)` and synthesize zeros when labels absent, or adjust pipeline to support unlabeled runs.

- Issue: Evaluation entry point in pyproject points to non-existent function
  - Severity: P0 (packaging/CLI break)
  - Location: pyproject.toml: [project.scripts] `evaluate = "src.experiment.evaluate:main"`; but `src/experiment/evaluate.py` has no `main`.
  - Symptoms: Installed script `evaluate` fails to run/import.
  - Repro: `uv run evaluate` (after install)
  - Impact: Broken CLI for evaluation when installed as a package.
  - Workaround: Use `python -m src.cli evaluate ...` instead.
  - Owner: Packaging/CLI
  - Notes: Either add `main()` in evaluate.py or point script to `src.cli:main` with a subcommand.

- Issue: FA/24h computed on overlapped windows without stitching
  - Severity: P0 (metric correctness)
  - Location: src/experiment/evaluate.py: sensitivity_at_fa_rates(), find_threshold_for_fa_eventized(); uses `labels.numel()` for time and per-window events without stitching.
  - Symptoms: `total_hours = labels.numel() / (fs * 3600)` double-counts time when windows overlap; predicted events are computed per-window and not stitched across boundaries.
  - Impact: FA/24h and threshold search are inaccurate; downstream sensitivity@FA and TAES become unreliable.
  - Workaround: For now, evaluate on non-overlapping segments only; or pre-stitch probabilities to full timelines before calling evaluation.
  - Owner: Evaluation/Post-processing
  - Notes: Blocked on Phase 4 stitching APIs. Replace with `stitch_windows()` and compute actual per-record duration.

- Issue: Eventization semantics unclear; unused variable suggests mismatch
  - Severity: P1 (logic ambiguity, not a crash)
  - Location: src/experiment/evaluate.py: batch_probs_to_events()
  - Symptoms: Variable `binary = (prob_np > threshold)` computed but unused; hysteresis gates directly on raw probs (τ_on/τ_off). It mixes a global threshold and hysteresis, but only hysteresis affects the final mask.
  - Impact: Threshold parameter may not influence outputs as expected; harder to reason about FA/threshold search.
  - Workaround: Treat `threshold` as unused and rely on hysteresis only; or remove hysteresis temporarily.
  - Owner: Evaluation/Post-processing
  - Notes: To be resolved when Phase 4 APIs land (apply_hysteresis vs global thresholding).

- Issue: Memory-heavy window materialization in dataset
  - Severity: P1 (scaling blocker)
  - Location: src/experiment/data.py: EEGWindowDataset
  - Symptoms: All windows are precomputed and kept in memory.
  - Impact: OOM for large corpora; not viable for full TUH.
  - Workaround: Keep for unit tests and small runs; switch to streaming/dynamic windowing for large runs.
  - Owner: Data/Infra
  - Notes: Planned improvement; not urgent for unit/integration tests.

- Issue: BCE with logits reconstructed from probabilities
  - Severity: P2 (numerical suboptimality)
  - Location: src/experiment/pipeline.py: train_epoch()
  - Symptoms: Model outputs Sigmoid probabilities; training reconstructs logits via `torch.logit(probs.clamp(...))` to use `BCEWithLogitsLoss`.
  - Impact: Extra non-linear step; may reduce numerical headroom at extremes.
  - Workaround: Keep as-is; stable with clamp.
  - Owner: Training
  - Notes: Consider changing detection head to output raw logits and apply Sigmoid only at inference.

- Issue: Balanced sampler label aggregation cost
  - Severity: P2 (performance)
  - Location: src/experiment/pipeline.py: main(), create_balanced_sampler()
  - Symptoms: Builds `all_labels` by iterating full dataset, materializing labels for sampler.
  - Impact: Slow startup and high memory on large sets.
  - Workaround: Disable balanced sampling or compute window-level label presence offline.
  - Owner: Pipeline

- Issue: Unused global threshold vs hysteresis definitions in docs/code
  - Severity: P2 (documentation drift)
  - Location: README.md, PHASE4_POSTPROCESSING.md vs evaluate.py
  - Impact: Confusion over intended thresholding pipeline.
  - Owner: Docs/Eval

# Coordination Notes
- Phase 4 implementation (postprocess/events/export) will address stitching, morphology, and eventization; evaluation should be refactored to consume those APIs.
- Mamba CUDA/CPU dispatch appears robust; no P0 found after gating and kernel-width coercion.

# Triage Owners
- Pipeline/Data: unlabeled dataset handling; sampler assumptions
- Packaging/CLI: evaluate entrypoint mismatch
- Evaluation/Post-processing: FA/24h time accounting; stitching; eventization semantics
- Training: logits vs probabilities

