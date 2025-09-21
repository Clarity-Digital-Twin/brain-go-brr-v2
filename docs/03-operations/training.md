Training (Loop, Dataloaders, Guards)

Scope
- Train/val split handling, dataset choice, samplers, logging, guardrails.

Dataset selection
- If manifest exists and use_balanced is enabled → BalancedSeizureDataset for train.
- Otherwise → EEGWindowDataset (classic) for train.
- Validation/test always use EEGWindowDataset (no balancing).
- WeightedRandomSampler is bypassed when using BalancedSeizureDataset.

Guardrails
- On scan‑cache/build‑cache: exit non‑zero if partial==0 and full==0.
- On train startup: abort if BalancedSeizureDataset length is 0.
- Logs dataset composition (partial/full/background counts and ratios).

Checkpoints & early stopping
- Checkpoints live under `<config.experiment.output_dir>/checkpoints/`:
  - `best.pt` (best by early_stopping.metric)
  - `last.pt` (last epoch)
- Early stopping config: `training.early_stopping.{metric,mode,patience}`
- Resume restores model/optimizer/scheduler and saved config snapshot.

Config notes
- Local (WSL2): prefer `num_workers: 4`, `pin_memory: true`, `persistent_workers: true`, `prefetch_factor: 2`. If you encounter hangs, fall back to `num_workers: 0`, `pin_memory: false`.
- Keep configs/local/smoke.yaml for quick verification; use local/train.yaml vs modal/train_a100.yaml for full runs.
- Balanced sampling: training validates the manifest and rebuilds if empty/stale; set `BGB_FORCE_MANIFEST_REBUILD=1` to force a fresh manifest.

Code anchors
- src/brain_brr/train/loop.py (dataset selection, fail‑fast, sampler bypass, checkpoints)
- src/brain_brr/cli/cli.py (train entry)
- configs/local/{train.yaml,smoke.yaml}; configs/modal/{train_a100.yaml,smoke_a100.yaml}

Cross-references
- Architecture: ../02-architecture/canonical-spec.md (training configuration)
- Deployment: ./deploy-preflight.md, ./deploy-modal.md
