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
- WSL2 stability: num_workers: 0, pin_memory: false, persistent_workers: false.
- Keep configs/local/smoke.yaml for quick verification; use local/train.yaml vs modal/train_a100.yaml for full runs.

Code anchors
- src/brain_brr/train/loop.py (dataset selection, fail‑fast, sampler bypass, checkpoints)
- src/brain_brr/cli/cli.py (train entry)
- configs/local/{train.yaml,smoke.yaml}; configs/modal/{train_a100.yaml,smoke_a100.yaml}

Cross-references
- Architecture: ../02-architecture/canonical-spec.md (training configuration)
- Deployment: ./deploy-preflight.md, ./deploy-modal.md
