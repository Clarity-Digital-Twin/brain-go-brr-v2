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
- Best checkpoint path: `results/checkpoints/best.pt`; last: `results/checkpoints/last.pt`.
- Early stopping metric from config (e.g., sensitivity_at_10fa); patience configurable.
- Resume restores model/optimizer/scheduler and config snapshot.

Config notes
- WSL2 stability: num_workers: 0, pin_memory: false, persistent_workers: false.
- Keep smoke_test.yaml for quick verification; separate WSL2 vs A100 configs for batch sizes.

Code anchors
- src/brain_brr/train/loop.py (dataset selection, fail‑fast, sampler bypass)
- src/brain_brr/cli/cli.py (train entry)
- configs/* (tusz_train_wsl2.yaml, tusz_train_a100.yaml, smoke_test.yaml)

Docs
- phases/PHASE3_TRAINING_PIPELINE.md
- deployment/PREFLIGHT.md, deployment/MODAL_SSOT.md
