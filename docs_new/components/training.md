Training (Loop, Dataloaders, Guards)

Scope
- Train/val split handling, dataset choice, samplers, logging.

Code anchors
- src/brain_brr/train/loop.py (dataset selection, fail-fast, sampler bypass)
- src/brain_brr/cli/cli.py (train entry)
- configs/ (tusz_train_wsl2.yaml, tusz_train_a100.yaml, smoke_test.yaml)

Docs
- phases/PHASE3_TRAINING_PIPELINE.md
- deployment/PREFLIGHT.md, deployment/MODAL_SSOT.md

Notes
- BalancedSeizureDataset for train only; val/test use standard dataset.
- Fail fast if balanced manifest yields 0 windows.
- WSL2: num_workers=0; avoid pin_memory.
