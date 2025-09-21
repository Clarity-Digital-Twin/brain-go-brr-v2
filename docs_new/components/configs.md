Configs & Metrics (Reference)

Scope
- Config structure, split-specific configs, batch sizes, workers, metrics logging.

Code anchors
- configs/*
- src/brain_brr/train/loop.py (learning rate, schedulers, logging points)

Docs
- deployment/SENIOR_REVIEW_TRAINING_CONFIGS.md
- implementation/EVALUATION_CHECKLIST.md

Notes
- Keep WSL2-safe and A100-optimized configs separate.
- Use smoke_test.yaml for quick verification.
