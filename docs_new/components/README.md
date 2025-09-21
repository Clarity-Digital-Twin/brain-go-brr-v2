Components (Canonical Index)

Goal: Replace phase-based docs with component-oriented, code-aligned docs. Each page links to current code and the relevant Phase doc until content is migrated.

- Data I/O: data_io.md
- Caching & Sampling: caching_and_sampling.md
- Models: models/README.md
- Training: training.md
- Post-processing: postprocessing.md
- Evaluation: evaluation.md
- Configs & Metrics: configs.md

Migration map (Phase → Component)
- PHASE1_DATA_PIPELINE.md → Data I/O, Caching & Sampling
- PHASE2_* (UNet, ResCNN, Bi-Mamba, Decoder, Full) → Models/*
- PHASE3_TRAINING_PIPELINE.md → Training
- PHASE4_POSTPROCESSING.md → Post-processing
- PHASE5_EVALUATION.md → Evaluation

Status
- Live content remains in Phase docs; these component pages link to them and to code.
