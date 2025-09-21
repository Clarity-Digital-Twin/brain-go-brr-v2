Evaluation (TAES, FA/24h, Protocol)

Scope
- Metrics and protocol: sample‑based and event‑based; sensitivity vs FA/24h.

Protocol
- Tune on dev set only; reserve eval for one‑shot reporting.
- Report TAES metrics with event overlap criteria as per benchmark tooling.
- Practical targets (TAES):
  - 10 FA/24h: >95% sensitivity
  - 5 FA/24h: >90% sensitivity
  - 1 FA/24h: >75% sensitivity

Code anchors
- src/brain_brr/eval/* (if present)
- configs/tusz_dev_tuning.yaml, configs/tusz_eval_final.yaml (if present)

Docs
- phases/PHASE5_EVALUATION.md
- implementation/EVALUATION_CHECKLIST.md
- deployment/TRAINING_AUDIT_CHECKLIST.md
