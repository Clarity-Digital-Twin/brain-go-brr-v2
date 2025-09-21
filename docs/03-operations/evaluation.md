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

Threshold search (event‑level)
- Binary search τ_on in [0,1] to meet a target FA/24h; set τ_off = max(0, τ_on − 0.08).
- Post‑process with hysteresis+morphology+duration; compute FA/24h using per‑record durations.
- Sensitivity@FA is computed at the found τ_on for each target (10, 5, 2.5, 1).

Code anchors
- src/brain_brr/eval/* (if present)
- configs/local/{dev.yaml,eval.yaml}

CLI
- Evaluate a checkpoint: `python -m src evaluate <checkpoint_path> <data_path> --output-json <out.json>`

Cross-references
- Architecture: ../02-architecture/canonical-spec.md (evaluation metrics section)
- Post-processing: ./postprocessing.md
- Preflight checks: ./deploy-preflight.md
