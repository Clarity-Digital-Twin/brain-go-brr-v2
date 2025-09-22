# Modal Cloud Deployment (Directory Readme)

For end-to-end deployment instructions, use the Single Source of Truth:
- docs/deployment/MODAL_DEPLOYMENT_SSOT.md
- docs/deployment/MODAL_PIPELINE_SETUP.md
- docs/deployment/PREFLIGHT_STRATEGY.md

Quick commands
```bash
# Smoke test
modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml

# Full A100 training
modal run --detach deploy/modal/app.py -- --action train --config configs/modal/train.yaml

# Evaluate checkpoint
modal run deploy/modal/app.py -- --action evaluate --config /results/checkpoints/best.pt
```

Notes
- Always put `--detach` before the script path to keep long jobs alive.
- S3 bucket and secrets setup are documented in the SSOT.

