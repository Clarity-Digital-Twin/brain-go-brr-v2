Deployment (Canonical)

Audience: developers running training locally (WSL2/Linux) and on Modal.

What to read

- PREFLIGHT.md — Non-negotiable checks before any run
- LOCAL_WSL2.md — Local setup + stable training loop on WSL2/Linux
- MODAL_SSOT.md — Single source of truth for Modal runs
- TROUBLESHOOTING.md — Common failures and quick fixes
 - TRAINING_AUDIT_CHECKLIST.md — Data split integrity and evaluation protocol
 - SENIOR_REVIEW_TRAINING_CONFIGS.md — Config diffs for WSL2 vs A100

Key rules

- Never start training if the TUSZ cache manifest has zero partial/full windows
- Favor BalancedSeizureDataset for training (manifest-driven), standard dataset for val/test
- On WSL2, keep `num_workers=0` and avoid pin_memory to prevent hangs
- For Mamba CUDA kernels, be aware of d_conv coercion; force fallback if needed

Notes
- Prefer `modal run --detach deploy/modal/app.py --action ...` (detach BEFORE script); see MODAL_SSOT.md.

