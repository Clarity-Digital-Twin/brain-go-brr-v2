Bi-Mamba-2 (Bidirectional SSM)

Code anchors
- src/brain_brr/models/mamba.py

Spec
- 6 layers, d_model=512, d_state=16, selective SSM.
- CUDA kernels: d_conv coerced to supported sizes; Conv1d fallback available.

Runtime notes
- Force fallback: `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- See deployment/MODAL_SSOT.md for Modal CUDA compilation status.

Docs
- phases/PHASE2.3_BIMAMBA.md
- architecture/MAMBA_KERNEL_DECISION.md
