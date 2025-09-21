Bi-Mamba-2 (Bidirectional SSM)

Code anchors
- src/brain_brr/models/mamba.py

Spec
- 6 layers on (B, L, D) with D=512, d_state=16.
- Bidirectional: run Mamba forward on x and on time‑reversed x, then flip back and fuse.
- Output projection from concat([fwd, bwd], dim=-1) to D; LayerNorm + Dropout + residual.
- d_conv default 5 (CUDA kernels may support {2,3,4}; CUDA path coerces internally; Conv1d fallback available).

Runtime notes
- Force Conv1d fallback: `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- See deployment/MODAL_SSOT.md for CUDA compilation status and caveats.

Shapes
- Input to block: (B, 512, 960) → transpose to (B, 960, 512) for Mamba layers; transpose back.
