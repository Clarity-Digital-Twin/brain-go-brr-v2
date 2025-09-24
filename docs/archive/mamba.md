Bi-Mamba-2 (Bidirectional SSM)

Code anchors
- src/brain_brr/models/mamba.py

Spec
- Node stream: 6 layers on (B, L, D) with D=512, d_state=16.
- Edge stream: 2 layers on learned 1→D→1 embeddings; D is multiple‑of‑8 (default 16) to satisfy CUDA alignment and add capacity.
- Bidirectional: run Mamba forward on x and on time‑reversed x, then flip back and fuse.
- Output projection from concat([fwd, bwd], dim=-1) to D; LayerNorm + Dropout + residual.
- d_conv default 4 (CUDA kernels support {2,3,4}); Conv1d fallback available for CPU.

Runtime notes
- Force Conv1d fallback: `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- Mamba2 layers use expand=2 factor internally.
- See deployment/MODAL_SSOT.md for CUDA compilation status and caveats.

Shapes
- Node path: (B, 512, 960) in detector bottleneck; node features per electrode batched as (B*19, 64, 960) for Mamba.
- Edge path: edge scalar series per pair shaped (B*E, 1, 960) → learned lift (1→D, k=1) → Mamba (B*E, D, 960) → project back (D→1, k=1).
- Internally, Bi‑Mamba2 transposes (B, C, L) ↔ (B, L, C) for SSM kernels.
