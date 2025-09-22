U‑Net (Encoder/Decoder) — Legacy (pre‑v2.3)

Note: This document describes the legacy U‑Net encoder/decoder used before the TCN front‑end
replacement. The active runtime path uses TCN → Bi‑Mamba → Projection+Upsample → Detection.
See: `docs/02-model/architecture/current-state.md`.

Code anchors
- (legacy) `src/brain_brr/models/unet.py`

Spec
- Encoder: 4 stages; channel progression [64, 128, 256, 512].
- Decoder: 4 stages; upsample ×2 each stage and fuse with matching skip.
- Skips saved after each encoder block, before downsample.
- Initial projection: Conv1d(k=7, pad=3) to 64 ch; blocks use k=5, pad=2.
- Downsample: Conv1d(k=2, stride=2) per stage (×16 total reduction).
- Activations: ReLU; normalization: BatchNorm1d per block.

Shapes (time length shown; channels in middle)
- Input: (B, 19, 15360)
- Encoder outputs/skips: [(64,15360), (128,7680), (256,3840), (512,1920)]
- Bottleneck: (B, 512, 960)
- Decoder path: (512,960) → (256,1920) → (128,3840) → (64,7680) → (64,15360)

Notes
- Preserve 19‑ch semantics by decoding back to 19 channels before the head.
- Use residual/skip fusion by concatenation, then Conv1d blocks.
