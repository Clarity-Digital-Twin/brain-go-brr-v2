Decoder (Upsampling) — Legacy (pre‑v2.3)

Note: This document describes the U‑Net decoder path that was used prior to v2.3. The current
runtime architecture uses a Projection+Upsample head in the TCN path instead of this decoder.
See: `docs/02-model/architecture/current-state.md` and `docs/02-model/architecture/tcn-replacement.md`.

Code anchors
- (legacy) `src/brain_brr/models/unet.py` (decoder path)
- `src/brain_brr/models/detector.py` (current detection head)

Spec
- 4 stages; each stage upsamples by ×2, concatenates with the matching encoder skip, then applies Conv1d blocks.
- Channel progression mirrors encoder in reverse: 512→256→128→64→64.
- Final reconstruction to 19 channels before the detection head.

Detection head
- 1×1 Conv1d from 19→1 producing raw logits (B, 1, 15360) → squeeze to (B, 15360). Apply Sigmoid at inference.

Shapes
- Bottleneck (512,960) → (256,1920) → (128,3840) → (64,7680) → (64,15360) → head → (1,15360).

