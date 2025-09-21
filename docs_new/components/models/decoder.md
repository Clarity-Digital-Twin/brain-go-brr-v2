Decoder (Upsampling)

Code anchors
- src/brain_brr/models/unet.py (decoder path)
- src/brain_brr/models/detector.py (heads)

Spec
- Upsample ×2 then Conv1d per stage; residual skips from encoder.

Docs
- phases/PHASE2.4_DECODER.md
