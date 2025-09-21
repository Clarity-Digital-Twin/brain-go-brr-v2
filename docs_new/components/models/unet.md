U-Net (Encoder/Decoder)

Code anchors
- src/brain_brr/models/unet.py

Spec
- Channels: [64, 128, 256, 512], ×16 downsample, residual skip connections.
- Activation: ELU (per spec), batch norm per block.

Docs
- phases/PHASE2.1_UNET_ENCODER.md
- phases/PHASE2.4_DECODER.md
