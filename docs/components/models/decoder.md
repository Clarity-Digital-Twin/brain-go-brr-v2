Decoder (Upsampling)

Code anchors
- src/brain_brr/models/unet.py (decoder path)
- src/brain_brr/models/detector.py (heads)

Spec
- 4 stages; each stage upsamples by ×2, concatenates with the matching encoder skip, then applies Conv1d blocks.
- Channel progression mirrors encoder in reverse: 512→256→128→64→64.
- Final reconstruction to 19 channels before the detection head.

Detection head
- 1×1 Conv1d from 19→1 + Sigmoid to produce (B, 1, 15360) → squeeze to (B, 15360).

Shapes
- Bottleneck (512,960) → (256,1920) → (128,3840) → (64,7680) → (64,15360) → head → (1,15360).
