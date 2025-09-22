Full Model (Detector Wiring)

Note: The current runtime path uses the TCN front‑end instead of U‑Net + ResCNN. See `docs/02-model/architecture/tcn-replacement.md` for the architecture that is training on Modal now. The U‑Net path below remains documented for ablations and historical reference.

Code anchors
- src/brain_brr/models/detector.py (SeizureDetector)

Spec
- Encoder (U‑Net) → ResCNN → Bi‑Mamba‑2 → Decoder → detection head.
- Assumes input (B, 19, 15360) with per‑channel z‑score normalization.
- Output: per‑timestep logits (B, 15360); apply Sigmoid at inference.

Shape tracking
- Input: (B, 19, 15360)
- Encoder: (64,15360) → (128,7680) → (256,3840) → (512,1920); bottleneck (512,960)
- ResCNN: (512,960)
- Bi‑Mamba‑2: (512,960)
- Decoder: (256,1920) → (128,3840) → (64,7680) → (64,15360)
- Head: (1,15360) → squeeze → (B, 15360)

Notes
- Keep decoder output at 19 channels prior to head to preserve channelwise features.
- Training uses BCEWithLogitsLoss on logits; do not add Sigmoid in the module.
