# TCN Replacement (Current Runtime Path)

This document captures the plan and rationale for replacing the U‑Net encoder/decoder and ResCNN stack with a TCN front‑end feeding Bi‑Mamba, while preserving all public contracts (shapes, head, post‑processing). This is the architecture currently training on Modal.

## New Path (API‑compatible)
```
EEG (B,19,15360)
 → TCN Encoder (B,512,960)
 → Bi‑Mamba‑2 (B,512,960)
 → 1×1 (512→19) (B,19,960)
 → Upsample ×16 (B,19,15360)
 → Detection head → (B,15360)
```

Highlights
- Maintains per‑sample logits at 256 Hz; no loss/postproc changes.
- Replaces U‑Net + ResCNN with simpler, faster dilated temporal blocks.
- Bi‑Mamba unchanged; set `conv_kernel=4` in configs for CUDA compatibility.

See also: docs/04-research/future/CANONICAL-ROADMAP.md for roadmap status and next steps.

---

## Motivation vs Previous Path
- U‑Net is overkill for 1D signals here and adds latency/VRAM.
- ResCNN’s role overlaps with TCN’s residual temporal blocks.
- TCN + Bi‑Mamba provides local + global temporal modeling with O(N) cost.

---

## Interfaces and Shapes
- Input: `(B, 19, 15360)`
- TCN output: `(B, 512, 960)` (×16 downsample)
- Post‑Mamba: `(B, 512, 960)`
- Projection: `(B, 19, 960)`
- Upsample: `(B, 19, 15360)`
- Head: `(B, 15360)` logits

---

## Config Gating
Use `model.architecture: tcn` to select this path. Keep the U‑Net path available for ablations until metrics are finalized.

---

## Testing Strategy
- Unit: shape checks through encoder/head; logits length 15360.
- Integration: detector forward on both paths; no NaNs; device/dtype preserved.
- Smoke: run with `BGB_SMOKE_TEST=1` to validate pipeline without sampler cost.

---

## Migration Steps (done)
- Add TCN encoder module and wire path in `SeizureDetector`.
- Add 1×1 projection and ×16 upsample.
- Update configs to set `architecture: tcn` and `mamba.conv_kernel=4`.

---

## Notes
- Prefer non‑causal TCN for offline training; expose causal later for streaming.
- Optional: depthwise+pointwise pre‑conv as denoiser if needed.

