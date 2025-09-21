# Phase 2 Architecture Audit Checklist (Docs + Schemas)

Goal: Verify critical alignment between Phase 2 docs, Phase 1 specs, and schemas.
Resolve any blocking inconsistencies before implementation. Keep names consistent
with Brain-Go-Brr v2 (no class name suffix “V2”).

Status legend: [OK] aligned, [FIXED] updated now, [DECIDE] awaits choice, [TODO] to implement in code.

1) Decoder upsampling kernel
- Verify schema default matches docs: decoder.kernel_size = 2
  - Schemas: src/experiment/schemas.py: DecoderConfig.kernel_size default=2 [FIXED]
  - Docs: PHASE2.4_DECODER.md and PHASE2_MODEL_ARCHITECTURE.md use ConvTranspose1d(..., kernel_size=2, stride=2) [OK]

2) Mamba layers default
- Verify default number of layers is 6 across schema/docs
  - Schemas: MambaConfig.n_layers default=6 [FIXED]
  - Docs: PHASE2.3_BIMAMBA.md, PHASE2_MODEL_ARCHITECTURE.md assume 6 [OK]

3) Param naming (schema ↔ docs ↔ model init)
- Source of truth is ModelConfig (schemas). Implementation must accept config fields or a direct ModelConfig.
  - encoder.stages → 4 [OK]
  - rescnn.n_blocks → 3 [OK]
  - mamba.n_layers → 6 [OK]
  - mamba.conv_kernel → 5 [OK]
- Action: In implementation, prefer constructor `from_config(cfg: ModelConfig)` or map schema names to init args. [DECIDE/TODO]

4) Initial encoder conv (kernel=7) vs block kernel (5)
- Docs explicitly show `input_conv` kernel_size=7; blocks use kernel 5. [OK]
- Ensure note is present: PHASE2.1_UNET_ENCODER.md & PHASE2_MODEL_ARCHITECTURE.md [OK]

5) Bi-Mamba CPU fallback conv span
- Docs fallback now uses kernel=5 to match conv_kernel. [FIXED] (PHASE2.3_BIMAMBA.md)

6) Class naming
- Use `SeizureDetector` (no “V2”) consistently in Phase 2 docs. [FIXED]
  - PHASE2_MODEL_ARCHITECTURE.md, PHASE2.5_FULL_MODEL.md updated.

7) Decoder output channels rationale
- Add short rationale: decoder outputs 19 channels and a 1×1 conv fuses to 1. [FIXED]

8) Terminology consistency
- Use “skip connection fusion” (not “skip fusion”). [FIXED] in checklist header.

9) Phase 1 → Phase 2 interface clarity
- Reiterate: Inputs are 19 canonical 10–20 channels, per-channel z-scored, shape (B,19,15360) @256 Hz. [FIXED]

10) ResCNN branch split explanation
- Ensure docs note 512→170+170+172 split; fusion back to 512. [OK]

11) Config completeness / width
- Width is fixed (base_channels=64; encoder channels [64,128,256,512]) for Phase 2. Documented in schema validator. [OK]

12) Tests alignment once implemented
- Unit tests in Phase 2 docs must reflect any future param/name tweaks (e.g., decoder kernel). [TODO]

Sign-off checks (before coding)
- Schemas defaults and keys match Phase 2 docs [OK]
- Model class name finalized (`SeizureDetector`) [OK]
- Clear plan to consume ModelConfig in implementation (constructor or mapping) [DECIDE]

Recommended decisions
- Constructor: implement `SeizureDetector.from_config(model_cfg: ModelConfig) -> SeizureDetector` to avoid name drift. [DECIDE]
- Keep CPU fallback conv=5 to mirror Mamba conv span. [DONE]

> Note: This Phase doc is being replaced by component‑oriented docs. See components/models/* and components/training.md for current audit points.
