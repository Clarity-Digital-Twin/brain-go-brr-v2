# TCN Replacement Plan (Drop‑in for U‑Net + ResCNN)

## Executive Summary

Goal: Replace the U‑Net encoder/decoder and ResCNN stack with a modern TCN front‑end that feeds Bi‑Mamba, while preserving all public contracts (input/output shapes, detection head API, per‑sample 256 Hz logits, post‑processing).

What changes:
- U‑Net encoder/decoder + ResCNN → TCN front‑end producing a 512‑channel sequence at 1/16 time resolution (960 steps).
- Add a tiny projection + upsampling head after Bi‑Mamba to restore full 15360‑step resolution for loss/post‑processing.
- Keep Bi‑Mamba unchanged; set conv kernel to 4 in configs to remove coercion warning.
- Gate the new path via config (`model.architecture: tcn`); do not hard‑delete U‑Net/ResCNN until metrics are validated.

Why this is safe:
- Maintains per‑sample logits at 256 Hz → no changes to loss, labels, or post‑processing.
- Preserves `detection_head` API: still takes 19 channels at full temporal resolution.
- Enables smaller, faster model with simpler optimization.

---

## Current Architecture (to be gated off, not immediately removed)

```python
# src/brain_brr/models/detector.py:127-131
encoded, skips = self.encoder(x)       # U‑Net encoder
features = self.rescnn(encoded)        # ResCNN
temporal = self.mamba(features)        # Bi‑Mamba (keep)
decoded = self.decoder(temporal, skips)# U‑Net decoder
output  = self.detection_head(decoded) # Conv1d(19→1)
```

Issues in our use case:
- U‑Net is 2D‑centric; here we have 1D sequences.
- 16× down/upsampling via encoder/decoder adds memory/latency without clear benefit once Mamba handles global context.
- ResCNN overlaps with TCN’s residual temporal blocks.

---

## New Architecture (drop‑in, API‑safe)

```python
x        = input_eeg                   # (B, 19, 15360)
features = self.tcn_encoder(x)         # (B, 512, 960)  downsampled by ×16
temporal = self.mamba(features)        # (B, 512, 960)
chan19   = self.proj_512_to_19(temporal)# (B, 19, 960)   1×1 projection
decoded  = self.upsample(chan19)       # (B, 19, 15360) upsample ×16
output   = self.detection_head(decoded)# (B, 1, 15360) → squeeze → (B, 15360)
```

Key points:
- Tiny head = `Conv1d(512→19,k=1)` + `Upsample(scale=16)`; preserves existing detection head and per‑sample label alignment.
- Bi‑Mamba stays identical; only its input changes (now from TCN instead of ResCNN).

---

## TCN Front‑End

Preferred: Non‑causal TCN for offline training (bidirectional receptive field). Causal can be toggled later for streaming inference.

Implementation options:
- External: `pytorch-tcn` (mature implementation). Use as an optional extra to avoid CI friction.
- Internal fallback: minimal dilated residual stack if the package is unavailable.

Shape contract:
- Input:  `(B, 19, 15360)`
- Output: `(B, 512, 960)` (use either stride‑based downsampling or a final `Conv1d(stride=16)` after keeping full length inside the TCN blocks).

Recommended hyperparameters (from Bai et al. 2018, adapted to EEG):
- Layers: 8 temporal residual blocks
- Kernel size: 7
- Dilation: powers of 2 (1..128) or grouped in stages
- Dropout: 0.1–0.15
- Norm: weight norm on temporal convs

Note: Set `model.mamba.conv_kernel = 4` in configs to avoid CUDA coercion warning.

---

## Labels, Loss, Post‑Processing (unchanged)

- Labels remain per‑sample at 256 Hz: vectors of length 15360.
- Loss is element‑wise on `(B, 15360)` (focal or BCE as configured).
- Post‑processing expects full‑resolution logits; the upsampling head ensures compatibility.

---

## Config & Gating (no hard deletes)

Add an architecture flag and TCN block in schema/configs. Example YAML (new files):

```yaml
# configs/modal/train_tcn.yaml
model:
  architecture: tcn
  tcn:
    channels: [64, 128, 256, 512]
    kernel_size: 7
    num_layers: 8
    dropout: 0.15
    causal: false            # offline training
    stride_down: 16          # 15360 → 960
mamba:
  conv_kernel: 4             # remove CUDA coercion warning
training:
  mixed_precision: true
  batch_size: 128
```

From code (pseudo):

```python
if cfg.model.architecture == "tcn":
    self.tcn_encoder     = TCNEncoder(...)
    self.proj_512_to_19  = nn.Conv1d(512, 19, 1)
    self.upsample        = nn.Upsample(scale_factor=16, mode="nearest")
    # keep detection_head as‑is
else:
    # existing UNet + ResCNN path
```

Keep `UNetEncoder`, `ResCNN`, and `UNetDecoder` available until we confirm non‑regression. Remove in a follow‑up PR once metrics are validated and configs migrated.

---

## Dependency Strategy (Optional Extra)

Avoid breaking CI (which avoids GPU builds and extras):
- Add `pytorch-tcn==1.2.3` under an optional extra in `pyproject.toml`, e.g.:

```toml
[project.optional-dependencies]
tcn = ["pytorch-tcn==1.2.3"]
```

- Local/Modal enablement: `uv sync -E tcn`
- In `models/tcn.py`, import‑guard the external package and fall back to a minimal in‑repo TCN if missing.

---

## TDD Plan (focused, non‑brittle)

Write tests first; keep them deterministic (avoid perf/timing assertions in CI):

Unit
- TCN shape: `(B,19,15360) → (B,512,960)`
- Head shape: `(B,512,960) → (B,19,15360) → detection_head → (B,15360)`
- No label/contract changes required (assert equality of lengths)

Integration
- Detector forward (tcn path) produces `(B,15360)` logits matching device/dtype of labels.
- Config gating selects the intended path; unet path still works.

Do NOT assert wall‑clock speed/GB in CI. Measure memory/throughput locally and document.

---

## Success Criteria

Mandatory
- Shape compatibility: logits length = 15360, unchanged post‑proc pipeline.
- Non‑regression on TAES@10FA vs UNet baseline (±5% tolerance for first pass; aim ≥ baseline).
- Clear parameter reduction from removing UNet+ResCNN path.

Expected
- Faster iteration time and lower memory vs UNet+ResCNN.
- Cleaner, easier‑to‑tune front‑end; simpler gradients.

---

## Migration Steps (no code deletions until validated)

1) Branching
- `git checkout -b feature/v2.3-tcn-architecture`
- Backup reference: `git branch backup/v2.0-unet-final && git push origin backup/v2.0-unet-final`

2) Tests first
- Add unit + integration tests for shapes and config gating.

3) Implement TCN front‑end + tiny head
- `models/tcn.py` (wrapper with import guard)
- Wire detector’s TCN path; keep UNet path intact under flag.

4) Configs
- Add `configs/modal/train_tcn.yaml`, `configs/local/test_tcn.yaml`.
- Set `mamba.conv_kernel=4` (removes CUDA coercion warning).

5) Validate
- Local smoke; Modal smoke (W&B + TB logging intact).
- Compare TAES + AUROC vs baseline checkpoints.

6) Remove legacy (follow‑up PR)
- After validation, remove UNet/ResCNN code and tests; migrate default configs to `architecture: tcn`.

---

## Notes & Rationale

- Non‑causal TCN is preferable for our offline batch training; causal can be exposed via config later for streaming inference.
- EEGWaveNet’s multi‑scale via stride pyramids is unnecessary here; TCN dilations + Mamba’s global context already cover multi‑scale effectively. Optional: a depthwise+pointwise pre‑conv is a cheap per‑channel denoiser if desired.
- Keeping the existing detection head reduces change surface and risk. The projection+upsample head is intentionally tiny to avoid re‑tuning large decoders.

---

## Open Risks & Mitigations

- Risk: Metric regression on first pass.
  - Mitigation: Keep UNet path gated; iterate TCN depth/dilations; verify mixed precision and batch size.
- Risk: External dependency friction in CI/Modal.
  - Mitigation: optional extra + import guard + internal fallback.
- Risk: Shape drift.
  - Mitigation: Shape tests; explicit projection+upsample head.

---

## Quick References

- Bai et al. 2018: TCNs outperform RNNs on 11/11 sequence tasks; k=7 often strong.
- Our stack invariants:
  - Input: 19×15360 (60 s @ 256 Hz)
  - Bi‑Mamba: 512×960 features (keep)
  - Output logits: length 15360, per‑sample
  - Post‑processing unchanged

---

## TL;DR Implementation Checklist

- [ ] Tests: TCN shape, head shape, detector integration, config gating
- [ ] TCN front‑end (non‑causal), output `(B,512,960)`
- [ ] Head: `Conv1d(512→19,1)` + upsample ×16
- [ ] Keep detection head; logits `(B,15360)`
- [ ] Config flag `model.architecture: tcn`; UNet path retained
- [ ] Optional extra for dependency; import guard
- [ ] Set `mamba.conv_kernel=4` in configs
- [ ] Validate metrics; then deprecate/remove UNet/ResCNN
