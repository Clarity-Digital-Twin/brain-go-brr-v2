ResCNN Stack — Legacy (pre‑v2.3)

Note: This document describes the legacy ResCNN bottleneck used before the TCN front‑end
replacement. The current runtime path does not include ResCNN.
See: `docs/02-model/architecture/current-state.md`.

Code anchors
- (legacy) `src/brain_brr/models/rescnn.py`

Spec
- Input/output channels: 512 (bottleneck width).
- 3 residual blocks; each block has multi‑scale conv branches with kernel sizes [3,5,7].
- Channel split across branches sums to 512; concatenate and fuse with 1×1 Conv1d.
- Dropout1d over channels in fusion; ReLU activations; BatchNorm per branch and fusion.
- Residual add: output = ReLU(fused + input).

Notes
- Maintains time length (no pooling here); preserves shape for Mamba input.
