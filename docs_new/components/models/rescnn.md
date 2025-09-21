ResCNN Stack

Code anchors
- src/brain_brr/models/rescnn.py

Spec
- Input/output channels: 512 (bottleneck width).
- 3 residual blocks; each block has multi‑scale conv branches with kernel sizes [3,5,7].
- Channel split across branches sums to 512; concatenate and fuse with 1×1 Conv1d.
- Dropout1d over channels in fusion; ReLU activations; BatchNorm per branch and fusion.
- Residual add: output = ReLU(fused + input).

Notes
- Maintains time length (no pooling here); preserves shape for Mamba input.
