# TCN Encoder

File: `src/brain_brr/models/tcn.py`

- 8 layers; channels `[64, 128, 256, 512]`
- Kernel size 7; stride-down 16
- Output shape: `(B, 512, 960)` from `(B, 19, 15360)`

Backend selection

- Prefers a minimal internal TCN by default for stability.
- External `pytorch_tcn` can be forced with `BGB_FORCE_TCN_EXT=1`; otherwise the minimal TCN is used.
- CUDA optimizations (TF32) are enabled when available.

Input contract

- Input length must be divisible by `stride_down` (default 16); otherwise an assertion is raised.

Projection head

- `ProjectionHead`: `Conv1d(512→19)` then nearest‑neighbor upsample (×16) to restore `(B,19,15360)`.
