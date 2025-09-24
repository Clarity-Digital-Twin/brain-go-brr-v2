# TCN Encoder

File: `src/brain_brr/models/tcn.py`

- 8 layers; channels `[64, 128, 256, 512]`
- Kernel size 7; stride-down 16
- Output shape: `(B, 512, 960)` from `(B, 19, 15360)`
