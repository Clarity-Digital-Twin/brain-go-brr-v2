# BiMamba2 Streams

Files: `src/brain_brr/models/mamba.py`, `src/brain_brr/models/detector.py`

Node stream

- d_model=64, n_layers=6, d_state=16, expand=2, headdim=8

Edge stream

- Edge features 1→16→1 (Conv1d), BiMamba2 d_model=16, n_layers=2, d_state=8, expand=2, headdim=4, Softplus

CUDA alignment

- `(d_model*expand)/headdim` must be an integer multiple of 8
- Node: `(64*2)/8 = 16` → aligned
- Edge: `(16*2)/4 = 8` → aligned

Fallback behavior

- If `mamba-ssm` is unavailable or `SEIZURE_MAMBA_FORCE_FALLBACK=1`, a Conv1d fallback is used.
- V3 uses explicit headdim to prevent unintended fallback due to misalignment.

Kernel details

- `d_conv` uses the CUDA‑supported set {2, 3, 4}; we choose 4 for best temporal coverage and hardware efficiency.
