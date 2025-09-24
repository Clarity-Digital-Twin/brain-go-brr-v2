# BiMamba2 Streams

Files: `src/brain_brr/models/mamba.py`, `src/brain_brr/models/detector.py`

Node stream

- d_model=64, n_layers=6, d_state=16, expand=2, headdim=8

Edge stream

- Edge features 1→16→1 (Conv1d), BiMamba2 d_model=16, n_layers=2, d_state=8, expand=2, headdim=4, Softplus

CUDA alignment

- `(d_model*expand)/headdim` must be multiple of 8
