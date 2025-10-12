# State-Space Streams (BiMamba2 + FLA)

Files: `src/brain_brr/models/mamba.py`, `src/brain_brr/models/gated_deltanet.py`, `src/brain_brr/models/detector.py`

## BiMamba2 Baseline

Node stream

- d_model=64, n_layers=6, d_state=16, expand=2, headdim=8

Edge stream

- Edge features 1→16→1 (Conv1d), BiMamba2 d_model=16, n_layers=2, d_state=8, expand=2, headdim=4, Softplus

CUDA alignment

- `(d_model * expand) / headdim` must be an integer multiple of 8.
- Node: `(64 × 2) / 8 = 16` → aligned.
- Edge: `(16 × 2) / 4 = 8` → aligned.

Fallback behavior

- If `mamba-ssm` is unavailable or `SEIZURE_MAMBA_FORCE_FALLBACK=1`, the builder swaps to a Conv1d fallback.
- V3 sets headdim explicitly to keep kernels on the CUDA path.

Kernel details

- `d_conv` uses the CUDA-supported set {2, 3, 4}; we choose 4 for best temporal coverage and hardware efficiency.

## Flash Linear Attention (BiGatedDeltaNet) Variant

- Enable by setting `model.mamba.temporal_type: gated_deltanet` (node) and the same for the edge builder in configs such as `configs/local/train_fla.yaml`.
- Relies on `flash-linear-attention`’s `GatedDeltaNet` kernels packaged in `src/brain_brr/models/gated_deltanet.py`.

Node stream (Phase 1b validation complete)

- d_model=64, n_layers=6, headdim=8 (0.75× head constraint satisfied).
- Uses BiGatedDeltaNet with flash-linear attention for O(N) global context per channel.
- Parameter count drops to ~284k vs 398k for BiMamba2 (≈29% reduction).

Edge stream (Phase 1a validation complete)

- d_model increases to 32 (from 16) to meet causal_conv1d alignment requirements in FLA.
- Two-layer BiGatedDeltaNet stack mirrors the node architecture; overall parameter count grows to ~30k (~190% vs BiMamba2 edge), still negligible compared to node stream.

Alignment constraints

- Edge stream must set `edge_mamba_d_model: 32` in config to keep `(d_model × expand) / headdim` divisible by 8 for the Triton kernels.
- Node stream keeps `d_model=64`, `headdim=8`, `n_heads=8` (0.75× head constraint tracked in `src/brain_brr/constants.py`).

Operational notes

- Local smoke/full configs: `configs/local/{smoke,train}_fla.yaml`.
- Modal configs mirror BiMamba2 with the same data settings (batch 48, mixed precision true); see `configs/modal/train_fla.yaml`.
- Both streams share the same GNN and gated fusion stack; swapping the temporal type is isolated to the builder factory (`build_node_stream` / `build_edge_stream`).
- Use `flash-linear-attention==0.2.3` (pinned in `pyproject.toml`); verify with `python -c "from fla.layers import GatedDeltaNet"`.

Validation summary

- Phase 0: Factory wiring and schema fields added (temporal_type, head constraints, constants).
- Phase 1a/1b: Edge-only and node-only migrations validated via targeted smoke runs.
- Phase 2: Full stack (both streams GDN) validated via local smoke + medium runs; now live in production (v4.0.0 dual-stack milestone).

Research workflow

- Train BiMamba2 baseline on Modal (`make train-bimamba` → 100 epochs, timeout-resume).
- Train FLA variant locally (RTX 4090, batch 8, mixed precision false) or on Modal using the dedicated config once baseline completes.
- Compare sensitivity/FA curves after both 100-epoch runs; both results are publishable regardless of winner (novel architectures with identical downstream pipeline).
