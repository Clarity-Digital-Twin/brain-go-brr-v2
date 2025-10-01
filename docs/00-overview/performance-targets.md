# Performance Targets and System Profile

Targets (TAES)

- 10 FA/24h → Sensitivity > 95%
- 5 FA/24h → Sensitivity > 90%
- 1 FA/24h → Sensitivity > 75%

Training times (typical)

- Local (RTX 4090): ~3–4 hours/epoch; 100 epochs ~300–400 hours
- Modal (A100‑80GB): ~1 hour/epoch; 100 epochs ~100 hours (~$319)
- Smoke test: ~5 minutes

Resource usage

- VRAM: ~10-16GB (RTX 4090 with V3, batch 4); 60–75GB (A100 with batch 32, grad_accum 2)
- Cache size: ~50GB processed NPZ files
- Checkpoint: ~125MB per epoch

Complexity summary

- TCN: O(N); Node/Edge Mamba: O(N) per stream; GNN: O(E+V) per timestep (vectorized over time)

Post‑processing defaults

- Hysteresis: τ_on=0.86, τ_off=0.78
- Morphology: Opening(11), Closing(31)
- Duration: 3–600s; Merge within 2s
