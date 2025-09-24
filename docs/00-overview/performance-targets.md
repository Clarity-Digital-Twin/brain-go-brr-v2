# Performance Targets and System Profile

Targets (TAES)

- 10 FA/24h → Sensitivity > 95%
- 5 FA/24h → Sensitivity > 90%
- 1 FA/24h → Sensitivity > 75%

Training times (typical)

- Local (RTX 4090): ~2–3 hours/epoch; 100 epochs ~200–300 hours
- Modal (A100‑80GB): ~1 hour/epoch; 100 epochs ~100 hours (~$319)
- Smoke test: ~5 minutes

Resource usage

- VRAM: 12–20GB (RTX 4090, batch ~8–12); 40–60GB (A100, batch ~48–64)
- Cache size: ~50GB processed NPZ files
- Checkpoint: ~125MB per epoch

Complexity summary

- TCN: O(N); Node/Edge Mamba: O(N) per stream; GNN: O(E+V) per timestep (vectorized over time)

Post‑processing defaults

- Hysteresis: τ_on=0.86, τ_off=0.78
- Morphology: Opening(11), Closing(31)
- Duration: 3–600s; Merge within 2s
