# Performance Targets and System Profile

Targets (TAES)

- 10 FA/24h → Sensitivity > 95%
- 5 FA/24h → Sensitivity > 90%
- 1 FA/24h → Sensitivity > 75%

Training times (typical)

- Local (RTX 4090, batch 8): ~3–4 hours/epoch; 100 epochs ~300–400 hours
- Modal (A100‑80GB, batch 48): **7-12 hours/epoch** (training 1-2h + validation 5.8h documented); 100 epochs ~700-1200 hours. **Cost: $3,400-$5,300+ for 100 epochs** @ $4.40/hr (GPU $2.50 + CPU $1.13 + RAM $0.77). Actual costs may be higher due to bottlenecks.
- Smoke test (local or Modal): ~5 minutes

Resource usage (current configs)

- VRAM: ~20 GB on RTX 4090 (batch 8, mixed precision off); ~55–60 GB on A100‑80GB (batch 48, mixed precision on)
- Cache size: ≈500 GB memory-mapped NPY pairs (`cache/tusz_mmap/*`)
- Checkpoints: ~125 MB per epoch (per run)

Complexity summary

- TCN: O(N); Node/Edge Mamba: O(N) per stream; GNN: O(E+V) per timestep (vectorized over time)

Post‑processing defaults

- Hysteresis: τ_on=0.86, τ_off=0.78
- Morphology: Opening(11), Closing(31)
- Duration: 3–600s; Merge within 2s
