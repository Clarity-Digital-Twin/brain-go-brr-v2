# MODEL PIPELINE DIAGRAM (TCN + Bi‑Mamba, optional GNN)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                SEIZURE DETECTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐       ┌──────────────────────────────────────────────────┐
│  RAW EDF     │       │                PREPROCESSING                     │
│              │       │                                                  │
│ Multi-channel│─────▶│  • Load 19-ch 10-20 montage (fixed order)        │
│   EEG data   │       │  • Resample to 256 Hz                            │
│              │       │  • Bandpass 0.5-120 Hz + 60 Hz notch             │
└──────────────┘       │  • Per-channel z-score (full recording)          │
                       │  • Keep timestamps for stitching                 │
                       └────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────────────┐
                       │              WINDOW EXTRACTION                   │
                       │                                                  │
                       │  • 60-second windows (15360 samples @ 256 Hz)    │
                       │  • 10-second stride (83% overlap)                │
                       │  • Shape: (B, 19, 15360)                         │
                       └────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              DEEP LEARNING MODEL                                      │
│                                                                                       │
│  ┌────────────────┐     ┌────────────────┐     ┌───────────────────────────────┐      │
│  │   TCN Encoder  │───▶│   Bi‑Mamba‑2   │────▶│   Optional GNN + LPE (PyG)    │      │
│  │ (×16 down)     │     │  (6 layers)    │     │ (vectorized, static PE by     │      │
│  │ (B,512,960)    │     │ (B,512,960)    │     │  default in v3)               │      │
│  └────────────────┘     └────────────────┘     └───────────────┬───────────────┘      │
│                                                                │                      │
│                                   Bottleneck: (B, 19, 960, 64) ▼                      │
│                                                        ┌───────────────────────┐      │
│                                                        │ Projection + Upsample │      │
│                                                        │   19*64→512, ×16 up   │      │
│                                                        └──────────┬────────────┘      │
│                                                                   ▼                   │
│                                                            ┌──────────────┐           │
│                                                            │  Detection   │           │
│                                                            │   Head 1×1   │           │
│                                                            └──────────────┘           │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Notes:
- v2 (architecture: tcn) uses heuristic adjacency when `graph.enabled=true`.
- v3 (architecture: v3) uses learned adjacency via an edge stream (Bi‑Mamba) + vectorized PyG with static PE.

