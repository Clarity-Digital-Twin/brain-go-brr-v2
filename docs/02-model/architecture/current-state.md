# Current Architecture State (Read First)

This summarizes the active runtime path vs historical design to avoid confusion during development and deployment.

## Current Reality (v2.3)
```
EEG → TCN Encoder → Bi‑Mamba → Projection → Upsample → Detection
```
- TCN replaced both U‑Net and ResCNN.
- Modal configs use `architecture: tcn` and this path is training now.

## Legacy Path (kept for ablations)
```
EEG → U‑Net Encoder → ResCNN → Bi‑Mamba → U‑Net Decoder → Detection
```

## Integration Point for Next Step (GNN)
Insert GNN after Bi‑Mamba at the bottleneck (before projection/upsample):

```
TCN → Bi‑Mamba → [GNN here] → Projection → Upsample → Detection
```

See also:
- docs/02-model/architecture/tcn-replacement.md (details and rationale)
- docs/04-research/future/CANONICAL-ROADMAP.md (status and next steps)

