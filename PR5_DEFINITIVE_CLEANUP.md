# PR‑5: Definitive Clamp Cleanup & Stable Defaults

Status: Proposal for review and alignment

Owner: Core V3 architecture team

Scope: V3 dual‑stream stack (TCN + BiMamba + GNN + LapPE)

As‑of: current HEAD (line anchors included for precision)

## Executive Summary

PR‑5 finalizes the stability work by removing all non‑essential clamps and nan_to_num calls made redundant by PR‑1/2/3, and by setting stable‑by‑construction defaults in configs. We keep only mathematically required guards (cosine/division), input sanity, pre‑loss/output clamps, and PE guards. The result is a cleaner, faster, and reproducible V3 without band‑aids.

## What Stays vs What Goes

The lists below reflect exact sites at HEAD. If lines drift slightly, use the path anchors to locate the snippets.

### Keep (essential guards)

- Input sanity clamp
  - src/brain_brr/models/tcn.py:241
- Edge feature math bounds and division safety
  - src/brain_brr/models/edge_features.py:73, src/brain_brr/models/edge_features.py:81, src/brain_brr/models/edge_features.py:87, src/brain_brr/models/edge_features.py:91
- Laplacian PE numerical safety
  - src/brain_brr/models/gnn_pyg.py:229 (eigenvalue clamp), src/brain_brr/models/gnn_pyg.py:249 (pe nan_to_num)
- Final decoder/logit safety (pre‑loss)
  - src/brain_brr/models/detector.py:368, src/brain_brr/models/detector.py:369, src/brain_brr/models/detector.py:375, src/brain_brr/models/detector.py:376
- Loss‑level probability/logit clamps
  - src/brain_brr/train/loop.py:205, src/brain_brr/train/loop.py:212, src/brain_brr/train/loop.py:218, src/brain_brr/train/loop.py:223

### Remove (redundant after PR‑1/2/3)

- TCN internal tier clamps
  - src/brain_brr/models/tcn.py:248, src/brain_brr/models/tcn.py:255, src/brain_brr/models/tcn.py:262
- Detector feature safe‑clamps now covered by PR‑1 norms
  - src/brain_brr/models/detector.py:226, src/brain_brr/models/detector.py:361
- Detector edge projection clamp (replaced by PR‑2 bounded lift)
  - src/brain_brr/models/detector.py:291
- Duplicate edge feature clamp at detector stage (upstream features already bounded)
  - src/brain_brr/models/detector.py:273
- GNN vectorized safe clamp (now redundant with PR‑3 conditioning)
  - src/brain_brr/models/gnn_pyg.py:359
- Mamba internal clamps (stage removal after monitoring confirms clean runs)
  - src/brain_brr/models/mamba.py:180, src/brain_brr/models/mamba.py:249, src/brain_brr/models/mamba.py:259, src/brain_brr/models/mamba.py:329, src/brain_brr/models/mamba.py:339, src/brain_brr/models/mamba.py:342

### Simplify nan_to_num (keep only critical)

- Remove (if clean for 10k batches with PR‑1/2/3):
  - src/brain_brr/models/tcn.py:238
  - src/brain_brr/models/mamba.py:177, src/brain_brr/models/mamba.py:328
  - src/brain_brr/models/gnn_pyg.py:358
  - src/brain_brr/models/detector.py:225, src/brain_brr/models/detector.py:360
- Keep (critical):
  - src/brain_brr/models/gnn_pyg.py:249 (PE path)
  - src/brain_brr/models/detector.py:368, src/brain_brr/models/detector.py:375 (final outputs)

## Updated Defaults (configs)

Make these the steady‑state defaults in both local and training configs to ensure PR‑1/2/3 protections are on by default:

```yaml
model:
  norms:
    boundary_norm: layernorm    # or: rmsnorm
    boundary_eps: 1.0e-5
  graph:
    enabled: true
    # PR‑2: bounded edge lift
    edge_lift_activation: tanh
    edge_lift_norm: layernorm    # or: rmsnorm
    edge_lift_init_gain: 0.1
    # PR‑3: adjacency conditioning
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.9
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true
  fusion:
    fusion_type: gated           # PR‑4; A/B vs "add" if desired
    fusion_heads: 4
    fusion_dropout: 0.1
  clamp_retirement:
    remove_intermediate_clamps: true
    remove_nan_to_num: true
    keep_input_clamp: true
    keep_output_clamp: true
    keep_loss_clamps: true
    log_clamp_hits: false        # enable only during initial monitoring
```

Environment: default `BGB_SAFE_CLAMP=0` (debug only).

## Acceptance Criteria

- Zero NaN/Inf in forward+backward for ≥10,000 consecutive batches.
- Clamp hit rate = 0 at all removed sites (during PR‑4 monitoring).
- Eigendecomposition success under PR‑3 conditioning; PE fallback rate ≈ 0.
- TAES sensitivity preserved; latency delta < 5%; memory delta < 10%.

## Implementation Checklist

1) Delete redundant clamps and nan_to_num
- TCN: remove internal clamps (src/brain_brr/models/tcn.py:248, :255, :262) and input nan_to_num (src/brain_brr/models/tcn.py:238).
- Detector: remove feature clamps (src/brain_brr/models/detector.py:226, :361), edge_in clamp (src/brain_brr/models/detector.py:291), duplicate edge feature clamp (src/brain_brr/models/detector.py:273), and feature nan_to_num (src/brain_brr/models/detector.py:225, :360).
- GNN: remove x_batch nan_to_num (src/brain_brr/models/gnn_pyg.py:358) and safe clamp (src/brain_brr/models/gnn_pyg.py:359).
- Mamba: remove staged clamps after monitoring green (src/brain_brr/models/mamba.py:180, :249, :259, :329, :339, :342) and nan_to_num (src/brain_brr/models/mamba.py:177, :328).

2) Set stable defaults in configs
- Enable PR‑1/2/3/4 as above in `configs/local/smoke.yaml` and `configs/local/train.yaml`.
- Set clamp_retirement to remove intermediate clamps and non‑critical nan_to_num.

3) Tests/validation
- Unit: no‑intermediate‑clamps forward remains finite; gated fusion outputs finite.
- Integration: one training step (forward+backward+opt) with finite gradients.
- PE/GNN: eigendecomposition success in fp32 with sign consistency; eigenvalues finite and within [0, 2] tolerance.

4) Rollback plan
- If instability appears, re‑enable `clamp_retirement.log_clamp_hits=true` and temporarily set `remove_intermediate_clamps=false` at the specific module while investigating.

## Current Intervention Counts (for the record)

- torch.clamp: 24 in models, plus 4 in training loop = 28 total
- nan_to_num: 15 total (models + data + train loop)

These counts are a snapshot at HEAD and can be re‑grepped to verify after PR‑5.

## Rationale (short)

- PR‑1 adds boundary norms + LayerScale at seams, PR‑2 bounds the edge lift, PR‑3 conditions adjacency + stabilizes Laplacian PE, PR‑4 introduces learnable fusion and monitoring; together they render intermediate clamps and most nan_to_num unnecessary.
- Keeping essential math and pre‑loss guards is standard practice and low‑cost insurance.

## Risks & Mitigations

- Rare edge‑case NaNs after deletion → temporarily re‑enable a monitored clamp at that site and inspect activations; adjust norm eps/PR‑3 tau/beta if needed.
- Over‑eager Mamba clamp removal → stage deletions; confirm gradient norm P95 stability before final removal.

## Change Log Impact

- Code: selective deletions in TCN, Detector, GNN, Mamba as listed; no public APIs change.
- Configs: defaults updated; clamp_retirement and fusion present; back‑compat defaults preserved if fields omitted.
- Docs: this file (root), plus an update line in docs/04-model and configs/README.md noting new defaults and rationale.

---

Contact: open a PR titled “PR‑5: Definitive Clamp Cleanup & Stable Defaults” referencing this document and include grep diffs to confirm counts before/after.

