# Troubleshooting

Common issues

- Wrong cache dir: local `cache/tusz/`, Modal `/results/cache/tusz/` (persistent SSD)
- No seizures in batches: enable `use_balanced_sampling`
- NaN losses on 4090: set `mixed_precision: false`
- Modal stuck: increase CPU (24) and RAM (96GB)
- PyG install fails: use prebuilt wheels
- CI/CD: If PyG isn’t installed in the workflow, tests that require it are skipped by markers; install cu121 wheels to exercise GNN tests.
- Evaluate CLI exits early:
  - "No config found" — pass `--config` or use a checkpoint with embedded config
  - "No EDF files found" — verify `*.edf` exist under the provided data path

Operational P0 checklist (must pass)

- GPU stack versions match exactly (Torch 2.5.0+cu124, mamba-ssm 2.2.5, PyG 2.6.1 from cu124 wheels). Install order: `make setup` then `make setup-gpu`.
- Cache present with seizures and manifest built: run `python -m src scan-cache --cache-dir <cache_split_dir>`; expect partial>0 or full>0.
- Dynamic PE deps present if `graph.enabled: true`: `pip show torch_geometric` should list installed wheels.
- After preprocessing changes (e.g., ±10σ clip), rebuild cache: `rm -rf cache/tusz && python -m src build-cache ...`.

V3 NaN issues

- Primary contributing factor: dynamic PE eigendecomposition on poorly initialized adjacency.
- Default configs enable dynamic PE with safeguards; on consumer GPUs (RTX 4090), prefer increasing `graph.semi_dynamic_interval` first; as a last resort set `use_dynamic_pe: false`.
- Additional safeguards:
  - Edge similarity is clamped at the source with a configurable margin: `graph.edge_similarity_margin` (default 0.01)
  - Edge lift uses bounded activation + normalization (tanh + LayerNorm) by default (PR‑2). A conservative `[-3,3]` fallback only applies if PR‑2 is disabled.
  - Optimizer parameter groups (no weight decay on norms/bias)
  - Optional gradient sanitization (`BGB_SANITIZE_GRADS=1`)
- Details and timeline: `docs/08-operations/v3-nan-explosion-resolution.md`

Local training “gets stuck” checklist

- WSL2 dataloader: set `data.num_workers: 0` to avoid multiprocessing hangs.
- RTX 4090 NaNs: set `training.mixed_precision: false`; optionally reduce `learning_rate` or `batch_size`.
- Excessive CPU usage on Modal: ensure `resources.cpu: 24` and `resources.memory: 98304`.

Pre‑flight (before long runs)

- `make q` and `python -m src validate <config>` pass.
- `python -m src scan-cache --cache-dir <cache>` shows partial>0 or full>0.
- Startup logs show `BalancedSeizureDataset` and `Seizure ratio: ...`.

OOM root cause quick summary

- Full dynamic PE computes 960 eigendecompositions per window; the CUDA workspace across B×T can add several GB.
- Remedies, in order: increase `semi_dynamic_interval`, reduce `batch_size`, or (as a last resort) disable dynamic PE.

WSL2 OOM/driver artifact note

- If you see impossible memory in logs like `17179869184.00 GiB`, that is a reporting artifact after a hard OOM or driver fault.
- Fix: `wsl --shutdown` to reset the VM/GPU state, then relaunch. Also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` during smokes.

CI note (PyG optional in unit tests)

- The test suite skips GNN‑dependent tests when `torch_geometric` is absent. To validate GNN paths in CI, install PyG using prebuilt wheels for torch 2.5.0+cu124.

NaN logits root cause quick summary

- Tiny batches (e.g., 1) and aggressive FP16 can amplify numerical noise in dynamic PE and post‑Mamba projections.
- Remedies, in order: increase `batch_size` (≥4), set `training.mixed_precision: false` on RTX 4090, reduce `learning_rate`.
- Env toggles: `export BGB_NAN_DEBUG=1` (extra logging), `export SEIZURE_MAMBA_FORCE_FALLBACK=1` (force Conv1d fallback).

Modal cache hygiene

- Do not use S3 for caches; keep NPZs on Modal volume at `/results/cache/tusz/`
- Modal persistent volume only used for results at `/results/`
- Ensure `data.data_dir: /data/edf`; the loader enforces official patient-disjoint splits automatically (legacy `split_policy` field was removed in V4).

## Emergency Recovery Procedures

### Training Shows Zero Seizures
**Symptom**: "Seizure ratio: 0%" in logs
**Cause**: Missing or corrupted manifest file
```bash
# Force manifest rebuild locally
BGB_FORCE_MANIFEST_REBUILD=1 python -m src scan-cache --cache-dir cache/tusz/train

# Re-upload to S3 (including JSONs!)
aws s3 sync cache/tusz/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ \
  --exclude "*.log" --exclude "__pycache__/*"

# Verify manifest uploaded
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/manifest.json

# Re-run Modal populate-cache
modal run --detach deploy/modal/app.py --action populate-cache
```

### Modal populate-cache Stops
**Symptom**: Function exits after ~8 minutes
**Cause**: Terminal disconnection without --detach
```bash
# Always use --detach for long-running commands
modal run --detach deploy/modal/app.py --action populate-cache

# Alternative: Use tmux for safety
tmux new -s populate
modal run deploy/modal/app.py --action populate-cache
# Ctrl+B then D to detach
tmux attach -t populate  # To reattach
```

### Cache Corruption
**Symptom**: Unexpected errors during data loading
```bash
# Clean Modal cache completely
modal run deploy/modal/app.py --action clean-cache

# Delete local cache
rm -rf cache/tusz/

# Rebuild from scratch (takes 2-3 hours)
make train-local  # Will rebuild cache automatically

# Upload fixed cache to S3
aws s3 sync cache/tusz/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/ \
  --exclude "*.log" --exclude "__pycache__/*"

# Repopulate Modal
modal run --detach deploy/modal/app.py --action populate-cache
```

### Manifest Issues Quick Reference
| Issue | Solution |
|-------|----------|
| No train manifest | `python -m src scan-cache --cache-dir cache/tusz/train` |
| No dev manifest | `python -m src scan-cache --cache-dir cache/tusz/dev` (optional) |
| Stale manifest | `BGB_FORCE_MANIFEST_REBUILD=1` before scan-cache |
| Upload excluded JSONs | Remove `--exclude "*.json"` from S3 sync |
| Verify manifest size | Train: ~27MB, Dev: ~13MB |
