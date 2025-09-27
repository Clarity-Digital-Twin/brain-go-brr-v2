# Troubleshooting

Common issues

- Wrong cache dir: local `cache/tusz/`, Modal `/results/cache/tusz/` (persistent SSD)
- No seizures in batches: enable `use_balanced_sampling`
- NaN losses on 4090: set `mixed_precision: false`
- Modal stuck: increase CPU (24) and RAM (96GB)
- PyG install fails: use prebuilt wheels
- Evaluate CLI exits early:
  - "No config found" — pass `--config` or use a checkpoint with embedded config
  - "No EDF files found" — verify `*.edf` exist under the provided data path

V3 NaN issues

- Primary contributing factor: dynamic PE eigendecomposition on poorly initialized adjacency.
- Default configs enable dynamic PE with safeguards; on consumer GPUs (RTX 4090), prefer increasing `graph.semi_dynamic_interval` first; as a last resort set `use_dynamic_pe: false`.
- Additional safeguards:
  - Edge similarity is clamped at the source with a configurable margin: `graph.edge_similarity_margin` (default 0.01)
  - Edge lift uses bounded activation + normalization (tanh + LayerNorm when enabled); no hardcoded `[-3,3]` clamp remains
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

NaN logits root cause quick summary

- Tiny batches (e.g., 1) and aggressive FP16 can amplify numerical noise in dynamic PE and post‑Mamba projections.
- Remedies, in order: increase `batch_size` (≥4), set `training.mixed_precision: false` on RTX 4090, reduce `learning_rate`.
- Env toggles: `export BGB_NAN_DEBUG=1` (extra logging), `export SEIZURE_MAMBA_FORCE_FALLBACK=1` (force Conv1d fallback).

Modal cache hygiene

- Do not use S3 for caches; keep NPZs on Modal volume at `/results/cache/tusz/`
- Modal persistent volume only used for results at `/results/`
- Ensure `data.data_dir: /data/edf` and `data.split_policy: official_tusz`; app verifies patient disjointness on startup.
