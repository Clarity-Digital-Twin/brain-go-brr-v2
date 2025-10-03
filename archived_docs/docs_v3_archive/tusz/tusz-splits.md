# TUSZ Official Splits (Patient‑Disjoint)

Purpose
- Use TUSZ as designed: train on `train/`, validate/tune on `dev/`, evaluate once on `eval/`.
- Eliminate patient leakage by construction and verify at runtime.

Directory layout (v2.0.3)
- `edf/train/` → 579 patients (~4667 EDFs)
- `edf/dev/`   → 53 patients (~1832 EDFs)
- `edf/eval/`  → 43 patients (~865 EDFs)

Policy (SSOT)
- Training: use only `train/` patients.
- Validation & early stopping: use only `dev/` patients.
- Final test: run once on `eval/` after HPs are frozen; report final numbers.

Config (local and Modal)
```yaml
data:
  data_dir: <parent>/edf           # Parent containing train/dev/eval
  cache_dir: <cache_root>/tusz     # Will contain {train,dev} subdirs matching TUSZ
  split_policy: official_tusz      # Enforce patient‑disjoint official splits
```

Runtime verification
- The loader builds patient sets by split and asserts overlap is empty.
- Logs print patient/file counts per split and “✅ PATIENT DISJOINTNESS VERIFIED”.

Cache structure
- Local: `cache/tusz/{train,dev}/` (smoke uses SAME cache with `BGB_LIMIT_FILES` env var).
- Modal: `/results/cache/tusz/{train,dev}/` (persistent SSD volume; no S3 mounts).
- CRITICAL: We use 'dev' naming to match TUSZ's official split names, NOT 'val'!

Guardrails
- If any overlap is detected between train/dev patients, training aborts.
- Balanced dataset is used only for training (validation uses standard dataset).

Related
- Data overview: `docs/02-data/overview.md`
- Preflight & troubleshooting: `docs/tusz/tusz-preflight.md`
- Cache & manifest: `docs/02-data/cache-layout.md`
