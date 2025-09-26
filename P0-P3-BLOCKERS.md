# P0–P3 Blockers: Deep Audit and Action Plan

This document lists potential blockers and risks by severity (P0–P3), with fast checks and concrete mitigations. File references are provided to exact starting lines.

## P0 Blockers (Must Fix Before Training)
- GPU stack version mismatch (Mamba + PyG)
  - Symptom: Import errors or silent Conv1d fallback; training quality degrades or fails to start.
  - Checks:
    - Verify Mamba import: `.venv/bin/python -c "from mamba_ssm import Mamba2; print('OK')"`
    - Required versions and install order in `AGENTS.md` and `SETUP.md`.
  - Key refs:
    - `pyproject.toml:28` (torch==2.2.2), `pyproject.toml:62` (mamba-ssm notes), `AGENTS.md:131`, `SETUP.md:71`
  - Mitigation: `make setup` then `make setup-gpu`; use PyG prebuilt wheels; CUDA Toolkit 12.1 only.

- Cache presence and label coverage (no seizures)
  - Symptom: Training aborts; CLI reports no seizures in manifest.
  - Checks:
    - Build and scan: `python -m src build-cache ...` and `python -m src scan-cache --cache-dir <dir>`
    - CLI will exit if zero seizure windows found.
  - Key refs:
    - `src/brain_brr/cli/cli.py:202` (build-cache), `src/brain_brr/cli/cli.py:262` (scan-cache)
    - `src/brain_brr/data/datasets.py:290` (manifest required by BalancedSeizureDataset)
  - Mitigation: Rebuild cache with correct CSV_BI labels present; ensure `data_dir` points to TUSZ train/dev/eval.

- PyG required when graph.enabled=true (default configs)
  - Symptom: ImportError constructing GNN; crash at model init.
  - Checks: `pip show torch_geometric`; ensure companion wheels installed.
  - Key refs:
    - `src/brain_brr/models/gnn_pyg.py:15` (HAS_PYG), `configs/local/train.yaml:41` (graph.enabled: true)
  - Mitigation: Install PyG wheels for torch 2.2.2 + cu121 or set `model.graph.enabled: false`.

- Data normalization/outlier policy requires cache rebuild
  - Symptom: Extreme values in windows; unstable training if old cache used.
  - Checks: Inspect NPZ ranges; expect ~[-10, 10].
  - Key refs:
    - `src/brain_brr/data/preprocess.py:66` (clip to ±10σ), `src/brain_brr/data/preprocess.py:71` (nan_to_num)
  - Mitigation: Remove old cache and rebuild after 2025-09-26 change: `rm -rf cache/tusz && python -m src build-cache ...`

## P1 Risks (High Priority)
- Mixed precision on RTX 4090 can induce NaNs
  - Status: Disabled by default locally; enabled on A100.
  - Key refs: `configs/local/train.yaml:120` (mixed_precision: false), `configs/modal/train.yaml:109` (true)
  - Mitigation: Keep false on 4090; consider BF16 if needed; validate with smoke test first.

- Early-epoch gradient spikes (TCN)
  - Symptom: NaN gradients after tens of batches under some data mixes.
  - Guards: Gradient clipping + optional gradient sanitization.
  - Key refs:
    - `src/brain_brr/train/loop.py:560` (sanitization path), `src/brain_brr/train/loop.py:606` (logit sanitize fallback)
    - `configs/local/train.yaml:113` (gradient_clip)
  - Mitigation: Export `BGB_SANITIZE_GRADS=1` for first epochs; keep `gradient_clip: 0.1`.

- Mamba headdim constraint
  - Symptom: ValueError at init if `(d_model*expand)/headdim` not integer or not multiple of 8 (perf warning).
  - Key refs: `src/brain_brr/models/mamba.py:61` (validation), `src/brain_brr/models/detector.py:109` (headdim=64 ok)
  - Mitigation: Keep `d_model=512, expand=2, headdim=64` as-is; don’t override schema defaults.

- Final logits safety bounds must stay consistent with loss
  - Symptom: Overflow in BCE-with-logits if logits unclamped.
  - Key refs:
    - `src/brain_brr/models/detector.py:312` (nan_to_num), `src/brain_brr/models/detector.py:314` (clamp ±100)
    - `src/brain_brr/train/loop.py:180` (FocalLoss clamps) and `src/brain_brr/train/loop.py:206` (prob clamps)
  - Mitigation: Keep detector output clamp at ±100; ensure loss clamps remain.

## P2 Risks (Medium Priority)
- Build-cache lacks a quick `--limit-files` option
  - Impact: Slow rebuilds hamper iteration; env var `BGB_LIMIT_FILES` not wired here.
  - Key ref: `src/brain_brr/cli/cli.py:202` (build-cache options)
  - Mitigation: Add `--limit-files` to slice `edf_files`; or rebuild a small split dir.

- Env variables cached at import time
  - Impact: Changing envs at runtime has no effect on already-imported modules.
  - Key ref: `src/brain_brr/utils/env.py:13` (cache note), `src/brain_brr/utils/env.py:134` (safe_clamp)
  - Mitigation: Restart process after changing `BGB_*`; document in workflows.

- Silent Conv1d fallback when Mamba missing
  - Impact: Functional mismatch vs SSM; acceptable for CI/CPU, not for training quality.
  - Key refs: `src/brain_brr/models/mamba.py:95` (fallback), `src/brain_brr/models/mamba.py:335` (complexity text)
  - Mitigation: Ensure Mamba installed for training; surface a WARNING is already printed.

- Evaluate CLI is partial (dry-run path only)
  - Impact: Users expecting full evaluation may be blocked.
  - Key refs: `src/brain_brr/cli/cli.py:304` (evaluate stub)
  - Mitigation: Keep using training notebooks/scripts; plan full evaluator or clearly mark CLI as partial.

## P3 Risks (Low Priority)
- TCN minimal vs pytorch-tcn implementation differences
  - Impact: Minor discrepancies; minimal path is default if pytorch-tcn missing.
  - Key refs: `src/brain_brr/models/tcn.py:17` (HAS_PYTORCH_TCN), `src/brain_brr/models/tcn.py:126` (MinimalTCN)
  - Mitigation: Optional install `pytorch-tcn==1.2.3` or stay on MinimalTCN (stable).

- Dynamic PE numerical quirks (ill-conditioned graphs)
  - Impact: Spurious NaNs in eigenvectors; fallbacks engage, slight perf cost.
  - Key refs: `src/brain_brr/models/gnn_pyg.py:220` (eigenvalue clamp), `src/brain_brr/models/gnn_pyg.py:240` (nan_to_num)
  - Mitigation: Keep current clamps and cached PE fallback; tune `semi_dynamic_interval` if needed.

## Stability Guardrails (Implemented)
- 3-tier clamping policy
  - Input: TCN/Mamba clamp to ±10 (`src/brain_brr/models/tcn.py:240`, `src/brain_brr/models/mamba.py:173`)
  - Internal features: Optional safe clamp to ±50 (`src/brain_brr/models/tcn.py:246`)
  - Output logits: Clamp to ±100 (`src/brain_brr/models/detector.py:314`)

- Defensive sanitization
  - Data: z-score + clip ±10σ + nan_to_num (`src/brain_brr/data/preprocess.py:66`)
  - Detector: nan_to_num on prelogits/logits (`src/brain_brr/models/detector.py:306`, `src/brain_brr/models/detector.py:312`)
  - Loss: clamp logits and probabilities (`src/brain_brr/train/loop.py:180`)
  - Training fallback sanitizer for non-finite logits (`src/brain_brr/train/loop.py:606`)

## Quick Triage Checklist
- `.venv/bin/python -c "from mamba_ssm import Mamba2; print('OK')"` → OK
- `pip show torch-geometric` and required extension wheels → installed
- `python -m src scan-cache --cache-dir cache/tusz/train` → partial/full/no-seizure > 0
- `python -c "import numpy as np; d=np.load('cache/tusz/train/XXXX_windows.npz'); print(d['windows'].min(), d['windows'].max())"` → within ~[-10, 10]
- Local config uses `mixed_precision: false` and `gradient_clip: 0.1` (RTX 4090)

## Recommended Next Improvements
- Add `--limit-files` to build-cache for faster rebuilds (`src/brain_brr/cli/cli.py:202`).
- Add a finiteness smoke test on one cached window (unit/integration) to CI.
- Consider enabling BF16 on A100 only after a stable epoch.
- Surface a hard error (not just warning) when Mamba missing but device is CUDA.

—

Last reviewed: 2025-09-26
