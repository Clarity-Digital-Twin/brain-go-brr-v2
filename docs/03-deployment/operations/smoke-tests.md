# Smoke Tests (Pipeline and Unit)

## Purpose
- Verify the end-to-end training pipeline quickly without full data cost.
- Catch integration issues (EDF I/O, channel mapping, preprocessing, cache) early.
- Do not assess model quality; only mechanics and stability.

## Two Smoke Test Modes
- Pipeline smoke test (real data, minimal files): Exercises the full stack with production code paths; keep runtime under ~30s.
- Unit smoke test (synthetic tensors): Validates core training loop mechanics in <1s; no I/O.

## Common Pitfall (and Root Cause)
- Symptom: “Checking 80 windows for seizures…” hangs or exits with “No seizures found”.
- Cause: Balanced sampler probes many windows by calling `dataset[idx]`, which triggers EDF loads. With few files (alphabetical order), early slices may contain no seizures.

## Professional Fix (Explicit Test Mode)
- Use an explicit env flag to bypass expensive data-quality checks during smoke runs:
  - `BGB_SMOKE_TEST=1` → skip seizure sampling checks; fall back to default uniform sampling.
- This preserves production behavior (fail fast on bad data) while enabling fast pipeline validation in CI/local.

### Training Loop Behavior (concept)
```
if no_balanced_sampler:
  if BGB_SMOKE_TEST==1:
    log("[SMOKE TEST] No seizures found – using uniform sampling (pipeline validation only)")
    use_default_sampling()
  else:
    fatal("No seizures found in sample – aborting to protect training")
```

## Usage
### Pipeline smoke (real data)
```bash
export BGB_SMOKE_TEST=1
export BGB_LIMIT_FILES=2
python -m src train configs/local/smoke_tcn.yaml
```

### Unit smoke (synthetic)
```bash
pytest tests/unit/ -k "train or tcn" -q
```

### Modal smoke
```bash
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

## What Smoke Tests Validate
- Model init and forward across the active path (TCN → Bi‑Mamba → decoder → head).
- Loss/gradients, optimizer update, checkpoint I/O, CUDA memory behavior.
- Shapes and absence of NaNs/infs.

What they do not validate
- Convergence, TAES metrics, or seizure quality.

## Quick Checklist
- Set `BGB_SMOKE_TEST=1` for smoke; unset for real training.
- Limit files (`BGB_LIMIT_FILES=2`) for fast I/O.
- Expect an initial LR scheduler warning on first batch to be suppressed or harmless.
- Verify batch seizure ratio only on real runs; smoke runs may be uniform.

## Reference
- Discovery notes and analysis were consolidated from prior docs: SMOKE_TEST_DISCOVERY and SMOKE_TEST_FIX.

