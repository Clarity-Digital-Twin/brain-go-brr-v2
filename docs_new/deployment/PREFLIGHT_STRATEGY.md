# Modal & Local Preflight Strategy

Goal: Verify CSV_BI parsing and balanced dataset before burning compute.

## 1) Parser verification (mini-cache)
```bash
# Build a tiny cache (handful of files) and scan manifest
python -m src scan-cache --cache-dir cache/tusz/train
# Expect: partial>0 or full>0; otherwise STOP
```

## 2) Smoke test
```bash
python -m src train configs/smoke_test.yaml
# Verify: [DATASET] Windows with seizures: X/Y (Z%) appears > 0%
```

## 3) Full run
```bash
# Local
python -m src train configs/tusz_train_wsl2.yaml

# Modal
modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml
```

