# Pipeline Fix Plan - Brain-Go-Brr V3.2.0

**Priority**: HIGH - Current pipeline has critical gaps that waste time and money

## Executive Summary

The cache/manifest pipeline works but requires extensive manual coordination and has several breaking points. This plan provides immediate fixes and long-term improvements to make the system production-ready.

## Critical Issues (Fix Immediately)

### 1. S3 Upload Script Excludes Manifests
**Problem**: `upload_cache_to_s3.sh` uses `--exclude "*.json"`, preventing manifest upload
**Impact**: Modal must rebuild manifests (2+ hours) or training gets zero seizures
**Fix**:
```bash
# In scripts/upload_cache_to_s3.sh, change:
aws s3 sync cache/tusz/train/ s3://... --exclude "*.json" --exclude "*.log"
# To:
aws s3 sync cache/tusz/train/ s3://... --exclude "*.log" --include "*.json"
```

### 2. Dev Manifest Not Auto-Generated
**Problem**: Training only creates train manifest, not dev
**Impact**: Manual intervention required every time
**Fix**: Update `src/brain_brr/train/loop.py` to scan both splits:
```python
# After cache building completes
for split in ['train', 'dev']:
    manifest_path = cache_dir / split / 'manifest.json'
    if not manifest_path.exists():
        scan_existing_cache(cache_dir / split)
```

### 3. Modal Commands Missing --detach
**Problem**: Documentation doesn't emphasize --detach requirement
**Impact**: Functions stop after ~8 minutes when terminal disconnects
**Fix**: Update all documentation and examples to use:
```bash
modal run --detach deploy/modal/app.py --action populate-cache
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

## Quick Wins (< 1 Hour to Implement)

### 1. Add Makefile Targets
```makefile
# In Makefile
.PHONY: create-manifests upload-cache populate-modal train-modal

create-manifests:
	@echo "Creating manifests for train and dev..."
	$(PYTHON) -m src scan-cache --cache-dir cache/tusz/train
	$(PYTHON) -m src scan-cache --cache-dir cache/tusz/dev
	@echo "✅ Manifests created"

upload-cache: create-manifests
	@echo "Uploading cache to S3..."
	./scripts/upload_cache_to_s3_fixed.sh
	@echo "✅ Cache uploaded with manifests"

populate-modal:
	@echo "Populating Modal cache from S3..."
	modal run --detach deploy/modal/app.py --action populate-cache
	@echo "✅ Started cache population - monitor with: modal app list"

train-modal: populate-modal
	@echo "Starting Modal training..."
	modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
	@echo "✅ Training started - monitor with: modal app list"

# Complete pipeline
deploy-modal: upload-cache populate-modal train-modal
	@echo "✅ Full Modal deployment pipeline initiated"
```

### 2. Create Fixed Upload Script
```bash
#!/bin/bash
# scripts/upload_cache_to_s3_fixed.sh

set -e

BUCKET="s3://brain-go-brr-eeg-data-20250919"

echo "Creating any missing manifests..."
python -m src scan-cache --cache-dir cache/tusz/train 2>/dev/null || true
python -m src scan-cache --cache-dir cache/tusz/dev 2>/dev/null || true

echo "Uploading train cache with manifests..."
aws s3 sync cache/tusz/train/ ${BUCKET}/cache/tusz/train/ \
    --exclude "*.log" \
    --exclude "__pycache__/*"

echo "Uploading dev cache with manifests..."
aws s3 sync cache/tusz/dev/ ${BUCKET}/cache/tusz/dev/ \
    --exclude "*.log" \
    --exclude "__pycache__/*"

echo "Verifying manifests uploaded..."
aws s3 ls ${BUCKET}/cache/tusz/train/manifest.json
aws s3 ls ${BUCKET}/cache/tusz/dev/manifest.json

echo "✅ Upload complete with manifests!"
```

### 3. Add Validation Command
```python
# src/brain_brr/cli/cli.py - add new command

@cli.command("validate-pipeline")
def validate_pipeline() -> None:
    """Validate the entire cache pipeline is ready."""
    checks = []

    # Check local cache
    train_npz = len(list(Path("cache/tusz/train").glob("*.npz")))
    dev_npz = len(list(Path("cache/tusz/dev").glob("*.npz")))
    checks.append(("Local train NPZ files", train_npz, 4667))
    checks.append(("Local dev NPZ files", dev_npz, 1832))

    # Check manifests
    train_manifest = Path("cache/tusz/train/manifest.json").exists()
    dev_manifest = Path("cache/tusz/dev/manifest.json").exists()
    checks.append(("Train manifest", train_manifest, True))
    checks.append(("Dev manifest", dev_manifest, True))

    # Check S3 (if AWS CLI available)
    try:
        import subprocess
        result = subprocess.run(
            ["aws", "s3", "ls", "s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/manifest.json"],
            capture_output=True
        )
        s3_manifest = result.returncode == 0
        checks.append(("S3 manifests", s3_manifest, True))
    except:
        checks.append(("S3 manifests", "Unable to check", True))

    # Display results
    all_pass = True
    for name, actual, expected in checks:
        status = "✅" if actual == expected else "❌"
        print(f"{status} {name}: {actual} (expected: {expected})")
        if actual != expected:
            all_pass = False

    if all_pass:
        print("\n✅ Pipeline ready for deployment!")
    else:
        print("\n❌ Pipeline has issues - see above")
        sys.exit(1)
```

## Medium-Term Improvements (1 Day)

### 1. Automated Manifest Generation
Update `EEGWindowDataset` to always create manifests after cache building:
```python
# In src/brain_brr/data/datasets.py
def _build_cache_complete_hook(self, cache_dir: Path):
    """Called after cache building completes."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        logger.info(f"Auto-generating manifest for {cache_dir}")
        scan_existing_cache(cache_dir)
```

### 2. Improve Modal populate_cache
Add progress reporting and validation:
```python
def populate_cache():
    # ... existing code ...

    # Add progress reporting
    def copy_with_progress(src, dst, desc):
        files = list(src.glob("*.npz"))
        for i, file in enumerate(files):
            if i % 100 == 0:
                logger.info(f"[PROGRESS] {desc}: {i}/{len(files)} files")
            shutil.copy2(file, dst / file.name)

    # Add validation
    expected = {"train": 4667, "dev": 1832}
    for split in ["train", "dev"]:
        actual = len(list((dst / split).glob("*.npz")))
        if actual != expected[split]:
            logger.warning(f"[WARNING] {split}: {actual} files (expected {expected[split]})")

    # Verify manifests
    for split in ["train", "dev"]:
        manifest = dst / split / "manifest.json"
        if not manifest.exists():
            logger.error(f"[ERROR] Missing manifest: {manifest}")
            raise FileNotFoundError(f"Manifest not found: {manifest}")
```

### 3. Add Pipeline Status Dashboard
Create a simple status checker:
```python
# scripts/check_pipeline_status.py
#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
from rich.table import Table
from rich.console import Console

console = Console()

def check_pipeline():
    table = Table(title="Pipeline Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Local cache
    train_npz = len(list(Path("cache/tusz/train").glob("*.npz")))
    dev_npz = len(list(Path("cache/tusz/dev").glob("*.npz")))
    table.add_row(
        "Local Cache",
        "✅" if train_npz == 4667 and dev_npz == 1832 else "❌",
        f"Train: {train_npz}/4667, Dev: {dev_npz}/1832"
    )

    # Manifests
    manifests_exist = (
        Path("cache/tusz/train/manifest.json").exists() and
        Path("cache/tusz/dev/manifest.json").exists()
    )
    table.add_row(
        "Manifests",
        "✅" if manifests_exist else "❌",
        "Both train and dev" if manifests_exist else "Missing"
    )

    # S3 status
    try:
        s3_check = subprocess.run(
            ["aws", "s3", "ls", "s3://brain-go-brr-eeg-data-20250919/cache/tusz/"],
            capture_output=True, text=True
        )
        s3_status = "✅" if s3_check.returncode == 0 else "❌"
        s3_details = "Connected" if s3_check.returncode == 0 else "Not accessible"
    except:
        s3_status = "⚠️"
        s3_details = "AWS CLI not found"
    table.add_row("S3 Bucket", s3_status, s3_details)

    # Modal status
    try:
        modal_check = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True, text=True
        )
        modal_status = "✅" if modal_check.returncode == 0 else "❌"
        modal_details = "Connected" if modal_check.returncode == 0 else "Not authenticated"
    except:
        modal_status = "⚠️"
        modal_details = "Modal CLI not found"
    table.add_row("Modal", modal_status, modal_details)

    console.print(table)

if __name__ == "__main__":
    check_pipeline()
```

## Long-Term Vision (1 Week)

### 1. Direct Modal Upload (Skip S3)
Investigate using Modal's storage APIs to upload directly from local to Modal volume:
```python
# Potential approach (needs research)
modal volume upload cache/tusz/ brain-go-brr-results:/cache/tusz/
```

### 2. CI/CD Integration
GitHub Actions workflow:
```yaml
name: Deploy to Modal
on:
  push:
    branches: [main]
    paths:
      - 'configs/modal/*.yaml'
      - 'src/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Pipeline
        run: python -m src validate-pipeline
      - name: Deploy to Modal
        run: |
          modal run --detach deploy/modal/app.py \
            --action train \
            --config configs/modal/train.yaml
```

### 3. Pipeline Orchestration with Prefect/Airflow
Create a DAG that handles:
- Cache validation
- S3 sync
- Modal population
- Training launch
- Progress monitoring
- Failure recovery

## Implementation Priority

1. **TODAY**: Fix upload script to include manifests
2. **TODAY**: Add --detach to all Modal commands
3. **TOMORROW**: Implement Makefile targets
4. **THIS WEEK**: Add validation command
5. **NEXT WEEK**: Automate manifest generation
6. **FUTURE**: Direct Modal upload investigation

## Success Metrics

After implementing fixes:
- Zero manual manifest creation required
- Single command deployment: `make deploy-modal`
- No training failures due to missing manifests
- 100% of Modal runs survive disconnection
- Pipeline validation before any deployment

## Testing Plan

1. **Local test**: Delete cache, rebuild, verify manifests auto-created
2. **S3 test**: Run fixed upload script, verify JSONs uploaded
3. **Modal test**: Run populate-cache, verify manifests copied
4. **E2E test**: Fresh deployment from scratch with single command
5. **Failure test**: Disconnect during populate-cache, verify continues

## Rollback Plan

If fixes cause issues:
1. Revert to manual process documented in WORKFLOW.md
2. Use original upload script (accepting manifest rebuilding cost)
3. Manually run scan-cache before each deployment
4. Document any new issues discovered

---

**Priority**: Implement fixes 1-3 immediately to stop wasting A100 compute on cache rebuilding
**Estimated Savings**: $100-300 per training run
**Implementation Time**: 2-4 hours for critical fixes
**Testing Time**: 1-2 hours validation