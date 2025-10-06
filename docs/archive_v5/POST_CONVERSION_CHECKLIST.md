# Post-Conversion Checklist for Mmap Cache Migration

**Run this checklist AFTER conversion completes**

## Phase 1: Local Validation (~1 hour)

```bash
# 1. Verify conversion completed
tmux attach -t cache-convert  # Check for "✅ All files converted successfully!"
du -sh cache/tusz_mmap/       # Should be ~400-500 GB

# 2. Regenerate manifests (CRITICAL!)
bash scripts/regenerate_manifests.sh

# Expected output:
# - "Manifest created from NPY (mmap): X partial, Y full, Z no-seizure"
# - Seizure ratio: ~8-12%

# 3. Run quality checks and tests
make q
make test

# Expected: All 342+ tests pass

# 4. Run smoke test
make s

# Watch for:
# - [CACHE] Format: NPY (mmap)
# - RSS <2 GB
# - Completes in <10 min

# 5. Benchmark memory usage
python - <<'PY'
import psutil, numpy as np
from pathlib import Path

files = sorted(Path('cache/tusz_mmap/train').glob('*_data.npy'))[:100]
start_rss = psutil.Process().memory_info().rss / (1024**3)

mmaps = []
for f in files:
    mmap = np.load(f, mmap_mode='r')
    mmaps.append(mmap)
    _ = mmap[0]

end_rss = psutil.Process().memory_info().rss / (1024**3)
print(f'RSS increase: {end_rss - start_rss:.2f} GB')
print(f'Expected if fully loaded: {sum(m.nbytes for m in mmaps) / (1024**3):.2f} GB')
PY

# Expected:
# - RSS increase: <1 GB
# - Expected if fully loaded: ~8-10 GB (proves OS paging works!)
```

## Phase 2: S3 Upload (~2 hours)

```bash
# 6. Upload to S3
aws s3 sync cache/tusz_mmap/train/ \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/ \
  --exclude "*.pyc" --exclude "__pycache__/*"

aws s3 sync cache/tusz_mmap/dev/ \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/dev/ \
  --exclude "*.pyc" --exclude "__pycache__/*"

# 7. Verify upload
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/manifest.json
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/dev/manifest.json
```

## Phase 3: Modal Deployment (~3 hours)

```bash
# 8. Populate Modal cache
modal run --detach deploy/modal/app.py --action populate-cache

# Monitor:
modal app logs <app-id>

# Expected output:
# - "Copied 4667 data files + 4667 labels files"
# - "Copied 1832 data files + 1832 labels files"
# - "Format: Memory-mapped NPY (2025 ML best practice)"

# 9. Verify Modal cache
modal run deploy/modal/app.py --action check-cache

# Expected:
# - ✅ 4667 *_data.npy + 4667 *_labels.npy in train/
# - ✅ 1832 *_data.npy + 1832 *_labels.npy in dev/

# 10. Run Modal smoke test
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Watch for (first 15 minutes):
# - [CACHE] Format: NPY (mmap)
# - data_time ≪ compute_time
# - Worker RSS <2 GB
# - Validation epoch <2 min
```

## Phase 4: Full Training

```bash
# 11. Launch full training (if smoke passes)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor:
modal app logs <app-id>
```

## Success Criteria

- [ ] make q && make test passes
- [ ] Smoke test completes in <10 min
- [ ] Memory benchmark shows <1 GB RSS increase
- [ ] Modal smoke runs without OOM
- [ ] Validation epoch <2 min
- [ ] Modal cache populated correctly

## If Something Fails

**Rollback procedure:**
```bash
# 1. Update configs to point back to NPZ cache
sed -i 's|cache/tusz_mmap|cache/tusz|g' configs/local/*.yaml
sed -i 's|cache/tusz_mmap|cache/tusz|g' configs/modal/*.yaml

# 2. Old NPZ cache still exists - system will work immediately
make s  # Should still work with NPZ cache
```

**Get help:**
- Check MMAP_MIGRATION_STATUS.md for full context
- Review TECHNICAL_DEBT.md for original problem statement
- Review CACHE_PIPELINE_SSOT.md for architecture details

