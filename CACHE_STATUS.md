# Cache Status and S3 Upload Plan

## Current Status (Sep 24, 2025 23:13)

### Local Full Training Cache
- **Location**: `cache/tusz/`
- **Progress**: Building dev set (941/1832 files, ~51%)
- **Expected completion**: Overnight
- **Expected final size**:
  - Train: 3734 NPZ files
  - Dev: 1832 NPZ files (val in our config)
  - Total: ~5566 files, ~50GB

### Current Cache Progress
```
cache/tusz/
├── train/     # 20 files built (will reach 3734)
├── dev/       # 925 files built (will reach 1832)
└── [NO METADATA YET - will be created after completion]
```

### Cleaned Up Old Caches
Successfully deleted ALL old/contaminated caches:
- ✅ Deleted: cache/smoke_tcn/
- ✅ Deleted: cache/v2.6_full/
- ✅ Deleted: cache/v2.6_smoke/
- ✅ Deleted: cache/smoke/
- ✅ Deleted: cache/tcn_full/
- ✅ Deleted: cache/data/

**Only `cache/tusz/` remains** - the clean, patient-disjoint cache being built.

## Tomorrow's S3 Upload Plan

After cache completes overnight, follow CACHE_UPLOAD_PROCEDURE.md:

1. **Verify completion**:
```bash
ls cache/tusz/train/*.npz | wc -l  # Should be 3734
ls cache/tusz/dev/*.npz | wc -l    # Should be 1832
```

2. **Upload to S3**:
```bash
aws s3 sync cache/tusz/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/ \
    --exclude "*.log" \
    --exclude "*.tmp" \
    --storage-class STANDARD_IA
```

3. **Verify upload**:
```bash
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/ --recursive | wc -l
# Should show 5566+ files
```

## Key Points
- Cache is building with **official_tusz split policy** (patient-disjoint)
- All old contaminated caches have been deleted
- Modal will use this S3 cache to save ~$10 per training run
- Modal smoke test uses separate mini-cache at `/results/cache/smoke/`