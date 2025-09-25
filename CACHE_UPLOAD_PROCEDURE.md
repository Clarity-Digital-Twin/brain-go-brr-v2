# Cache Upload Procedure for Modal

## Overview
This document outlines the procedure to upload locally-built cache to S3/Modal to avoid rebuilding cache on Modal A100 (saves ~2-3 hours and ~$10 compute).

## Prerequisites
- Local cache fully built with patient-disjoint splits
- AWS CLI configured locally
- Modal secret `aws-s3-secret` configured
- S3 bucket: `brain-go-brr-eeg-data-20250919`

## Step 1: Verify Local Cache is Complete

```bash
# Check cache is fully built
ls cache/tusz/train/*.npz | wc -l  # Should be ~3734 files
ls cache/tusz/dev/*.npz | wc -l    # Should be ~933 files

# Verify metadata exists
cat cache/tusz/.cache_metadata.json  # Should show split_policy: "official_tusz"
```

## Step 2: Upload Cache to S3

```bash
# Upload cache to S3 bucket (already mounted in Modal)
# Note: Using separate cache/ prefix to keep raw data and cache separate
aws s3 sync cache/tusz/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/ \
    --exclude "*.log" \
    --exclude "*.tmp" \
    --storage-class STANDARD_IA \
    --no-progress

# Verify upload
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/ --recursive | wc -l
```

## Step 3: Update Modal App to Use S3 Cache

In `deploy/modal/app.py`, add cache mount:

```python
# Add cache mount alongside data mount
cache_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",
    secret=s3_secret,
    key_prefix="cache/tusz/",  # Mount cache separately
    read_only=True,  # Cache is read-only during training
)

# In train function, mount both:
@app.function(
    mounts=[
        modal.Mount.from_local_python_packages("src"),
        modal.Mount.from_local_file("configs/modal/train.yaml"),
    ],
    volumes={
        "/data": data_mount,           # Raw EDF data
        "/s3-cache": cache_mount,       # Pre-built cache from S3
        "/results": results_volume,    # Output results
    },
    ...
)
```

## Step 4: Configure Training to Check S3 Cache First

Update config to check S3 cache before building:

```yaml
# configs/modal/train.yaml
data:
  cache_dir: /results/cache/tusz  # Local Modal volume cache
  s3_cache_dir: /s3-cache         # S3-mounted cache (read-only)
  cache_priority: s3_first         # Check S3 cache first
```

## Step 5: Smart Cache Logic

The training script should check caches in order:
1. Check `/s3-cache` (S3-mounted) - if valid, copy to `/results/cache/tusz`
2. Check `/results/cache/tusz` (Modal volume) - if valid, use it
3. If neither valid, build fresh (but this wastes compute!)

## Alternative: Direct Modal Volume Upload

If S3 sync is slow, use Modal CLI directly:

```bash
# Upload directly to Modal persistent volume
modal volume put brain-go-brr-results cache/tusz /results/cache/tusz

# This uploads from local → Modal volume directly
# Slower than S3 but simpler
```

## Cost Savings

- Building cache on A100: ~2-3 hours @ $3.19/hour = ~$10
- S3 storage: ~50GB @ $0.023/GB/month = ~$1.15/month
- S3 transfer to Modal: FREE (same AWS region with gateway endpoints)

**Total savings per training run: ~$10**

## Important Notes

1. **Region alignment**: Modal runs in us-east-1, ensure S3 bucket is also us-east-1
2. **Cache validation**: Always check `.cache_metadata.json` for `split_policy: "official_tusz"`
3. **No sensitive data**: Cache contains only preprocessed NPZ arrays, no patient identifiers
4. **Version control**: Consider adding cache version to metadata for tracking

## Verification Commands

```bash
# Check S3 cache structure
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ | head -5
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ | head -5

# Check metadata
aws s3 cp s3://brain-go-brr-eeg-data-20250919/cache/tusz/.cache_metadata.json - | jq

# Monitor Modal training to ensure it uses cache
modal app logs <app-id> | grep -i cache
```