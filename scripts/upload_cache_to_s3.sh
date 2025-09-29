#!/bin/bash

# Script to upload rebuilt cache to S3 after fixing preprocessing
# This uploads the locally rebuilt cache with outlier clipping to S3
#
# CRITICAL: We use 'dev' naming to match TUSZ official splits (not 'val')!
# TUSZ provides train/dev/eval - we use dev for validation during training.

set -e

echo "================================================"
echo "Uploading rebuilt cache to S3..."
echo "================================================"

# Check if local cache exists
if [ ! -d "cache/tusz/train" ] || [ ! -d "cache/tusz/dev" ]; then
    echo "ERROR: Local cache not found at cache/tusz/"
    echo "Please ensure training has completed cache building first"
    exit 1
fi

# Count files
TRAIN_COUNT=$(ls -1 cache/tusz/train/*.npz 2>/dev/null | wc -l)
DEV_COUNT=$(ls -1 cache/tusz/dev/*.npz 2>/dev/null | wc -l)

echo "Found $TRAIN_COUNT train files and $DEV_COUNT dev files"
echo ""

# Upload to S3 (INCLUDING manifests!)
echo "Uploading train split to S3..."
~/.local/bin/aws s3 sync cache/tusz/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ \
    --exclude "*.log" \
    --exclude "__pycache__/*" \
    --exclude ".DS_Store"

echo "Uploading dev split to S3..."
~/.local/bin/aws s3 sync cache/tusz/dev/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ \
    --exclude "*.log" \
    --exclude "__pycache__/*" \
    --exclude ".DS_Store"

echo ""
echo "Verifying manifest upload..."
~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/manifest.json >/dev/null 2>&1 && \
    echo "✅ Train manifest uploaded" || \
    echo "⚠️  Train manifest NOT found on S3!"

~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/manifest.json >/dev/null 2>&1 && \
    echo "✅ Dev manifest uploaded (optional)" || \
    echo "ℹ️  Dev manifest not found (optional for validation)"

echo ""
echo "================================================"
echo "✅ Cache uploaded to S3 successfully!"
echo "Train files: $TRAIN_COUNT"
echo "Dev files: $DEV_COUNT"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Run 'modal run --detach deploy/modal/app.py --action populate-cache' to copy to Modal SSD"
echo "2. Run Modal training with the populated cache"