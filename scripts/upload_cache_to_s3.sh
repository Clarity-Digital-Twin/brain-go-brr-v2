#!/bin/bash

# Script to upload rebuilt cache to S3 after fixing preprocessing
# This uploads the locally rebuilt cache with outlier clipping to S3

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

# Upload to S3
echo "Uploading train split to S3..."
~/.local/bin/aws s3 sync cache/tusz/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ \
    --exclude "*.json" \
    --exclude "*.log"

echo "Uploading dev split to S3..."
~/.local/bin/aws s3 sync cache/tusz/dev/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ \
    --exclude "*.json" \
    --exclude "*.log"

echo ""
echo "================================================"
echo "✅ Cache uploaded to S3 successfully!"
echo "Train files: $TRAIN_COUNT"
echo "Dev files: $DEV_COUNT"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Run 'modal run deploy/modal/app.py --action populate-cache' to copy to Modal SSD"
echo "2. Run Modal training with the fixed cache"