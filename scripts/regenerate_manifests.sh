#!/bin/bash
# Regenerate manifests for mmap cache
# Run this AFTER conversion completes

set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "📋 REGENERATING MANIFESTS FOR MMAP CACHE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /home/jj/proj/brain-go-brr-v2

# Check conversion completed
if [ ! -d "cache/tusz_mmap/train" ] || [ ! -d "cache/tusz_mmap/dev" ]; then
    echo "❌ ERROR: Mmap cache directories not found!"
    echo "   Make sure conversion completed successfully"
    exit 1
fi

# Count NPY files
train_count=$(find cache/tusz_mmap/train -name "*_data.npy" | wc -l)
dev_count=$(find cache/tusz_mmap/dev -name "*_data.npy" | wc -l)

echo "Found NPY files:"
echo "  Train: $train_count"
echo "  Dev:   $dev_count"
echo ""

if [ "$train_count" -lt 4600 ]; then
    echo "⚠️  WARNING: Expected ~4,667 train files, found $train_count"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "▶ Regenerating train manifest..."
python -m src scan-cache --cache-dir cache/tusz_mmap/train
echo "✅ Train manifest complete"
echo ""

echo "▶ Regenerating dev manifest..."
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
echo "✅ Dev manifest complete"
echo ""

# Copy metadata file
if [ -f "cache/tusz/.cache_metadata.json" ]; then
    echo "▶ Copying cache metadata..."
    cp cache/tusz/.cache_metadata.json cache/tusz_mmap/
    echo "✅ Metadata copied"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ MANIFEST REGENERATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Verify manifests:"
echo "  ls -lh cache/tusz_mmap/train/manifest.json"
echo "  ls -lh cache/tusz_mmap/dev/manifest.json"
echo ""
echo "Next step: Update dataset code for mmap"
echo ""
