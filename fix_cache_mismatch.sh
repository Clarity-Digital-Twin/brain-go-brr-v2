#!/bin/bash
# Emergency fix for cache directory mismatch
# Run AFTER current training completes

echo "=== CACHE DIRECTORY FIX ==="
echo "This will fix the cache/train vs cache/tusz mismatch"
echo ""

# Check current state
echo "Current cache directories:"
ls -la cache/ 2>/dev/null || echo "No cache directory found"
echo ""

# Check if cache/train exists
if [ -d "cache/train" ]; then
    echo "Found cache/train with $(ls cache/train/*.npz 2>/dev/null | wc -l) files"

    # Check if cache/tusz exists
    if [ -d "cache/tusz" ]; then
        echo "WARNING: cache/tusz already exists!"
        echo "Contents: $(ls cache/tusz/*.npz 2>/dev/null | wc -l) files"
        echo ""
        echo "Options:"
        echo "1) Merge cache/train into cache/tusz"
        echo "2) Replace cache/tusz with cache/train"
        echo "3) Create symlink (recommended for testing)"
        read -p "Choose option [1/2/3]: " option

        case $option in
            1)
                echo "Merging cache/train into cache/tusz..."
                cp -n cache/train/*.npz cache/tusz/
                echo "Merged (skipped existing files)"
                ;;
            2)
                echo "Backing up cache/tusz to cache/tusz.backup..."
                mv cache/tusz cache/tusz.backup
                echo "Moving cache/train to cache/tusz..."
                mv cache/train cache/tusz
                ;;
            3)
                echo "Creating symlink cache/tusz -> cache/train"
                rm -rf cache/tusz
                ln -s $(pwd)/cache/train $(pwd)/cache/tusz
                ;;
        esac
    else
        echo "cache/tusz doesn't exist. Creating symlink..."
        ln -s $(pwd)/cache/train $(pwd)/cache/tusz
        echo "Created symlink: cache/tusz -> cache/train"
    fi
else
    echo "ERROR: cache/train doesn't exist!"
    echo "The cache might be in a different location"
fi

echo ""
echo "Final state:"
ls -la cache/
echo ""
echo "Cache statistics:"
if [ -d "cache/tusz" ]; then
    echo "cache/tusz: $(ls cache/tusz/*.npz 2>/dev/null | wc -l) files"
    echo "Total size: $(du -sh cache/tusz 2>/dev/null | cut -f1)"
fi

echo ""
echo "To test if fix worked, run:"
echo "python -c \"from pathlib import Path; print(f'Cache files found: {len(list(Path(\"cache/tusz\").glob(\"*.npz\")))}')\""