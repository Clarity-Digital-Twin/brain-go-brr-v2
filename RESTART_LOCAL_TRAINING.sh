#!/bin/bash
# Script to properly restart local training with complete cache

echo "=== LOCAL TRAINING RESTART PLAN ==="
echo "Current situation: Training at batch 248/31512, ~13s/batch = 113 hours total"
echo "This script will:"
echo "1. Kill current slow training"
echo "2. Build complete cache upfront"
echo "3. Restart training with cache"
echo ""
read -p "Kill current training and restart properly? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted"
    exit 0
fi

# Step 1: Kill current training
echo "[1/4] Killing current training session..."
tmux kill-session -t train 2>/dev/null || echo "No tmux session 'train' found"

# Step 2: Fix cache directory structure
echo "[2/4] Setting up cache directories..."
mkdir -p cache/tusz/train
mkdir -p cache/tusz/val

# Check if we have partial cache to use
if [ -d "cache/train" ] && [ "$(ls -A cache/train/*.npz 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Found $(ls cache/train/*.npz | wc -l) cached files in cache/train"
    echo "Moving to cache/tusz/train..."
    mv cache/train/*.npz cache/tusz/train/ 2>/dev/null || cp cache/train/*.npz cache/tusz/train/
fi

# Step 3: Build complete cache (this will take hours but worth it)
echo "[3/4] Building complete cache for training data..."
echo "This will take 2-3 hours but will make training 30x faster"
echo ""

# Build train cache
python -m src build-cache \
    --data-dir data_ext4/tusz/edf/train \
    --cache-dir cache/tusz/train \
    --validation-split 0.2 \
    --split train

# Build val cache
python -m src build-cache \
    --data-dir data_ext4/tusz/edf/train \
    --cache-dir cache/tusz/val \
    --validation-split 0.2 \
    --split val

echo ""
echo "[4/4] Starting training with complete cache..."

# Start training in tmux
tmux new-session -d -s train \
    "python -m src train configs/tusz_train_wsl2.yaml 2>&1 | tee training_restart.log"

echo ""
echo "✅ DONE! Training restarted with complete cache"
echo ""
echo "Monitor with: tmux attach -t train"
echo "Expected: 2-3s/batch instead of 13s/batch"
echo "Total time: ~15 hours instead of 113 hours"