#!/bin/bash
# MODAL V3 FULL DEPLOYMENT SCRIPT
# Run this after cache population completes!

set -e  # Exit on error

echo "🚀 MODAL V3 DEPLOYMENT - FULL PIPELINE"
echo "======================================="

# Step 1: Verify cache populated
echo ""
echo "📦 Step 1: Verifying cache population..."
echo "Expected: 4667 train + 1832 dev files"
echo "Check logs: tail -100 modal_populate.log"
echo ""
read -p "Is cache population complete? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborting. Wait for cache population to complete."
    exit 1
fi

# Step 2: Test Mamba CUDA
echo ""
echo "🔧 Step 2: Testing Mamba CUDA kernels..."
modal run deploy/modal/app.py --action test-mamba
if [ $? -ne 0 ]; then
    echo "❌ Mamba CUDA test failed!"
    exit 1
fi
echo "✅ Mamba CUDA working!"

# Step 3: Run smoke test
echo ""
echo "🧪 Step 3: Running smoke test (1 epoch, 50 files)..."
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml 2>&1 | tee modal_smoke.log
if [ $? -ne 0 ]; then
    echo "❌ Smoke test failed! Check modal_smoke.log"
    exit 1
fi
echo "✅ Smoke test completed!"

# Step 4: Launch full training
echo ""
echo "🎯 Step 4: Launching FULL V3 TRAINING (100 epochs)..."
echo "This will run for ~100 hours and cost ~$319"
read -p "Launch full training? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml 2>&1 | tee modal_train.log
    echo "✅ Full training launched!"
    echo ""
    echo "Monitor with:"
    echo "  modal app list"
    echo "  modal app logs <app-id>"
    echo "  tail -f modal_train.log"
else
    echo "⏸️ Full training skipped. Run manually:"
    echo "modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml"
fi

echo ""
echo "🎉 V3 DEPLOYMENT COMPLETE!"
echo "=========================="
echo "Next steps:"
echo "1. Monitor training: modal app list"
echo "2. Check W&B: https://wandb.ai/"
echo "3. Celebrate: WE'RE GONNA SHOCK THE TECH WORLD!"