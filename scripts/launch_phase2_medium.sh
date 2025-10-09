#!/bin/bash
# Launch Phase 2 Medium Validation
# Purpose: Integration test (40-50 files, 5-6 epochs) to surface scaling bugs
# NOT for performance comparison - that's Modal's job

set -e  # Exit on error

echo "=========================================="
echo "Phase 2 Medium Validation Launch"
echo "=========================================="
echo ""
echo "Purpose: Surface scaling bugs before Modal"
echo "  - SSM memory spikes"
echo "  - Optimizer drift"
echo "  - Checkpoint size/integrity"
echo "  - GPU/RAM peaks"
echo "  - Gradient clipping trends"
echo ""
echo "Config: configs/local/phase2_medium_gdn.yaml"
echo "Scale: 40-50 files, 6 epochs"
echo "ETA: ~2-3 hours (RTX 4090)"
echo ""

# Check smoke test completed
if ! grep -q "Training complete" /tmp/phase2_smoke.log 2>/dev/null; then
    echo "⚠️  WARNING: Phase 2 smoke test not completed yet!"
    echo "   Check: tail -f /tmp/phase2_smoke.log"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Verify smoke test passed (no NaNs, no crashes)
if grep -q "NaN" /tmp/phase2_smoke.log 2>/dev/null; then
    echo "❌ ERROR: Phase 2 smoke test had NaNs!"
    echo "   Review: /tmp/phase2_smoke.log"
    exit 1
fi

echo "✅ Smoke test verification passed"
echo ""

# Set environment
export BGB_LIMIT_FILES=50
export BGB_NAN_DEBUG=1

# Launch in tmux
echo "🚀 Launching medium validation in tmux..."
tmux new-session -d -s phase2_medium \
    ".venv/bin/python -m src train configs/local/phase2_medium_gdn.yaml 2>&1 | tee /tmp/phase2_medium.log"

echo ""
echo "✅ Medium validation launched!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t phase2_medium     # Watch live"
echo "  tail -f /tmp/phase2_medium.log   # Follow log"
echo ""
echo "Success criteria:"
echo "  ✅ No NaNs"
echo "  ✅ Loss converges"
echo "  ✅ Gradient clip % < 80% after warmup"
echo "  ✅ GPU < 22GB, RAM < 28GB"
echo "  ✅ Checkpoints save/load correctly"
echo ""
echo "ETA: ~2-3 hours"
echo "=========================================="
