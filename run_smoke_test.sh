#!/bin/bash
# Proper smoke test script for V3 architecture
# Limits to 3 files for quick validation

echo "Running V3 smoke test with limited files..."
echo "=========================================="

# Set environment variables for smoke test
export BGB_LIMIT_FILES=3        # Limit to 3 files for quick test
export BGB_SMOKE_TEST=1          # Skip seizure sampling checks

# Run the smoke test
.venv/bin/python -m src train configs/local/smoke.yaml

echo "Smoke test complete!"