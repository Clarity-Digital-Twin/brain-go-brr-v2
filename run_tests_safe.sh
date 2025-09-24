#!/bin/bash
# Safe test runner that avoids OOM crashes
# Problem: Running tests with -n auto spawns many processes, each creating V3 models

echo "🧪 Running tests safely to avoid OOM..."
echo "================================================"

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. First run unit tests WITHOUT V3 models (serial to avoid OOM)
echo -e "${CYAN}1/5: Running unit tests (excluding V3 and heavy models)...${NC}"
.venv/bin/pytest tests/unit -n 1 \
    -k "not (v3 or V3 or detector_from_config or SeizureDetector)" \
    -m "not gpu and not performance" \
    --tb=short -q

# 2. Run V3 tests separately in serial
echo -e "${CYAN}2/5: Running V3 tests (serial to avoid OOM)...${NC}"
.venv/bin/pytest tests/unit/models/test_detector_v3.py -n 0 \
    -m "not gpu and not performance" \
    --tb=short -v

# 3. Run integration tests WITHOUT creating full models
echo -e "${CYAN}3/5: Running integration tests (excluding model creation)...${NC}"
.venv/bin/pytest tests/integration -n 1 \
    -k "not (from_config or tcn_integration or gnn_integration)" \
    -m "not gpu and not performance" \
    --tb=short -q

# 4. Run clinical tests (lightweight)
echo -e "${CYAN}4/5: Running clinical validation tests...${NC}"
.venv/bin/pytest tests/clinical -n 2 \
    -m "not performance" \
    --tb=short -q

# 5. Skip performance tests (they're memory intensive)
echo -e "${YELLOW}5/5: Skipping performance tests (run separately with 'make test-performance')${NC}"

echo -e "${GREEN}✅ Safe test run complete!${NC}"
echo ""
echo "For full coverage including heavy tests, run:"
echo "  - Heavy integration: .venv/bin/pytest tests/integration/test_tcn_integration.py -n 0"
echo "  - Performance tests: make test-performance"
echo "  - GPU tests: make test-gpu"