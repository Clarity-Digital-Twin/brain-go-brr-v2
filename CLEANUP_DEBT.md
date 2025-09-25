# Technical Debt & Cleanup Tasks

## ✅ COMPLETED (V3 Implementation Done)
- [x] Implement edge stream (edge_mamba in detector.py:207-229 with 16D projection)
- [x] Assemble learned adjacency (assemble_adjacency in edge_features.py)
- [x] PyG + Laplacian PE canonical (gnn_pyg.py fully implemented)
- [x] Detector consumes learned adjacency (V3 path lines 207-241)
- [x] Configs updated (edge_metric, edge_top_k, edge_threshold, edge_mamba_* all present)

## 🧹 PRIORITY 1: V2/V3 Path Consolidation
- [ ] Remove heuristic adjacency builder (still used by V2 path detector.py:375)
- [ ] Remove graph_builder.py entirely after V2 deprecation
- [ ] Consolidate to single V3 architecture (remove V2 branches)
- [ ] Clean up architecture config (remove 'tcn' vs 'v3' distinction)

## 🧹 PRIORITY 2: Legacy Parameter Cleanup
- [ ] Remove `base_channels`, `encoder_depth`, `rescnn_blocks` from detector.__init__ (lines 48-50, 87-89)
- [ ] Add deprecation warnings before removal
- [ ] Update all test fixtures to stop passing these parameters
- [ ] Remove from conftest.py and test files
- [ ] Clean up schemas.py if referenced there

## 🧹 PRIORITY 3: Test Suite Improvements
- [x] Tests marked with `@pytest.mark.serial` (25 tests properly marked)
- [x] Timeout decorators added (test_latency.py, test_training_edge_cases.py, test_memory.py)
- [ ] Investigate if worker crashes still occur
- [ ] Add resource cleanup fixtures
- [ ] Memory leak investigation (unknown if still an issue)

## 🧹 PRIORITY 4: Future Enhancements (from EvoBrain)
- [ ] Support alternative SNNs for edge stream (GRU/LSTM)
- [ ] Implement full K‑hop SSGConv (currently 1‑hop approximation in pure‑torch)
- [ ] Add proper Chebyshev polynomial filters
- [ ] Explore coherence‑based edge features

## 📝 Code Organization
- [ ] Consolidate duplicate test fixtures
- [ ] Move common test utilities to shared module
- [ ] Standardize parameter naming across models
- [ ] Clean up config schemas (remove unused fields)
- [ ] Document all magic numbers/constants

## ⚠️ Current Issues
1. **V2/V3 Coexistence**: Both paths active making code complex (~100 lines of branching logic)
2. **Legacy Parameters**: Still accepting base_channels/encoder_depth/rescnn_blocks that are ignored
3. **Environment Variable Sprawl**: Debug flags scattered (BGB_NAN_DEBUG, BGB_EDGE_CLAMP, etc.)
4. **No Deprecation Warnings**: Legacy parameters silently accepted without warnings

## 🔧 New Technical Debt (Found During Investigation)
- [ ] Consolidate debug environment variables into config or debug mode
- [ ] Remove assert_finite calls once training is stable
- [ ] Add formal deprecation system with warnings
- [ ] Simplify detector.py after V2 removal (~400 lines → ~250 lines possible)
- [ ] Remove edge clamping workarounds after numerical stability proven
- [ ] Document all environment variables in single place
- [ ] Remove NaN safeguards once V3 proven stable in production
