# Technical Debt & Cleanup Tasks

## 🧹 PRIORITY 1: GNN Implementation Cleanup (Pure GNN + LPE)
- [ ] Remove heuristic adjacency builder path (cosine/correlation)
- [ ] Implement edge stream (extractor + Bi‑Mamba + Linear+Softplus)
- [ ] Assemble learned adjacency with top‑k + threshold + symmetry + identity fallback
- [ ] Make PyG + Laplacian PE the canonical backend (keep pure‑torch only for tests)
- [ ] Update detector to consume learned adjacency (no builder)
- [ ] Update configs: add `edge_*` fields; deprecate `similarity`, `top_k`, `temperature`, `threshold`

## 🧹 PRIORITY 2: Legacy Code Removal
- [ ] Remove UNet legacy imports/references
- [ ] Remove ResCNN legacy imports/references
- [ ] Clean up detector.py to remove legacy parameter handling
- [ ] Update all tests to use TCN parameters only
- [ ] Remove `base_channels`, `encoder_depth`, `rescnn_blocks` parameters

## 🧹 PRIORITY 3: Test Suite Stability
- [ ] Fix intermittent worker crashes in parallel tests
- [ ] Add proper resource cleanup in test fixtures
- [ ] Investigate memory leaks in training tests
- [ ] Ensure all tests properly marked with `@pytest.mark.serial` when needed
- [ ] Add timeout decorators for long-running tests

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

## ⚠️ Known Issues
1. **Fake SSGConv**: Pure PyTorch implementation is 1-hop only, not real K-hop
2. **Missing Edge Mamba**: EvoBrain uses Mamba for both node AND edge streams (addressed by Priority 1)
3. **Test Workers Crashing**: Resource exhaustion in parallel test execution
4. **Legacy Parameters**: Still accepting UNet/ResCNN params that are ignored

## 🔧 Refactoring Opportunities
- Unify graph construction logic
- Extract common patterns in model initialization
- Simplify configuration hierarchy
- Consolidate loss function implementations
- Streamline data pipeline
