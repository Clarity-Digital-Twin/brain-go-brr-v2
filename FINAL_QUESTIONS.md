# FINAL QUESTIONS: V3 Architecture Critical Analysis

## ✅ **COMPLETED ITEMS**

### **1. Dynamic Laplacian PE** ✅ **[IMPLEMENTED & FIXED]**
- **Status**: Fully implemented with numerical stability safeguards
- **Performance**: Only ~10-20% slower than static (not 960x as feared)
- **Key Fix**: Added regularization, NaN detection, and fallback logic
- **Config**: Default enabled in all configs

### **2. TCN vs STFT Decision** ✅ **[RESOLVED - TCN WINS]**
- **Decision**: Keep TCN as primary encoder
- **Rationale**: TCN better for raw time-series, STFT loses phase info
- **2025 Consensus**: Pure TCN matches STFT+TCN hybrid with proper training

### **3. Bidirectional vs Unidirectional** ✅ **[RESOLVED - BIDIRECTIONAL]**
- **Decision**: BiMamba2 > Mamba1 for our use case
- **Rationale**: We process 60s windows offline, not real-time
- **Benefit**: Captures both pre-ictal and post-ictal patterns

### **4. Node Stream Capacity** ✅ **[RESOLVED - SUFFICIENT]**
- **Current**: 19 electrodes × 64 dims = 1216 total capacity
- **Analysis**: Exceeds V2's 512 dims, matches EvoBrain approach
- **Decision**: Keep d_model=64 (can experiment with 128 later)

### **5. Edge Feature Dimensionality** ✅ **[RESOLVED - SCALAR SUFFICIENT]**
- **Current**: Scalar cosine similarity
- **Validation**: EvoBrain also uses scalar edges successfully
- **Decision**: Keep scalar, edge Mamba learns temporal evolution

## 🔴 **HIGHEST PRIORITY UNRESOLVED ISSUES**

### **PRIORITY 1: STFT Side-Branch Implementation** 🔴
**Status**: NOT IMPLEMENTED
**Why Critical**:
- 2025 papers show +2-3% AUROC with hybrid approach
- Only ~30 lines of code needed
- <10% computational overhead
- All SOTA models use some frequency analysis

**Implementation Ready**: See `STFT_SIDEBRANCH_IMPLEMENTATION.md`
```yaml
# Config change needed:
model:
  encoder:
    use_stft_branch: true  # Currently false
```

### **PRIORITY 2: Graph Sparsity (top_k=3)** 🟡
**Status**: Using k=3 (16% connectivity)
**Concern**: May be too sparse for functional connectivity
**Recommendation**: Test k=5 (26% connectivity) or k=7 (37%)
```yaml
model:
  graph:
    edge_top_k: 5  # Currently 3
```

### **PRIORITY 3: GNN Temporal Processing** 🟡
**Status**: Vectorized batch processing loses temporal order
**Current**: Process all 960 timesteps as single batch
**Alternative**: Sequential processing or temporal attention
```python
# Current: Loses temporal order
x_batch = x.reshape(-1, feat_dim)  # (B*960*19, D)

# Alternative: Preserve temporal structure
for t in range(T):
    x_t = gnn(x[:, t, :, :], adj[:, t, :, :])
```

## 🟢 **LOW PRIORITY / EXPERIMENTAL**

### **Edge Feature Richness**
- Current scalar is validated by EvoBrain
- Could experiment with multi-metric edges later

### **Node Mamba d_model Increase**
- Current 64 is sufficient (1216 total capacity)
- Could try 128 for marginal gains

### **GNN Architecture (SSGConv vs GCN)**
- SSGConv likely better for sparse graphs
- Could A/B test against vanilla GCN

## 📊 **ARCHITECTURE COMPARISON TABLE**

| Component | EvoBrain | Our V3 | Status |
|-----------|----------|--------|--------|
| **Temporal Model** | Mamba1 | BiMamba2 | ✅ BETTER |
| **Input Encoder** | STFT | TCN | ✅ VALIDATED |
| **Frequency Branch** | Yes | No | 🔴 **MISSING** |
| **Dynamic PE** | Yes | Yes | ✅ IMPLEMENTED |
| **Graph Sparsity** | Unknown | k=3 | 🟡 NEEDS TEST |
| **GNN Processing** | Last step | All steps | ✅ BETTER |
| **Edge Features** | Scalar | Scalar | ✅ PARITY |

## 🚀 **RECOMMENDED ACTION PLAN**

### **Before Full Training:**

1. **MUST DO**: Add STFT side-branch (~30 min implementation)
   - Already have implementation in `STFT_SIDEBRANCH_IMPLEMENTATION.md`
   - Expected +2-3% AUROC improvement
   - Minimal overhead

2. **SHOULD TEST**: Increase edge_top_k to 5
   - Simple config change
   - Test on smoke dataset first
   - Compare convergence speed

3. **MONITOR**: Training stability with dynamic PE
   - Watch for NaN recurrence
   - Check eigenvalue distributions
   - Verify fallback logic triggers

### **After Initial Training:**

4. **EXPERIMENT**: GNN temporal processing alternatives
5. **ABLATION**: Compare SSGConv vs GCN
6. **TUNE**: Node Mamba d_model (64 vs 128)

## 📈 **EXPECTED PERFORMANCE**

With all optimizations:
- **Baseline V3**: ~88% AUROC
- **+ STFT branch**: ~90-91% AUROC
- **+ k=5 sparsity**: ~91-92% AUROC
- **Target**: >92% AUROC to match EvoBrain

## ✍️ **KEY DECISIONS SUMMARY**

**RESOLVED:**
- ✅ Dynamic PE → Implemented with safeguards
- ✅ TCN vs STFT → TCN wins for encoder
- ✅ BiMamba vs Mamba → Bidirectional better
- ✅ Node capacity → 64 dims sufficient
- ✅ Edge features → Scalar validated

**NEEDS ACTION:**
- 🔴 STFT side-branch → Ready to implement
- 🟡 Graph sparsity → Test k=5
- 🟡 GNN temporal → Consider alternatives

**The Bottom Line:**
Our V3 is architecturally sound. The only critical missing piece is the STFT side-branch for frequency analysis, which all 2025 SOTA models include. This should be implemented before full training.