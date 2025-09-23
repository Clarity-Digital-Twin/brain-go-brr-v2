# 🧠 Architecture Evolution & Strategy
## From Transformers to O(N) Mamba to Dynamic Graphs

---

## 📚 The Journey: How We Got Here

### 1️⃣ **SeizureTransformer** (2024)
- **Architecture**: U-Net + ResCNN + Transformer
- **Strengths**:
  - SOTA on TUSZ (0.876 AUROC)
  - Time-step level detection (no window post-processing)
  - Global attention captures long-range dependencies
- **Weaknesses**:
  - O(N²) complexity kills us on long sequences
  - Memory explodes with 60s windows
  - Slow inference (77s per hour of EEG)

### 2️⃣ **EEG-BiMamba** (2024)
- **Architecture**: ST-Adaptive + Bidirectional Mamba + MoE
- **Key Innovation**: Replace Transformer with Mamba for O(N) complexity
- **Strengths**:
  - Linear complexity for long sequences
  - Bidirectional for non-causal EEG analysis
  - Multi-task learning with MoE
- **Our Take**: Good direction, but ST-Adaptive is overcomplicated

### 3️⃣ **Our v2.3-2.5** (Current Stable)
- **Architecture**: TCN + Bi-Mamba
- **Why TCN over U-Net/ResCNN?**:
  - Multi-scale temporal features (better than fixed U-Net scales)
  - Efficient downsampling (16x reduction)
  - Proven for time-series (not borrowed from image segmentation)
- **Status**: Working, 90% sensitivity @ 10 FA/24h

### 4️⃣ **EvoBrain** (2025 NeurIPS)
- **Architecture**: Time-then-Graph with Mamba + GNN + Laplacian PE
- **Revolutionary Claims**:
  - +23% AUROC over dynamic GNN baseline
  - Explicit dynamic graphs (adjacency changes over time)
  - Two-stream Mamba: one for nodes, one for edges
- **Key Insight**: Brain connectivity EVOLVES during seizures

### 5️⃣ **Our v2.6** (In Development)
- **Architecture**: TCN + Bi-Mamba + GNN + LPE
- **Status**: Transitional - we have GNN+LPE but using heuristic graphs, not learned

---

## 🤔 The Confusion: What EvoBrain Actually Does

### EvoBrain's Two-Stream Architecture:
```
1. NODE STREAM (temporal dynamics):
   EEG channels → Mamba → node embeddings over time

2. EDGE STREAM (connectivity dynamics):
   Channel pairs → Mamba → edge weights over time

3. FUSION:
   Learned adjacency from edges → GNN with node features → Detection
```

### Critical EvoBrain Details:
- **Uni vs Bi Mamba**: They use UNIDIRECTIONAL for causality in online detection
- **Node Mamba**: Processes each channel's temporal evolution
- **Edge Mamba**: Processes connectivity evolution between channel pairs
- **Time-then-Graph**: First model temporal, THEN apply spatial GNN
- **Laplacian PE**: Adds k=16 eigenvectors to node features for position encoding

### What We Currently Have:
- ✅ Bi-Mamba for temporal (better than uni for offline)
- ✅ GNN with Laplacian PE (PyG implementation)
- ❌ Edge stream Mamba (using heuristic cosine similarity instead)
- ❌ Learned dynamic adjacency (using static graph builder)

---

## 🎯 The Ideal Architecture (Theoretical Best)

### For OFFLINE Detection (Full Recording Available):
```
TCN → Bi-Mamba (nodes) → Edge Mamba → Learned Adjacency → GNN+LPE → Detection
     ↑                                                      ↑
     Better than Transformer O(N²)                         Spatial reasoning
```

**Why this is optimal:**
- TCN: Multi-scale temporal feature extraction
- Bi-Mamba: Bidirectional context (past AND future)
- Edge Mamba: Learn connectivity dynamics
- GNN+LPE: Capture spatial relationships with position awareness

### For ONLINE Detection (Real-time Streaming):
```
TCN → Uni-Mamba (causal) → Edge Stream → Dynamic Graph → GNN+LPE → Detection
```

**Key differences:**
- Unidirectional Mamba (can't look into future)
- Causal TCN padding
- Sliding window with minimal lookahead

---

## 🚀 Implementation Strategy

### Phase 1: Current v2.6 (Immediate)
```
TCN + Bi-Mamba + Heuristic Graph + GNN+LPE
```
- Ship what works NOW
- Heuristic graph is fine for baseline
- Focus on stability and benchmarking

### Phase 2: Full EvoBrain-style (v3.0)
```
TCN + Bi-Mamba(nodes) + Bi-Mamba(edges) + Learned Adjacency + GNN+LPE
```
- Implement edge stream
- Replace heuristic with learned adjacency
- Requires significant engineering

### Phase 3: Online vs Offline Variants
```
Offline: Bidirectional everything
Online: Unidirectional with causality
```

---

## 💭 Key Architectural Decisions

### 1. **Why Keep TCN?**
- EvoBrain doesn't have good multi-scale extraction
- TCN > U-Net for time-series
- Already proven in our pipeline

### 2. **Why Bi-Mamba for Offline?**
- EEG is non-causal for retrospective analysis
- Future context helps identify seizure onset
- EvoBrain uses uni for online only

### 3. **Why Time-then-Graph?**
- EvoBrain proves it's superior (Theorem 2)
- Graph-then-time loses temporal context
- Time-and-graph is 17x slower

### 4. **Do We Need Edge Stream?**
- **Yes for SOTA**: EvoBrain's main innovation
- **No for v2.6**: Heuristic graphs work adequately
- **Future work**: Implement for v3.0

---

## 📊 Benchmarking Targets

### TUSZ Online (Real-time)
- Target: >90% sensitivity @ 5 FA/24h
- Current SOTA: ~85%
- Architecture: Uni-Mamba + Causal

### TUSZ Offline (Full Recording)
- Target: >95% sensitivity @ 10 FA/24h
- Current SOTA: ~90%
- Architecture: Bi-Mamba + Non-causal

### CHB-MIT (Pediatric)
- Different distribution, good generalization test
- Target: >93% sensitivity

---

## 🔬 Experimental Questions

1. **Heuristic vs Learned Adjacency**
   - How much does edge stream actually help?
   - Is cosine similarity good enough?

2. **TCN vs Transformer Encoder**
   - EvoBrain uses Transformer AFTER Mamba
   - Do we need global attention if we have Mamba?

3. **Laplacian PE Dimension**
   - EvoBrain uses k=16
   - Should we tune for 19 channels?

4. **Graph Sparsity**
   - Top-k=3 (EvoBrain default)
   - More connections = more computation

---

## 🎲 The Bottom Line

### What We're Building (v2.6):
**TCN + Bi-Mamba + GNN+LPE** with heuristic graphs
- Pragmatic compromise
- Ships today
- Good enough for benchmarking

### What's Theoretically Best:
**TCN + Dual-Stream Mamba + Learned Dynamic Graphs + GNN+LPE**
- Full EvoBrain architecture
- Proven 23% improvement
- Significant engineering effort

### The Path Forward:
1. Ship v2.6 with what works
2. Benchmark thoroughly
3. Implement edge stream for v3.0
4. Separate online/offline variants

---

## 🤯 TL;DR

We evolved from:
- **Transformers** (too slow, O(N²))
- **→ Mamba** (fast, O(N))
- **→ TCN+Mamba** (better features)
- **→ +GNN** (spatial reasoning)
- **→ +Edge Stream** (dynamic connectivity) ← *future work*

Current confusion is because we're in transition:
- We HAVE: TCN + Bi-Mamba + GNN + LPE
- We USE: Heuristic graphs (cosine similarity)
- We WANT: Learned graphs via edge Mamba stream
- We NEED: To ship v2.6 and iterate

The "ideal" depends on online vs offline, but full EvoBrain-style with TCN frontend and dual Mamba streams is theoretically best.