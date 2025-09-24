# BiMamba Architecture Comparison for TUSZ Seizure Detection (Historical)

Note: This comparison was written before the TCN front‑end replacement. It discusses a
U‑Net + ResCNN + Bi‑Mamba design. The current runtime path uses TCN → Bi‑Mamba →
Projection+Upsample → Detection. See `../architecture/current-state.md`.

## Executive Summary

Our analysis of existing BiMamba/Mamba architectures reveals a **critical gap**: No model has successfully combined bidirectional state-space models with multi-scale feature extraction for TUSZ seizure detection. We are the first to implement Bi-Mamba-2 + U-Net + ResCNN specifically for this challenging clinical dataset.

## Literature Review: State-Space Models for EEG

### 1. EEGMamba (2024)
- **Architecture**: Bidirectional Mamba + Spatio-Temporal-Adaptive + MoE
- **Datasets**: CHB-MIT, Siena, DEAP, SEED, BCI-IV-2a
- **Key Results**:
  - CHB-MIT: ~97% accuracy (pediatric seizures)
  - Siena: >98% accuracy
- **Limitations for TUSZ**:
  - Never tested on adult clinical seizures
  - No multi-scale temporal modeling
  - MoE adds complexity without addressing seizure dynamics

### 2. FEMBA (2025)
- **Architecture**: Bidirectional Mamba + SSL pre-training
- **Datasets**: TUAB, TUAR, TUSL (Temple University, but NOT seizures)
- **Key Results**:
  - TUAB: 81.82% balanced accuracy (abnormal detection)
  - TUAR: 0.949 AUROC (artifact detection)
- **Limitations for TUSZ**:
  - Focused on abnormal/artifact detection, not seizures
  - Missing hierarchical feature extraction
  - No seizure-specific temporal modeling

### 3. Mentality (2024)
- **Architecture**: Basic Mamba + masked reconstruction
- **Datasets**: TUSZ v2.0.1 (finally!)
- **Key Results**:
  - TUSZ: 0.72 AUROC (vs 0.64 from scratch)
- **Limitations**:
  - Poor performance indicates architectural gaps
  - Unidirectional only (misses pre-ictal patterns)
  - No multi-scale processing

## Why Pure Mamba Fails for Seizure Detection

### Seizure Temporal Dynamics (Not Captured by Pure SSM)

```
Time Scale       | Phenomenon           | Required Architecture
-----------------|---------------------|----------------------
1-100 ms         | Spike/sharp waves   | High-res conv (ResCNN)
100ms-1s         | Rhythmic patterns   | Local features (U-Net encoder)
1-10s            | Pattern evolution   | Hierarchical (U-Net stages)
10-600s          | Full seizure event  | Global context (Bi-Mamba)
±5 min           | Pre/post-ictal      | Bidirectional (forward+backward)
```

### Critical Missing Components in Existing Work

1. **Multi-Scale Feature Hierarchy**
   - Mamba assumes single-scale dynamics
   - Seizures require 4-5 temporal scales simultaneously

2. **Nonlinear Pattern Detection**
   - SSMs model linear dynamics
   - Seizures have chaotic, nonlinear evolution

3. **Spatial-Channel Relationships**
   - Most implementations treat channels independently
   - Seizure spread requires cross-channel modeling

## Our Architecture: Filling the Gaps

### Bi-Mamba-2 + U-Net + ResCNN Synergy

```python
Component         | Purpose                    | Addresses Gap
------------------|----------------------------|---------------
U-Net Encoder     | Multi-scale extraction     | Hierarchical features
 └─ [64→512]      | 4 resolution levels        | Captures all time scales
ResCNN            | Local pattern refinement   | Nonlinear dynamics
 └─ [3,5,7]       | Multi-kernel sizes         | Frequency-specific patterns
Bi-Mamba-2        | Efficient global context   | O(N) long sequences
 └─ Bidirectional | Forward + backward         | Pre/post-ictal patterns
U-Net Decoder     | Multi-scale reconstruction | Preserves fine details
 └─ Skip connects | Feature preservation       | Maintains spike info
```

### Why This Combination Works

1. **U-Net handles what Mamba can't**: Multi-scale hierarchical features
2. **ResCNN captures what SSMs miss**: Nonlinear local patterns
3. **Bi-Mamba provides what CNNs lack**: Efficient global temporal context
4. **Skip connections preserve information**: Critical for precise boundaries

## Performance Targets

### Clinical Requirements (TAES Metrics)
- 10 FA/24h: >95% sensitivity (current SOTA: ~90%)
- 5 FA/24h: >90% sensitivity (current SOTA: ~85%)
- 1 FA/24h: >75% sensitivity (current SOTA: ~70%)

### Expected Improvements Over Existing Work
- vs Mentality (0.72): +15-20% AUROC from architecture alone
- vs FEMBA: Seizure-specific design should outperform generic abnormal detection
- vs EEGMamba: TUSZ-optimized should beat pediatric-trained models

## Implementation Strategy

### Phase 1: Baseline Validation
1. Implement pure Bi-Mamba baseline
2. Verify Mentality's 0.72 AUROC is reproducible
3. Identify specific failure modes

### Phase 2: Multi-Scale Integration
1. Add U-Net wrapper around Bi-Mamba
2. Measure improvement from hierarchical features
3. Analyze which scales contribute most

### Phase 3: Local Pattern Enhancement
1. Insert ResCNN between U-Net encoder and Mamba
2. Test different kernel combinations
3. Optimize for spike detection

### Phase 4: TUSZ-Specific Tuning
1. Weighted loss for seizure/non-seizure imbalance
2. Augmentation for clinical artifacts
3. Progressive training (CHB-MIT → TUSZ transfer)

## Key Differentiators

| Feature | EEGMamba | FEMBA | Mentality | **Ours** |
|---------|----------|-------|-----------|----------|
| Bidirectional | ✓ | ✓ | ✗ | ✓ |
| Multi-scale | ✗ | ✗ | ✗ | ✓ |
| Nonlinear patterns | ✗ | ✗ | ✗ | ✓ |
| TUSZ-tested | ✗ | ✗ | ✓ | ✓ |
| Hierarchical | ✗ | ✗ | ✗ | ✓ |
| Clinical focus | ✗ | Partial | ✓ | ✓ |

## Conclusion

We are positioned to achieve breakthrough performance on TUSZ by addressing fundamental architectural limitations of pure state-space models. The combination of U-Net's multi-scale extraction, ResCNN's pattern detection, and Bi-Mamba's efficient sequence modeling creates a synergy specifically suited for the complex temporal dynamics of clinical seizures.

**This is not just another Mamba variant - it's the first architecture designed from the ground up for TUSZ's unique challenges.**
