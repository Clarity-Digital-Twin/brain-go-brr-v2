# Literature References

**Purpose**: Comprehensive list of all papers/references cited in README.md, organized for easy verification and updating with live arXiv links.

**Status**: ✅ COMPLETE - All 8 references verified with arXiv links

---

## Referenced in README.md

### 1. EvoBrain (NeurIPS 2025)
- **Verified link**: https://arxiv.org/abs/2509.15857
- **Title**: "EvoBrain: Dynamic Multi-channel EEG Graph Modeling for Time-evolving Brain Network"
- **Authors**: Rikuto Kotoge, Zheng Chen, Tasuku Kimura, Yasuko Matsubara, Takufumi Yanagisawa, Haruhiko Kishima, Yasushi Sakurai
- **Context**: Time-then-graph paradigm, dynamic graphs, Theorem 1 & 2, two-stream Mamba + GCN with Laplacian PE
- **Local file**: `literature/markdown/EVOBRAIN.md/EVOBRAIN.md`
- **Quote in README**: "EvoBrain establishes two critical theorems"
- **Status**: ✅ VERIFIED

### 2. Mamba (Gu & Dao 2023)
- **Current link**: https://arxiv.org/abs/2312.00752
- **Context**: Selective state-space models
- **Status**: ✅ VERIFIED (arXiv link works)
- **Quote in README**: "Fast CUDA kernels, selective state propagation"

### 3. Gated DeltaNet (Yang et al., ICLR 2025)
- **Current link**: https://github.com/NVlabs/GatedDeltaNet
- **Context**: Memory erasure + delta rule
- **Local file**: `literature/markdown/GATED-DETLA/GATED-DETLA.md`
- **Status**: ✅ VERIFIED (GitHub repo is official)
- **Note**: Also has paper, but GitHub is primary reference

### 4. TCN - Temporal Convolutional Networks (Bai et al. 2018)
- **Current link**: https://arxiv.org/abs/1803.01271
- **Context**: Multi-scale temporal decomposition
- **Local file**: `literature/markdown/TCN/TCN.md`
- **Status**: ✅ VERIFIED (arXiv link works)
- **Quote in README**: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"

### 5. Focal Loss (Lin et al. 2017)
- **Current link**: https://arxiv.org/abs/1708.02002
- **Context**: Class imbalance handling
- **Local file**: `literature/markdown/FOCAL_LOSS/FOCAL_LOSS.md`
- **Status**: ✅ VERIFIED (arXiv link works)
- **Quote in README**: "Focal Loss for Dense Object Detection"

### 6. TUH EEG Seizure Corpus (TUSZ)
- **Current link**: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml
- **Context**: Dataset (504 hours, 592 patients)
- **Local file**: `literature/markdown/TUSZ-DATA/TUSZ-DATA.md`
- **Status**: ✅ VERIFIED (Temple University official site)
- **Note**: Not a paper, but dataset documentation

### 7. EEG-Mamba (2024)
- **Verified link**: https://arxiv.org/abs/2407.20254
- **Title**: "EEGMamba: Bidirectional State Space Model with Mixture of Experts for EEG Multi-task Classification"
- **Authors**: Yiyu Gui, MingZhi Chen, Yuqi Su, Guibo Luo, Yuchao Yang
- **Context**: Speed comparison reference ("128 Hz/batch (EEG-Mamba 2024) vs 8 Hz/batch for Transformers")
- **Local file**: `literature/markdown/EEG-BIMAMBA/EEG-BIMAMBA.md`
- **Status**: ✅ VERIFIED - Optional to add to README (only mentioned for speed benchmark, not foundational)

### 8. SeizureTransformer (Wu et al. 2025)
- **Verified link**: https://arxiv.org/abs/2504.00336
- **Title**: "SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings"
- **Authors**: Kerui Wu, Ziyue Zhao, Bülent Yener
- **Context**: SOTA baseline (26.89 FA/24h @ 45.63% sensitivity, EpilepsyBench #1)
- **Local file**: `literature/markdown/seizure_transformer/SeizureTransformer.md`
- **Status**: ✅ VERIFIED - READY TO ADD to README acknowledgments

---

## Additional Local Files (Not in README)

### 9. EEMG2
- **Local file**: `literature/markdown/EEMG2/`
- **Status**: Not referenced in README
- **Action**: Keep local only (not needed for README)

### 10. Picone 2021 NEDC Scoring
- **Local file**: `literature/markdown/picone-2021-NEDC-SCORING/`
- **Status**: Not referenced in README (scoring methodology)
- **Action**: Keep local only (technical reference)

### 11. Picone Model Benchmarks
- **Local file**: `literature/markdown/picone-model-benchmarks/`
- **Status**: Not referenced in README
- **Action**: Keep local only (internal benchmarking)

---

## Summary of Actions Needed

### All References Verified ✅
1. ✅ **Mamba** - https://arxiv.org/abs/2312.00752
2. ✅ **Gated DeltaNet** - https://github.com/NVlabs/GatedDeltaNet
3. ✅ **TCN** - https://arxiv.org/abs/1803.01271
4. ✅ **Focal Loss** - https://arxiv.org/abs/1708.02002
5. ✅ **TUSZ** - Temple official site (dataset)
6. ✅ **EvoBrain** - https://arxiv.org/abs/2509.15857 (UPDATED IN README)
7. ✅ **EEG-Mamba** - https://arxiv.org/abs/2407.20254 (OPTIONAL: add to README for speed benchmark citation)
8. ✅ **SeizureTransformer** - https://arxiv.org/abs/2504.00336 (ADDED TO README)

### Low Priority (local reference only)
- EEMG2 (not in README)
- Picone NEDC Scoring (not in README)
- Picone Model Benchmarks (not in README)

---

## ✅ Verification Complete!

**All steps completed** (2025-10-20):

✅ **Step 1**: User provided live arXiv links:
1. EvoBrain (NeurIPS 2025) - https://arxiv.org/abs/2509.15857
2. EEG-Mamba (2024) - https://arxiv.org/abs/2407.20254
3. SeizureTransformer (Wu et al. 2025) - https://arxiv.org/abs/2504.00336

✅ **Step 2**: Updated LITERATURE.md with verified links

✅ **Step 3**: Updated README.md:
- Replaced EvoBrain search link with direct arXiv
- Added SeizureTransformer to acknowledgments (was missing!)
- EEG-Mamba verified but optional (only speed benchmark reference)

✅ **Step 4**: All 8 references marked as ✅ VERIFIED

**Optional**: Add EEG-Mamba to README acknowledgments for speed benchmark citation (line 81: "128 Hz/batch (EEG-Mamba 2024)")

---

**Last Updated**: 2025-10-20
**Maintainer**: Keep this file in sync with README.md citations
