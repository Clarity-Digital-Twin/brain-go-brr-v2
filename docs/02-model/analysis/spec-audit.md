# CANONICAL SPEC AUDIT RESULTS (Legacy snapshot)

Note: This file reflects an older audit (pre‑v2.3). The active and verified
specs are documented in `../architecture/current-state.md` and
`../architecture/v3_tcn_evobrain_hybrid.md`. The sections below are kept for
historical context and ablation reference. Key discrepancies are corrected inline.

### Summary of Findings

#### ✅ FULLY IMPLEMENTED Components (current)
1. **Data Pipeline**
   - EDF loading with TUSZ header repair
   - Channel mapping and 10-20 montage ordering
   - Resampling to 256 Hz
   - Bandpass (0.5-120 Hz) and notch (60 Hz) filtering
   - Per-channel z-score normalization
   - Window extraction (60s windows, 10s stride)
   - Microvolts conversion in io.py

2. **Model Architecture**
   - TCN Encoder (×16 down): `src/brain_brr/models/tcn.py`
   - Bidirectional Mamba-2 (6 layers, d_model=512, d_state=16)
   - Projection + Upsample head and Detection head
   - Optional Dynamic GNN + LPE (PyG) after Bi‑Mamba
   - **Actual parameters (v2): ~13.4M**

3. **Training Components**
   - PyTorch Dataset implementation
   - Training loop with AMP support
   - Loss functions and optimization
   - Validation metrics

4. **Post-processing**
   - Hysteresis thresholding
   - Morphological operations
   - Window stitching capabilities

5. **Infrastructure**
   - Pydantic config schemas
   - Constants properly defined
   - File I/O and preprocessing

#### ⚠️ DISCREPANCIES Found:
1. **Parameter Count**: Documentation says ~25M, actual is ~13.4M
2. **ConvBlock Activation**: Uses ReLU (not ELU as mentioned in some docs)
3. **Mamba d_conv**: Documented as 5, coerced to 4 for CUDA kernels

#### ❌ OUTDATED (now implemented):
1. Dataset caching to NPZ files → Implemented (`EEGWindowDataset`, cache_utils)
2. Balanced sampling strategy → Implemented (`BalancedSeizureDataset`, manifest)
3. TAES metrics → Implemented (`src/brain_brr/eval/metrics.py`)
4. CLI commands → Implemented (`python -m src` with `build-cache`, `scan-cache`, `train`, `evaluate`, `validate`)
5. count_parameters/get_layer_info → N/A; use standard PyTorch utilities and logging

### Files Verified:
- ✓ src/brain_brr/models/detector.py
- ✓ src/brain_brr/models/unet.py
- ✓ src/brain_brr/models/rescnn.py
- ✓ src/brain_brr/models/mamba.py
- ✓ src/brain_brr/models/layers.py
- ✓ src/brain_brr/data/io.py
- ✓ src/brain_brr/data/preprocess.py
- ✓ src/brain_brr/data/windows.py
- ✓ src/brain_brr/data/datasets.py
- ✓ src/brain_brr/train/loop.py
- ✓ src/brain_brr/post/postprocess.py
- ✓ src/brain_brr/eval/metrics.py
- ✓ src/brain_brr/config/schemas.py
- ✓ src/brain_brr/constants.py

### Critical Implementation Notes:
1. **Mamba Fallback**: Currently using Conv1d fallback (mamba-ssm not installed)
2. **Channel Order**: Correctly maintained as Fp1→F3→...→O2
3. **Window Parameters**: Correct (60s windows, 10s stride, 256 Hz)
4. **Microvolts Conversion**: Properly handled in io.py (×1e6)
5. **Skip Connections**: Properly saved after block, before downsample

### Recommendations:
1. Update documentation to reflect actual 13.4M parameter count
2. Document ReLU usage in ConvBlock (not ELU)
3. Add missing dataset caching implementation
4. Implement balanced sampling for training
5. Complete TAES metric implementation
6. Add CLI interface
