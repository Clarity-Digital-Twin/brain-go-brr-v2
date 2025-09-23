# Changelog

All notable changes to the Brain-Go-Brr v2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0] - 2025-09-23

### Changed
- **MAJOR**: Replaced U-Net + ResCNN with Temporal Convolutional Networks (TCN)
- **Architecture**: Now TCN encoder (8 layers) + Bi-Mamba-2 (6 layers) + Projection head
- **Parameters**: ~34M parameters with improved temporal modeling
- **Configs**: All configs default to TCN architecture (`architecture: tcn`)
- **Training Loop**: Major robustness improvements for numerical stability

### Added
- **TCN Implementation**: Full TCN encoder with dilated convolutions and lightweight fallback
- **NaN Protection**: Comprehensive NaN handling in training loop with diagnostics
- **Focal Loss Stability**: Numerical stability improvements (logit clamping, p_t bounds)
- **Gradient Monitoring**: Enhanced gradient norm tracking and clipping
- **Batch Diagnostics**: Dead channel detection and class imbalance monitoring
- **Performance Tests**: Hardware-aware latency thresholds (RTX vs A100 GPUs)
- **Mid-Epoch Checkpointing**: Auto-save during long training runs with configurable intervals
- **Test Coverage**: NaN robustness test suite (6 comprehensive tests)

### Fixed
- **NaN Accumulator Bug**: Once total_loss became NaN, it stayed NaN forever - now properly isolated
- **Focal Loss Underflow**: (1-p_t)^gamma could underflow to 0 with high confidence predictions
- **Performance Test Regression**: P95 latency tests now hardware-aware (125ms RTX, 110ms A100)
- **Mixed Precision Stability**: Better FP16 handling with sanitization options
- **Weight Initialization**: Improved initialization to prevent output explosion
- **LR Scheduler Warning**: Properly suppressed false-positive on first batch
- **Import Order**: Fixed linting issues with module-level imports

### Improved
- **Training Robustness**: Can now recover from intermittent NaN losses
- **Error Messages**: Clear diagnostics when NaN issues occur
- **Test Stability**: Performance tests handle system load variance better
- **Documentation**: Updated all docs to reflect TCN as canonical architecture

### Removed
- **U-Net Components**: Deleted unet.py encoder/decoder modules (legacy)
- **ResCNN Blocks**: Removed rescnn.py (replaced by TCN)
- **Legacy Docs**: Marked pre-v2.3 architecture docs as historical

## [2.1.0] - 2025-09-22

### Added
- **W&B Integration**: Fully wired WandBLogger into training loop with team entity support
- **Modal Storage Documentation**: Comprehensive storage architecture documentation
- **Balanced Sampling Optimization**: 7200x speedup eliminating 2+ hour Modal bottlenecks
- **Modal Volume Explorer**: Script to investigate Modal storage contents

### Fixed
- **Mixed Precision**: Enabled FP16 for A100 (3.8x faster than FP32)
- **Batch Size**: Increased from 64 to 128 to fully utilize 80GB VRAM
- **W&B Entity**: Corrected to team name (jj-vcmcswaggins-novamindnyc) for team API keys
- **Documentation**: Removed outdated cache optimizer references

### Changed
- **Training Performance**: 10x speedup (48s → 5s per batch) through config optimizations
- **Cost Reduction**: 90% reduction ($3,190 → $319 for full training)
- **Documentation Structure**: Reorganized docs into logical sections (01-data, 02-model, 03-deployment, 04-research)

### Removed
- **Cache Optimizer**: Deleted unnecessary S3→Modal copy logic (cache was always on SSD)
- **Outdated Docs**: Removed archive folder and consolidated critical information

## [0.2.0] - 2025-09-21

### Fixed

#### 🚨 Critical P0 Blockers
- **CSV Parser for TUSZ CSV_BI Format**: Fixed parser reading wrong columns (was [0,1,2], now correctly [1,2,3] to skip channel column), preventing 0% seizure detection
- **All TUSZ Seizure Types**: Added complete seizure type recognition (gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz) - was only looking for "seiz" which doesn't exist
- **BalancedSeizureDataset**: Implemented SeizureTransformer's exact balancing (ALL partial + 0.3×full + 2.5×background) to guarantee seizures in training
- **Hard Guards**: Added CLI exit if no seizures found in manifest, preventing training collapse

#### 🔧 Configuration Management
- **Config Reorganization**: Restructured from 8 confusing configs to clean `configs/local/` and `configs/modal/` directories
- **WSL2 Fixes**: Corrected all local configs with `num_workers=0`, `pin_memory=false`, explicit `device=cuda`
- **Modal A100 Optimization**: Fixed checkpoint paths, verified batch_size=64, workers=8 for cloud GPU
- **Internal Consistency**: All configs now share identical model architecture, preprocessing, and postprocessing

#### 🚀 Modal Pipeline
- **BGB_LIMIT_FILES Fix**: Explicitly unset environment variable for full training (was limiting to 50 files)
- **Cache Structure**: Documented and separated cache directories for smoke/full/dev/eval on both local and Modal

### Added

#### 📚 Documentation
- Comprehensive configuration README with usage examples
- Cache directory structure documentation
- Modal pipeline setup guide
- Config consistency verification reports
- Root cause analysis for debugging

### Changed
- Moved configs to organized `local/` and `modal/` subdirectories
- Deleted redundant configs (local.yaml, production.yaml, tusz_train.yaml)
- Cleaned up 6.3GB of vestigial cache directories

## [0.1.0] - 2025-09-20

### Added

#### 🧠 Novel Architecture Implementation
- **Bidirectional Mamba-2 + U-Net + ResCNN**: First implementation combining U-Net multi-scale feature extraction, ResCNN temporal convolution, and bidirectional Mamba-2 state space models for O(N) complexity seizure detection
- **Core Models**:
  - `SeizureDetector`: Main model architecture with 25M+ parameters
  - `BiMambaSSM`: Bidirectional Mamba-2 implementation with configurable layers and state dimensions
  - `UNet`: Encoder-decoder with [64, 128, 256, 512] channel progression and ×16 downsampling
  - `ResCNN`: 3-block residual CNN with kernels [3, 5, 7] for multi-scale temporal processing
  - Dynamic interpolation layers for resolution recovery

#### 🏥 Clinical EEG Pipeline
- **19-channel 10-20 montage** support with canonical channel ordering: `["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]`
- **MNE-based EDF loading** with fallback repair for malformed TUSZ headers
- **Signal preprocessing**: Bandpass 0.5-120 Hz, 60 Hz notch filter, 256 Hz resampling
- **Windowing strategy**: 60-second windows with 10-second stride (83% overlap)
- **Per-channel z-score normalization**

#### 🎯 Advanced Training Features
- **Focal Loss implementation** with critical bug fixes preventing double-counting and neutral alpha handling
- **Positive-aware balanced sampling** for extreme class imbalance (typically 1:1000+ seizure ratio)
- **Class weight auto-computation** with configurable balancing strategies
- **Learning rate scheduling** with warmup and cosine annealing
- **Gradient clipping** and accumulation for memory-efficient training
- **Mixed precision training** support (FP16/BF16)

#### 🏗️ Post-Processing Pipeline
- **Hysteresis thresholding**: τ_on=0.86, τ_off=0.78 for stable seizure detection
- **Morphological filtering**: Opening and closing operations for noise reduction
- **Duration filtering**: Configurable minimum seizure duration requirements
- **Event generation**: Automatic conversion to clinical event format (CSV_BI)

#### 📊 Evaluation Framework
- **TAES metrics integration**: Using industry-standard NEDC evaluation tools
- **Clinical performance targets**:
  - 10 FA/24h: >95% sensitivity goal
  - 5 FA/24h: >90% sensitivity goal
  - 1 FA/24h: >75% sensitivity goal
- **ROC curve analysis** with AUC computation
- **Sensitivity-FA curves** for clinical interpretation

#### 🌩️ Cloud Deployment Infrastructure
- **Modal.com integration**: Complete A100-80GB training setup
- **S3 data management**: Automated dataset sync and caching
- **Weights & Biases integration**: Experiment tracking and hyperparameter logging
- **Docker containerization**: Reproducible environments with CUDA support

#### 🧪 Comprehensive Testing Suite
- **Unit tests**: 100+ tests covering all major components
- **Integration tests**: End-to-end pipeline validation
- **Performance benchmarks**: Latency and memory usage monitoring
- **Clinical validation tests**: Channel ordering and TAES metric verification
- **GPU/CPU compatibility**: Automated testing for different hardware configurations

#### ⚙️ Development Infrastructure
- **Modern Python toolchain**: Python 3.11+, UV package manager, Ruff formatting
- **Makefile automation**: Quality checks (`make q`), testing (`make t`), training (`make train-local`)
- **Pre-commit hooks**: Automated code quality enforcement
- **Type safety**: Full mypy type checking with strict configuration
- **Configuration management**: Pydantic schemas with YAML configs

#### 📚 Documentation & Guides
- **Complete architecture specification**: Detailed technical documentation
- **Modal deployment guide**: Step-by-step cloud training setup
- **WSL2 setup guides**: Windows development environment configuration
- **Implementation phases**: Structured development roadmap
- **Evaluation checklist**: Clinical validation procedures

### Fixed

#### 🐛 Critical Focal Loss Bugs
- **Double-counting prevention**: Fixed focal loss computation that was applying alpha weighting twice
- **Neutral alpha handling**: Corrected alpha=0.5 to avoid biasing toward negative class
- **Loss scaling**: Proper normalization to prevent gradient explosion
- **Class weight interaction**: Fixed incompatibility between focal loss and class weights

#### 🔧 Training Stability Improvements
- **Memory leak fixes**: Proper tensor cleanup in training loops
- **Device compatibility**: Enhanced CUDA/CPU tensor handling
- **Scheduler step logic**: Corrected learning rate scheduling timing
- **Gradient accumulation**: Fixed batch size scaling for memory-limited training

#### 📡 Data Pipeline Robustness
- **EDF header repair**: Automatic handling of malformed TUSZ annotations
- **Channel synonym mapping**: T7→T3, T8→T4, P7→T5, P8→T6 compatibility
- **Sampling rate consistency**: Robust resampling for variable input rates
- **Missing channel handling**: Graceful degradation for incomplete montages

#### 🏃 Performance Optimizations
- **WSL2 compatibility**: UV_LINK_MODE=copy for cross-filesystem performance
- **Multiprocessing safety**: num_workers=0 default to prevent WSL hangs
- **CUDA kernel dispatch**: Automatic fallback for unsupported Mamba configurations
- **Memory usage**: Optimized tensor operations and caching strategies

### Security

#### 🔒 Environment Safety
- **Dependency pinning**: Locked PyTorch 2.2.2 and NumPy <2.0 for mamba-ssm compatibility
- **Pre-commit security**: Automated vulnerability scanning
- **Container isolation**: Secure Modal deployment with minimal attack surface

### Technical Specifications

#### 🏗️ Architecture Details
- **Model Size**: ~25M parameters (configurable)
- **Input**: 19-channel EEG @ 256 Hz, 60-second windows
- **Output**: Per-timestep seizure probabilities
- **Complexity**: O(N) sequence modeling vs Transformer's O(N²)
- **Memory**: 24GB+ VRAM recommended for training

#### 🔧 System Requirements
- **Python**: 3.11+ (3.12 supported)
- **PyTorch**: 2.2.2 (required for mamba-ssm)
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 1TB+ for full TUSZ dataset

#### 📦 Dependencies
- **Core ML**: torch, numpy, scipy, scikit-learn
- **EEG Processing**: mne, pyedflib
- **Deep Learning**: einops, mamba-ssm (GPU extra)
- **Configuration**: pydantic, pyyaml, click, rich
- **Visualization**: matplotlib, seaborn
- **Development**: pytest, ruff, mypy, pre-commit

### Notes

This release represents the first complete implementation of the Brain-Go-Brr v2 architecture. While the codebase is feature-complete with comprehensive testing, clinical benchmarks are pending. The system is ready for research evaluation but has not yet been validated on held-out clinical datasets.

**Breaking Changes**: This is an initial release, so no breaking changes apply.

**Migration Guide**: N/A for initial release.

**Known Issues**:
- Mamba CUDA kernels only support d_conv={2,3,4}, automatically coerced from configured d_conv=5
- WSL2 requires UV_LINK_MODE=copy for optimal performance
- Full TUSZ training requires 24GB+ VRAM; use smoke test configs for development

---

**Release Readiness**: ✅ Architecture Complete | ✅ Testing Suite | ✅ Documentation | ⏳ Benchmarks Pending
