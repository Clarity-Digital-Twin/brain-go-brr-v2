# Documentation Index

## 📁 Documentation Structure

### 01-data/ - Data Pipeline & Processing
- **pipeline/** - Core data pipeline architecture
  - [`architecture.md`](01-data/pipeline/architecture.md) - Complete EDF → Training pipeline with optimizations
  - [`flow-diagram.md`](01-data/pipeline/flow-diagram.md) - Mermaid visualization of data flow
  - [`cache-rebuild.md`](01-data/pipeline/cache-rebuild.md) - Quick cache rebuild guide
- **tusz/** - TUSZ dataset specifics
  - [`overview.md`](01-data/tusz/overview.md) - Dataset structure
  - [`channels.md`](01-data/tusz/channels.md) - 10-20 montage mapping
  - [`csv-parser.md`](01-data/tusz/csv-parser.md) - CSV_BI label parsing
  - [`edf-repair.md`](01-data/tusz/edf-repair.md) - Header repair
  - [`data-flow.md`](01-data/tusz/data-flow.md) - Processing pipeline
  - [`cache-sampling.md`](01-data/tusz/cache-sampling.md) - Balanced sampling
  - [`preflight.md`](01-data/tusz/preflight.md) - Pre-training checks
- **issues/** - Historical issues and resolutions
  - [`critical-resolved.md`](01-data/issues/critical-resolved.md) - Resolved bugs and lessons learned
  - [`data-io.md`](01-data/issues/data-io.md) - I/O implementation details

### 02-model/ - Model Architecture
- **architecture/** - Core model specifications
  - [`canonical-spec.md`](02-model/architecture/canonical-spec.md) - **SOURCE OF TRUTH**
  - [`full-model.md`](02-model/architecture/full-model.md) - Complete model architecture
  - [`pipeline-diagram.md`](02-model/architecture/pipeline-diagram.md) - Model pipeline visualization
- **components/** - Individual model components
  - [`unet.md`](02-model/components/unet.md) - U-Net encoder/decoder
  - [`rescnn.md`](02-model/components/rescnn.md) - Residual CNN blocks
  - [`mamba.md`](02-model/components/mamba.md) - Bi-Mamba-2 (O(N))
  - [`decoder.md`](02-model/components/decoder.md) - Output decoder
- **deployment/** - Deployment-specific architecture
  - [`architecture.md`](02-model/deployment/architecture.md) - CUDA/Modal deployment details
  - [`mamba-kernels.md`](02-model/deployment/mamba-kernels.md) - d_conv kernel decisions
- **analysis/** - Architecture analysis and audits
  - [`comparison.md`](02-model/analysis/comparison.md) - Architecture comparisons
  - [`stack-analysis.md`](02-model/analysis/stack-analysis.md) - Stack breakdown
  - [`audit-summary.md`](02-model/analysis/audit-summary.md) - Audit results
  - [`spec-audit.md`](02-model/analysis/spec-audit.md) - Specification audit

### 03-deployment/ - Deployment & Operations
- **modal/** - Modal.com cloud deployment
  - [`deploy.md`](03-deployment/modal/deploy.md) - Modal deployment guide
  - [`storage.md`](03-deployment/modal/storage.md) - Storage architecture (S3 mount + Modal SSD)
  - [`PERFORMANCE_OPTIMIZATION.md`](03-deployment/modal/PERFORMANCE_OPTIMIZATION.md) - A100 performance tuning
  - [`preflight.md`](03-deployment/modal/preflight.md) - Pre-deployment checks
- **local/** - Local development setup
  - [`wsl2.md`](03-deployment/local/wsl2.md) - WSL2-specific setup
  - [`setup.md`](03-deployment/local/setup.md) - General setup guide
- **operations/** - Training and evaluation
  - [`training.md`](03-deployment/operations/training.md) - Training procedures
  - [`evaluation.md`](03-deployment/operations/evaluation.md) - TAES metrics
  - [`postprocessing.md`](03-deployment/operations/postprocessing.md) - Post-processing pipeline
- [`troubleshooting.md`](03-deployment/troubleshooting.md) - Comprehensive troubleshooting guide

### 04-research/ - Research & Future Work
- **future/** - Future directions
  - [`direction.md`](04-research/future/direction.md) - Overall future plans
  - [`roadmap.md`](04-research/future/roadmap.md) - Experimental stack roadmap
  - [`gnn-tcn-stack.md`](04-research/future/gnn-tcn-stack.md) - GNN-TCN architecture exploration
- **benchmarks/** - Performance benchmarking
  - [`plans.md`](04-research/benchmarks/plans.md) - Benchmark planning
  - [`results.md`](04-research/benchmarks/results.md) - Benchmark results
- **experiments/** - Experimental features
  - [`channel-annotations.md`](04-research/experiments/channel-annotations.md) - Channel-specific analysis
  - [`streaming.md`](04-research/experiments/streaming.md) - Real-time streaming plans

## 🚀 Quick Start

1. **[Setup Guide](03-deployment/local/setup.md)** - Complete installation instructions
2. **[Architecture Overview](02-model/architecture/canonical-spec.md)** - Model specification
3. **[Training](03-deployment/operations/training.md)** - How to train models

## ⚡ Critical Information

### GPU Setup Requirements
- **CUDA Toolkit 12.1** (MUST match PyTorch)
- **mamba-ssm 2.2.2** (not 2.2.4/2.2.5 - have bugs)
- **causal-conv1d 1.4.0** (1.5+ needs PyTorch 2.4+)
- Use `make setup-gpu` to install

### Key Commands
```bash
# Setup
make setup          # Base dependencies
make setup-gpu      # GPU extensions

# Training
make train-local    # Uses .venv/bin/python
# DON'T use: uv run python -m src train

# Quality
make q              # Lint + format + type check
```

### Known Issues
- **UV can't build GPU packages** - Use `--no-build-isolation`
- **WSL2 multiprocessing** - Configs set `num_workers: 0`
- **Mamba fallback** - Ensure CUDA 12.1 installed

## 📁 Root Documents
- [README.md](../README.md) - Project overview
- [SETUP.md](../SETUP.md) - Quick setup reference
- [CLAUDE.md](../CLAUDE.md) - AI assistant guide
- [AGENTS.md](../AGENTS.md) - Agent configuration
- [CHANGELOG.md](../CHANGELOG.md) - Version history
