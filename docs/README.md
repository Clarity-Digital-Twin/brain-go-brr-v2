# Documentation Index

## 🚀 Quick Start
1. **[Setup Guide](03-operations/setup-guide.md)** - Complete installation instructions
2. **[Root SETUP.md](../SETUP.md)** - Quick reference for what works
3. **[Training](03-operations/training.md)** - How to train models

## 01 - Data Pipeline
**TUSZ dataset handling and preprocessing**
- [tusz-overview.md](01-data-pipeline/tusz-overview.md) - Dataset structure
- [tusz-data-flow.md](01-data-pipeline/tusz-data-flow.md) - Processing pipeline
- [CACHE_MANIFEST_ARCHITECTURE.md](01-data-pipeline/CACHE_MANIFEST_ARCHITECTURE.md) - Cache strategy
- [cache-rebuild.md](01-data-pipeline/cache-rebuild.md) - Cache rebuild playbook
- [tusz-csv-parser.md](01-data-pipeline/tusz-csv-parser.md) - Label parsing
- [tusz-channels.md](01-data-pipeline/tusz-channels.md) - 10-20 montage
 

## 02 - Architecture
**Model components and specifications**
- [canonical-spec.md](02-architecture/canonical-spec.md) - **SOURCE OF TRUTH**
- [model-full.md](02-architecture/model-full.md) - Complete architecture
- [model-unet.md](02-architecture/model-unet.md) - U-Net encoder/decoder
- [model-mamba.md](02-architecture/model-mamba.md) - Bi-Mamba-2 (O(N))
- [model-rescnn.md](02-architecture/model-rescnn.md) - Residual CNN
- [pipeline-diagram.md](02-architecture/pipeline-diagram.md) - Visual overview

## 03 - Operations
**Training, deployment, evaluation**
- **[setup-guide.md](03-operations/setup-guide.md)** - Complete setup instructions
- [training.md](03-operations/training.md) - Training configurations
- [evaluation.md](03-operations/evaluation.md) - TAES metrics
- [postprocessing.md](03-operations/postprocessing.md) - Hysteresis
- [deploy-local-wsl2.md](03-operations/deploy-local-wsl2.md) - WSL2 setup
- [deploy-modal.md](03-operations/deploy-modal.md) - Cloud deployment
- [troubleshooting.md](03-operations/troubleshooting.md) - Common issues

## 04 - Research
**Future work and experiments**
- [FUTURE_DIRECTION.md](04-research/FUTURE_DIRECTION.md) - Roadmap
- [benchmarks.md](04-research/benchmarks.md) - Performance targets
- [streaming.md](04-research/streaming.md) - Real-time inference
- [FUTURE_STACK_GNN_TCN.md](04-research/FUTURE_STACK_GNN_TCN.md) - Experimental

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
