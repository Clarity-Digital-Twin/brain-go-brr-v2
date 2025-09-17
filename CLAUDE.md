# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Brain-Go-Brr v2 - Novel Seizure Detection Architecture

This project pioneers the **first systematic evaluation** of bidirectional Mamba-2 state space models combined with U-Net CNN encoders and Residual CNN stacks for clinical-grade EEG seizure detection. **No one has tested this specific architecture combination before.**

## Core Commands

### Development Workflow
```bash
make quality      # Run all code quality checks (lint, format, type-check)
make test         # Run tests with coverage
make test-fast    # Run tests without coverage (faster)
make lint         # Run ruff linter with auto-fix
make format       # Format code with ruff
make type-check   # Run mypy type checking

# Shortcuts
make q            # quality
make t            # test-fast
make l            # lint
make f            # format
```

### Training & Evaluation
```bash
make train-local  # Train with local config (reduced data)
make train        # Full training with wandb logging
uv run train      # Direct access to training CLI
uv run evaluate   # Direct access to evaluation CLI
uv run run-experiment  # Main experiment runner
```

### Project Setup
```bash
make setup        # Initial project setup with UV
make install      # Install all dependencies
make dev          # Install dev dependencies + pre-commit hooks
make clean        # Clean all artifacts
make update       # Update all dependencies
```

## Novel Architecture: Bi-Mamba-2 + U-Net + ResCNN

### Pipeline Flow
```
EEG Input (19ch, 256Hz) → U-Net Encoder → ResCNN Stack → Bi-Mamba-2 → U-Net Decoder → Hysteresis → TAES
```

### Key Innovations
1. **Bi-Mamba-2 Core**: Replaces transformers with O(N) complexity SSM (vs O(N²))
2. **Hybrid Architecture**: CNN locality + SSM global context in unified model
3. **Hysteresis Thresholding**: Dual thresholds (τ_on=0.86, τ_off=0.78) for FA reduction
4. **TAES-First Design**: Optimizes Time-Aligned Event Scoring, not just accuracy

### Architecture Specifications
- **U-Net Encoder**: 4 stages [64, 128, 256, 512], ×16 downsampling
- **ResCNN**: 3 residual blocks, multi-kernel [3, 5, 7] at bottleneck
- **Bi-Mamba-2**: 6 layers, d_model=512, d_state=16, bidirectional
- **Output**: Per-timestep probabilities @ 256 Hz

## TAES Evaluation Strategy

The project focuses on **Time-Aligned Event Scoring (TAES)** at NEDC thresholds:
- 10 FA/24h: >95% sensitivity (clinical usability)
- 5 FA/24h: >90% sensitivity (ICU standard)
- 1 FA/24h: >75% sensitivity (home monitoring)

TAES prioritizes temporal alignment over point-wise accuracy, matching clinical needs.

## Preprocessing Pipeline (MNE + scipy hybrid)

1. **MNE for I/O**: Robust EDF reading, channel selection, montage handling
2. **scipy for DSP**: Bandpass 0.5-120 Hz, 60 Hz notch, resample to 256 Hz
3. **Windowing**: 60-second windows, 10-second stride (50s overlap)
4. **Normalization**: Per-channel z-score over full recording

## Modern Tooling (2025 Best Practices)

- **UV Package Manager**: 10-100x faster than pip (Rust-powered)
- **Ruff**: Single tool replacing Black, isort, flake8 (Rust-powered)
- **Type Checking**: Strict mypy configuration with no untyped definitions
- **Pre-commit**: Automatic quality checks on every commit

## Key Dependencies

- **mamba-ssm>=2.0.0**: Bi-Mamba-2 implementation
- **mne>=1.5.0**: EEG file I/O and preprocessing
- **torch>=2.0.0**: Deep learning framework
- **einops>=0.7.0**: Tensor operations
- **pydantic>=2.0.0**: Configuration validation

## Important Context

### Why This Architecture?
- **Transformers fail** on long EEG due to O(N²) complexity
- **Pure CNNs** lack global context for seizure evolution
- **Bi-Mamba-2** provides O(N) global modeling with bidirectional context

### Training Strategy
- Cannot use SeizureTransformer pretrained weights (architecture mismatch)
- Training from scratch on TUH EEG Seizure Corpus
- Validation on CHB-MIT, final eval on epilepsybenchmarks.com

### Current Status
- Architecture fully specified in README.md
- Development environment configured with modern tooling
- Implementation placeholders in src/ ready for development
- Literature review complete with key citations identified