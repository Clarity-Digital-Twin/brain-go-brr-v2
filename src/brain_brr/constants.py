"""Central constants for Brain-Go-Brr EEG seizure detection.

This module contains ALL magic numbers, hyperparameters, and configuration
defaults used throughout the codebase. Following Google/DeepMind best practices,
we centralize constants for:
- Data pipeline (channels, sampling rate, windowing)
- Numerical stability (epsilons for different purposes)
- Clinical thresholds (hysteresis, FA targets, event durations)
- Model hyperparameters (dropout, focal loss, optimizer)
- Training configuration (logging, checkpointing, validation)
- File names and metric names (single source of truth)
- Time conversions and preprocessing defaults

Each constant is documented with WHY that value was chosen, not just WHAT it is.

NOTE ON UNUSED CONSTANTS (as of v3.7.0):
Some constants are defined but not actively used in the current codebase.
These are INTENTIONAL RESERVES for future features:
- LABEL_* constants: For future multi-class seizure type detection
- METRIC_* constants: Canonical metric name strings for future standardization
- Hyperparameter constants (ADAMW_*, FOCAL_GAMMA_PRODUCTION, etc.): Documentation
  of standard values even when configs override them

This is acceptable because:
1. They document standard/recommended values
2. They enable future features without breaking changes
3. They provide single source of truth for domain knowledge
4. Cost is minimal (< 1KB in memory)

Constants were audited 2025-10-05 and 6 truly dead constants were removed
(CSV_VERSION_HEADER, AGGREGATE_WINDOW, LOG_BUFFER_CAPACITY, PROB_THRESHOLD_DEFAULT,
SECONDS_PER_DAY, ZSCORE_CLIP_SIGMA).
"""

from __future__ import annotations

import os

# ==============================================================================
# Data Pipeline Constants
# ==============================================================================

# Canonical 10-20 montage order (19 channels)
# CRITICAL CONTRACT: The model is trained assuming this EXACT channel order.
# Channel position defines spatial relationships that U-Net convolutions learn.
# Upstream code MUST map any dataset names to these and preserve this order.
# Breaking this order will cause complete model failure as spatial patterns
# become meaningless (e.g., frontal activity interpreted as occipital).
CHANNEL_NAMES_10_20: list[str] = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    # Midline
    "Fz",
    "Cz",
    "Pz",
    # Right hemisphere
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
]

# Synonyms observed in various datasets (map alternative → canonical)
CHANNEL_SYNONYMS: dict[str, str] = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

# Number of channels in 10-20 montage
N_CHANNELS: int = len(CHANNEL_NAMES_10_20)  # 19

# Sampling / windowing
SAMPLING_RATE: int = 256
WINDOW_SIZE_SEC: int = 60
STRIDE_SIZE_SEC: int = 10

WINDOW_SAMPLES: int = WINDOW_SIZE_SEC * SAMPLING_RATE  # 15360
STRIDE_SAMPLES: int = STRIDE_SIZE_SEC * SAMPLING_RATE  # 2560

# ==============================================================================
# Clinical Thresholds - Threshold Search Configuration
# ==============================================================================

# Binary search bounds for FA rate calibration
# v3.6.0: Expanded from [0.1, 1.0] to [0.0, 1.0] to support low-confidence models
# This allows the search to find thresholds for models that output low probabilities
THRESHOLD_SEARCH_LOW: float = 0.0
THRESHOLD_SEARCH_HIGH: float = 1.0

# Binary search configuration
# Max iterations: 2^10 = 1024 divisions of [0,1] = 0.001 precision (sufficient for clinical use)
THRESHOLD_SEARCH_MAX_ITERS: int = 10
THRESHOLD_SEARCH_TOLERANCE: float = 1e-4  # Convergence criterion

# ==============================================================================
# Numerical Stability Constants
# ==============================================================================

# Ultra-stable epsilon for probability clamping (focal loss)
# Domain: [0, 1], need 7 orders of magnitude below clinical thresholds (0.5-0.95)
EPSILON_PROB_CLAMP: float = 1e-7

# Standard numerical stability for weights/features
# Domain: [-10, 10] normalized features, 6 orders of magnitude safety margin
EPSILON_NUMERICAL: float = 1e-6

# Coarse-grained checks (zero detection)
# For checks like "if count < EPS" where we just want "effectively zero"
EPSILON_ZERO_CHECK: float = 1e-8

# LayerNorm stability (PyTorch default)
EPSILON_NORM: float = 1e-5

# Laplacian regularization (graph theory)
# Eigenvalue conditioning for graph Laplacian (domain-specific)
EPSILON_LAPLACIAN: float = 1e-4

# Optimizer epsilon (AdamW default)
EPSILON_ADAMW: float = 1e-8

# ==============================================================================
# Clinical Hysteresis Thresholds
# ==============================================================================

# Source: Optimized on TUSZ v2.0.3 dev set for 10 FA/24h operating point (v3.4.1)
# Last validated: October 1, 2025
# Clinical justification: Balances sensitivity (>95%) with acceptable FA rate (<10/24h)

HYSTERESIS_TAU_ON: float = 0.86
HYSTERESIS_TAU_OFF: float = 0.78
HYSTERESIS_DELTA: float = 0.08

# ==============================================================================
# Clinical Event Constraints
# ==============================================================================

# False alarm rate targets (per 24 hours) for evaluation
FA_TARGETS: list[float] = [10.0, 5.0, 2.5, 1.0]

# Event duration limits (seconds)
# Clinical constraint: seizures shorter than 3s are artifacts, longer than 10min is status epilepticus
MIN_EVENT_DURATION_S: float = 3.0
MAX_EVENT_DURATION_S: float = 600.0

# Event merging gap (seconds)
# Events within 2s are considered part of same seizure
EVENT_MERGE_GAP_S: float = 2.0

# ==============================================================================
# Post-processing - Morphology
# ==============================================================================

# Morphological operation kernel sizes (samples)
# Source: Optimized on TUSZ v2.0.3 dev set (v3.4.1)
# Opening (11 samples @ 256Hz ≈ 43ms): Removes narrow false positive spikes
# Closing (31 samples @ 256Hz ≈ 121ms): Fills small gaps within seizure events
MORPHOLOGY_OPENING_KERNEL: int = 11
MORPHOLOGY_CLOSING_KERNEL: int = 31

# ==============================================================================
# Model Hyperparameters
# ==============================================================================

# Dropout rates (architecture-specific)
DROPOUT_MAMBA: float = 0.1  # Lower for Mamba (built-in regularization from state-space models)
DROPOUT_TCN: float = 0.15  # Higher for TCN (more parameters, needs stronger regularization)
DROPOUT_GNN: float = 0.1  # GNN dropout
DROPOUT_FUSION: float = 0.1  # Fusion layer dropout (matches GNN/Mamba)

# Focal Loss parameters
# RetinaNet paper defaults (Lin et al. 2017)
FOCAL_ALPHA_DEFAULT: float = 0.25
FOCAL_GAMMA_DEFAULT: float = 2.0

# Brain-BRR production settings (v3.6.0+)
# We use alpha=0.5 (neutral class weighting) with gamma=2.0 for hard-example mining
# For 12:1 imbalance, focal loss focuses learning on difficult cases rather than easy negatives
FOCAL_ALPHA_PRODUCTION: float = 0.5
FOCAL_GAMMA_PRODUCTION: float = 2.0

# Warmup schedule for focal loss gamma
FOCAL_GAMMA_WARMUP_START: float = 1.0  # Start with standard BCE (less amplification)
FOCAL_GAMMA_WARMUP_END: float = 2.0  # Ramp to full hard-example mining

# Loss safety constraints
FOCAL_LOSS_MAX_CLAMP: float = 100.0  # Prevent focal loss explosion
POS_WEIGHT_MAX_CLAMP: float = 20.0  # Prevent over-weighting minority class

# AdamW optimizer defaults
ADAMW_BETA1: float = 0.9
ADAMW_BETA2: float = 0.999
ADAMW_EPS: float = 1e-8

# ==============================================================================
# Training Configuration
# ==============================================================================

# Logging frequency
# Note: Can be overridden via BGB_LOG_EVERY_N_STEPS env var
LOG_EVERY_N_STEPS: int = int(os.getenv("BGB_LOG_EVERY_N_STEPS", "50"))

# Heartbeat interval for long-running training jobs (seconds)
# Periodic "still alive" logging during training to detect hangs/crashes
HEARTBEAT_INTERVAL_SEC: int = 120

# Validation sanity checks
AUROC_FAILURE_THRESHOLD: float = 0.55  # Stop if model barely better than random (0.5)
AUROC_FAILURE_MIN_EPOCH: int = 2  # Grace period for warmup (don't check first 2 epochs)

# ==============================================================================
# File Names and Formats
# ==============================================================================

# Checkpoint file names
CHECKPOINT_LAST: str = "last.pt"
CHECKPOINT_BEST: str = "best.pt"

# Cache and export formats
MANIFEST_FILENAME: str = "manifest.json"

# ==============================================================================
# Metric Names (Canonical Strings for Dict Keys)
# ==============================================================================

METRIC_AUROC: str = "auroc"
METRIC_TAES: str = "taes"
METRIC_SENSITIVITY: str = "sensitivity"
METRIC_SPECIFICITY: str = "specificity"
METRIC_PR_AUC: str = "pr_auc"
METRIC_ECE: str = "ece"

# ==============================================================================
# Time Conversions
# ==============================================================================

HOURS_PER_DAY: int = 24
SECONDS_PER_HOUR: int = 3600

# ==============================================================================
# Preprocessing Defaults
# ==============================================================================

# Bandpass filter (Hz)
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 120.0

# Notch filter (Hz) - US power line frequency
NOTCH_FILTER_HZ: int = 60

# ==============================================================================
# TUSZ Seizure Type Labels (v2.0.3)
# ==============================================================================

LABEL_BACKGROUND: str = "bckg"
LABEL_SEIZURE_GENERIC: str = "seiz"
LABEL_GENERALIZED_NONSPECIFIC: str = "gnsz"
LABEL_FOCAL_NONSPECIFIC: str = "fnsz"
LABEL_COMPLEX_PARTIAL: str = "cpsz"
LABEL_ABSENCE: str = "absz"
LABEL_SIMPLE_PARTIAL: str = "spsz"
LABEL_TONIC_CLONIC: str = "tcsz"
LABEL_TONIC: str = "tnsz"
LABEL_MYOCLONIC: str = "mysz"

SEIZURE_LABELS: set[str] = {
    LABEL_SEIZURE_GENERIC,
    LABEL_GENERALIZED_NONSPECIFIC,
    LABEL_FOCAL_NONSPECIFIC,
    LABEL_COMPLEX_PARTIAL,
    LABEL_ABSENCE,
    LABEL_SIMPLE_PARTIAL,
    LABEL_TONIC_CLONIC,
    LABEL_TONIC,
    LABEL_MYOCLONIC,
}

# ==============================================================================
# Calibration Metrics
# ==============================================================================

ECE_NUM_BINS: int = 10
"""Number of bins for Expected Calibration Error (ECE) calculation.

Standard calibration curve resolution per Guo et al. 2017:
"On Calibration of Modern Neural Networks" (ICML 2017)
https://arxiv.org/abs/1706.04599
"""

TAES_ALPHA_DEFAULT: float = 0.15
"""Default false alarm penalty weight for TAES metric."""

# ==============================================================================
# Balanced Sampling Configuration
# ==============================================================================

BALANCED_SAMPLER_SAMPLE_SIZE: int = 500
"""Number of windows to sample when checking seizure presence in balanced sampler."""

BALANCED_SAMPLER_MAX_SAMPLE: int = 20000
"""Maximum number of windows to check for safety in balanced sampler."""

DATASET_DISTRIBUTION_SAMPLE_SIZE: int = 100
"""Number of windows to sample when checking dataset distribution."""

# ==============================================================================
# Statistics and Logging
# ==============================================================================

PERCENTILE_P25: float = 25.0
"""25th percentile for gradient/weight statistics."""

PERCENTILE_P50: float = 50.0
"""50th percentile (median) for gradient/weight statistics."""

PERCENTILE_P75: float = 75.0
"""75th percentile for gradient/weight statistics."""

PERCENTILE_P95: float = 95.0
"""95th percentile for gradient/weight statistics (outlier detection)."""

# ==============================================================================
# GNN Configuration Defaults
# ==============================================================================

GNN_SSGCONV_ALPHA_DEFAULT: float = 0.05
"""Default alpha mixing parameter for GNN SSGConv layer (when not specified in config)."""

EIGENVALUE_CLAMP_MAX: float = 2.0
"""Maximum eigenvalue for Laplacian stability (prevents numerical overflow)."""

# ==============================================================================
# Model Architecture Defaults
# ==============================================================================

FUSION_NUM_HEADS: int = 4
"""Number of heads for multi-head fusion (PR-4 gated fusion architecture)."""

LAYERSCALE_ALPHA_FALLBACK: float = 0.1
"""Fallback LayerScale alpha when config is missing (defensive default)."""

# ==============================================================================
# FLA / Gated DeltaNet Constants (v4.0+)
# ==============================================================================

FLA_LIBRARY_NAME: str = "flash-linear-attention"
"""PyPI package name for Flash Linear Attention library."""

FLA_MIN_VERSION: str = "0.3.0"
"""Minimum FLA library version (tested and compatible)."""

FLA_MAX_VERSION: str = "0.4.0"
"""Maximum FLA library version (API stability boundary)."""

GDN_QK_PROJECTION_RATIO: float = 0.75
"""GDN design: 0.75* q/k projection for parameter efficiency.

Source: ICLR 2025 paper (arxiv.org/abs/2412.06464), Section 3.2
Maintains ~6* hidden_size² parameter budget when use_gate=True.
"""

GDN_FUSION_MODE_DEFAULT: str = "sum"
"""Bidirectional fusion default: 'sum' (additive) is simpler than 'concat'."""

GDN_ALLOW_NEG_EIGVAL_DEFAULT: bool = False
"""Conservative default: disable β_t ∈ (0,2) until validated."""

GDN_USE_SHORT_CONV_DEFAULT: bool = True
"""CRUCIAL: Short convolution (ablation shows 5.6% ppl drop if disabled)."""

GDN_USE_GATE_DEFAULT: bool = True
"""CRUCIAL: Output gating (ablation shows 6.5% ppl drop if disabled)."""

NODE_D_MODEL: int = 64
"""Node stream model dimension (per-electrode features)."""

NODE_D_STATE: int = 16
"""Node stream SSM state dimension."""

NODE_NUM_LAYERS: int = 6
"""Node stream number of bidirectional layers."""

NODE_EXPAND: int = 2
"""Node stream expansion factor for inner dimension."""

NODE_HEADDIM_BIMAMBA2: int = 8
"""BiMamba2 node stream head dimension (no constraint)."""

GDN_NODE_HEADDIM_DEFAULT: int = 8
"""GDN node stream head dimension: 6 heads * 8 = 48 = 0.75 * 64."""

GDN_NODE_NUM_HEADS_DEFAULT: int = 6
"""GDN node stream number of heads (computed from 0.75* constraint)."""

EDGE_D_MODEL: int = 16
"""Edge stream model dimension (inter-electrode connectivity)."""

EDGE_D_STATE: int = 8
"""Edge stream SSM state dimension."""

EDGE_NUM_LAYERS: int = 2
"""Edge stream number of bidirectional layers."""

EDGE_EXPAND: int = 2
"""Edge stream expansion factor for inner dimension."""

EDGE_HEADDIM_BIMAMBA2: int = 4
"""BiMamba2 edge stream head dimension (no constraint)."""

GDN_EDGE_HEADDIM_DEFAULT: int = 4
"""GDN edge stream head dimension: 3 heads * 4 = 12 = 0.75 * 16."""

GDN_EDGE_NUM_HEADS_DEFAULT: int = 3
"""GDN edge stream number of heads (computed from 0.75* constraint)."""

# ==============================================================================
# Metric Key Formatting
# ==============================================================================

METRIC_SENSITIVITY_TEMPLATE: str = "sensitivity_at_{}fa"


def format_sensitivity_key(fa_rate: float) -> str:
    """Format sensitivity metric key for given FA rate.

    Args:
        fa_rate: False alarm rate (e.g., 10.0 for 10 FA/24h)

    Returns:
        Formatted metric key (e.g., "sensitivity_at_10.0fa")

    Example:
        >>> format_sensitivity_key(10.0)
        'sensitivity_at_10.0fa'
        >>> format_sensitivity_key(1)
        'sensitivity_at_1fa'
    """
    return METRIC_SENSITIVITY_TEMPLATE.format(fa_rate)
