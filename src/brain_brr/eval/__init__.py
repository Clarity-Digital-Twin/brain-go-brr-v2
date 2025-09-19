"""Evaluation metrics and utilities.

This module will contain:
- metrics.py: AUROC, sensitivity, specificity
- taes.py: TAES (Time-Aligned Event Scoring)
- fa_threshold.py: False alarm threshold search
- sensitivity.py: Sensitivity at FA rates
"""

# During migration, re-export from experiment
try:
    from src.experiment.evaluate import (
        compute_taes,
        find_fa_thresholds,
        calculate_metrics_at_thresholds,
        evaluate_model,
        plot_roc_curve,
        plot_sensitivity_fa_curve,
    )

    __all__ = [
        "compute_taes",
        "find_fa_thresholds",
        "calculate_metrics_at_thresholds",
        "evaluate_model",
        "plot_roc_curve",
        "plot_sensitivity_fa_curve",
    ]
except ImportError:
    # Clean-slate imports will go here after migration
    pass
