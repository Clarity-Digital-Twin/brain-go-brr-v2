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
        calculate_metrics_at_thresholds,
        compute_taes,
        evaluate_model,
        find_fa_thresholds,
        plot_roc_curve,
        plot_sensitivity_fa_curve,
    )

    __all__ = [
        "calculate_metrics_at_thresholds",
        "compute_taes",
        "evaluate_model",
        "find_fa_thresholds",
        "plot_roc_curve",
        "plot_sensitivity_fa_curve",
    ]
except ImportError:
    # Clean-slate imports will go here after migration
    pass
