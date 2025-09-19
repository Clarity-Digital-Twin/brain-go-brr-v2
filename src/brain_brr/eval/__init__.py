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
        calculate_ece,
        calculate_taes,
        evaluate_predictions,
        fa_per_24h,
        find_threshold_for_fa_eventized,
        sensitivity_at_fa_rates,
    )

    __all__ = [
        "calculate_ece",
        "calculate_taes",
        "evaluate_predictions",
        "fa_per_24h",
        "find_threshold_for_fa_eventized",
        "sensitivity_at_fa_rates",
    ]
except ImportError:
    # Clean-slate imports will go here after migration
    pass
