"""Metric utilities for training.

Handles metric key normalization to fix the "New best 0.0000" bug where
metric keys like "sensitivity_at_10.0fa" don't match config keys like
"sensitivity_at_10fa" due to float formatting differences.
"""

from __future__ import annotations


def normalize_metric_key(key: str) -> str:
    """Normalize metric key for consistent lookup.

    Removes trailing ".0" from FA rate metrics to ensure config keys
    match validation output keys.

    Args:
        key: Metric key (e.g., "sensitivity_at_10.0fa" or "sensitivity_at_10fa")

    Returns:
        Normalized key with ".0fa" replaced by "fa"

    Examples:
        >>> normalize_metric_key("sensitivity_at_10.0fa")
        'sensitivity_at_10fa'
        >>> normalize_metric_key("sensitivity_at_1.0fa")
        'sensitivity_at_1fa'
        >>> normalize_metric_key("sensitivity_at_10fa")
        'sensitivity_at_10fa'
        >>> normalize_metric_key("auroc")
        'auroc'
        >>> normalize_metric_key("taes")
        'taes'
    """
    return key.replace(".0fa", "fa")


def normalize_metrics_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize all metric keys in dictionary.

    Creates BOTH normalized and original keys for backward compatibility.
    This ensures config keys like "sensitivity_at_10fa" will match validation
    output keys like "sensitivity_at_10.0fa".

    Args:
        metrics: Dictionary of metric name -> value

    Returns:
        New dictionary with both original and normalized keys

    Examples:
        >>> result = normalize_metrics_dict({"sensitivity_at_10.0fa": 0.5, "auroc": 0.8})
        >>> "sensitivity_at_10fa" in result
        True
        >>> "sensitivity_at_10.0fa" in result
        True
        >>> result["sensitivity_at_10fa"] == result["sensitivity_at_10.0fa"]
        True
    """
    normalized = {}

    for key, value in metrics.items():
        normalized_key = normalize_metric_key(key)

        # Always keep original key
        normalized[key] = value

        # Add normalized key if different (for backward compat)
        if normalized_key != key:
            normalized[normalized_key] = value

    return normalized
