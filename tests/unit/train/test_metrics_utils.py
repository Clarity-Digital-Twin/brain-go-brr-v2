"""Tests for metric key normalization utilities.

Testing philosophy: Test BEHAVIOR, not implementation.
"""

from __future__ import annotations

from src.brain_brr.train.metrics_utils import normalize_metric_key, normalize_metrics_dict


class TestNormalizeMetricKey:
    """Test metric key normalization for FA rate matching."""

    def test_normalize_fa_rate_with_decimal(self):
        """Keys with .0fa should be normalized to fa."""
        assert normalize_metric_key("sensitivity_at_10.0fa") == "sensitivity_at_10fa"
        assert normalize_metric_key("sensitivity_at_1.0fa") == "sensitivity_at_1fa"
        assert normalize_metric_key("sensitivity_at_5.0fa") == "sensitivity_at_5fa"

    def test_normalize_already_normalized_key(self):
        """Keys without .0fa should be unchanged."""
        assert normalize_metric_key("sensitivity_at_10fa") == "sensitivity_at_10fa"
        assert normalize_metric_key("sensitivity_at_1fa") == "sensitivity_at_1fa"

    def test_normalize_non_fa_metrics(self):
        """Non-FA metrics should be unchanged."""
        assert normalize_metric_key("auroc") == "auroc"
        assert normalize_metric_key("taes") == "taes"
        assert normalize_metric_key("accuracy") == "accuracy"
        assert normalize_metric_key("f1_score") == "f1_score"

    def test_normalize_edge_cases(self):
        """Edge cases like empty string or unusual formats."""
        assert normalize_metric_key("") == ""
        assert (
            normalize_metric_key("sensitivity_at_10.0fa_precision")
            == "sensitivity_at_10fa_precision"
        )


class TestNormalizeMetricsDict:
    """Test dictionary normalization for backward compatibility."""

    def test_creates_both_keys_for_fa_metrics(self):
        """Should create both original and normalized keys."""
        result = normalize_metrics_dict({"sensitivity_at_10.0fa": 0.85})

        assert "sensitivity_at_10.0fa" in result
        assert "sensitivity_at_10fa" in result
        assert result["sensitivity_at_10.0fa"] == 0.85
        assert result["sensitivity_at_10fa"] == 0.85

    def test_preserves_non_fa_metrics(self):
        """Non-FA metrics should only have original key."""
        result = normalize_metrics_dict({"auroc": 0.92, "taes": 0.88})

        assert "auroc" in result
        assert "taes" in result
        assert result["auroc"] == 0.92
        assert result["taes"] == 0.88

    def test_mixed_metrics_dict(self):
        """Dictionary with both FA and non-FA metrics."""
        metrics = {
            "sensitivity_at_10.0fa": 0.85,
            "sensitivity_at_5.0fa": 0.90,
            "auroc": 0.92,
            "taes": 0.88,
        }

        result = normalize_metrics_dict(metrics)

        assert len(result) == 6
        assert "sensitivity_at_10fa" in result
        assert "sensitivity_at_5fa" in result
        assert "auroc" in result
        assert "taes" in result

    def test_already_normalized_keys_unchanged(self):
        """Keys without .0fa should not create duplicates."""
        result = normalize_metrics_dict({"sensitivity_at_10fa": 0.85})

        assert len(result) == 1
        assert "sensitivity_at_10fa" in result
        assert result["sensitivity_at_10fa"] == 0.85

    def test_empty_dict(self):
        """Empty input should return empty output."""
        assert normalize_metrics_dict({}) == {}

    def test_value_preservation(self):
        """All values should be preserved exactly."""
        metrics = {
            "sensitivity_at_10.0fa": 0.123456789,
            "auroc": 0.987654321,
        }

        result = normalize_metrics_dict(metrics)

        assert result["sensitivity_at_10.0fa"] == 0.123456789
        assert result["sensitivity_at_10fa"] == 0.123456789
        assert result["auroc"] == 0.987654321


class TestMetricNormalizationBehavior:
    """Test real-world metric normalization scenarios."""

    def test_config_key_lookup_compatibility(self):
        """Config uses 'sensitivity_at_10fa', validation creates 'sensitivity_at_10.0fa'."""
        validation_metrics = {
            "sensitivity_at_10.0fa": 0.95,
            "sensitivity_at_5.0fa": 0.92,
            "auroc": 0.88,
        }

        normalized = normalize_metrics_dict(validation_metrics)

        config_key = "sensitivity_at_10fa"
        assert config_key in normalized
        assert normalized[config_key] == 0.95

    def test_backward_compatibility_both_formats(self):
        """Both old and new formats should work."""
        old_format = {"sensitivity_at_10fa": 0.85}
        new_format = {"sensitivity_at_10.0fa": 0.85}

        result_old = normalize_metrics_dict(old_format)
        result_new = normalize_metrics_dict(new_format)

        assert "sensitivity_at_10fa" in result_old
        assert "sensitivity_at_10fa" in result_new
        assert result_old["sensitivity_at_10fa"] == result_new["sensitivity_at_10fa"]

    def test_multiple_fa_rates(self):
        """Should handle multiple FA rates correctly."""
        metrics = {
            "sensitivity_at_1.0fa": 0.75,
            "sensitivity_at_5.0fa": 0.85,
            "sensitivity_at_10.0fa": 0.90,
        }

        result = normalize_metrics_dict(metrics)

        assert "sensitivity_at_1fa" in result
        assert "sensitivity_at_5fa" in result
        assert "sensitivity_at_10fa" in result
        assert result["sensitivity_at_1fa"] == 0.75
        assert result["sensitivity_at_5fa"] == 0.85
        assert result["sensitivity_at_10fa"] == 0.90
