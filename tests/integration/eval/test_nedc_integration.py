"""
Integration test for NEDCScorer with real nedc-bench sample data.

Uses actual sample files from nedc-bench repository to verify:
1. We can import nedc-bench successfully
2. We can call BetaPipeline.evaluate_*() methods
3. We get real results back
"""

from pathlib import Path

import pytest

from src.brain_brr.eval.nedc_wrapper import NEDCScorer


@pytest.mark.integration
class TestNEDCBenchIntegration:
    """Integration tests with nedc-bench"""

    def test_integration_with_nedc_bench_sample_data(self):
        """
        Integration test with nedc-bench sample data.

        Uses actual sample files from nedc-bench repository.
        Verifies we can call nedc-bench and get real results.
        """
        scorer = NEDCScorer()

        nedc_bench_root = Path("reference_repos/nedc-bench")
        sample_ref = nedc_bench_root / "nedc_eeg_eval/v6.0.0/data/csv/ref"
        sample_hyp = nedc_bench_root / "nedc_eeg_eval/v6.0.0/data/csv/hyp"

        if not sample_ref.exists() or not sample_hyp.exists():
            pytest.skip("nedc-bench sample data not found")

        metrics = scorer.score_predictions(
            reference_dir=sample_ref,
            hypothesis_dir=sample_hyp,
            algorithm="overlap",
        )

        assert metrics.tp > 0
        assert metrics.fp >= 0
        assert 0 < metrics.f1 < 1
