"""
Unit tests for NEDCScorer wrapper.

Tests NEDCScorer class that provides Python API to nedc-bench (FILE-LEVEL API).
"""

import pytest

from src.brain_brr.eval.nedc_wrapper import NEDCScorer


class TestNEDCScorerInit:
    """Test NEDCScorer initialization"""

    def test_init_success(self):
        """NEDCScorer initializes successfully with nedc-bench found"""
        scorer = NEDCScorer()
        assert scorer is not None
        assert hasattr(scorer, "beta")

    def test_init_nedc_bench_not_found(self, monkeypatch, tmp_path):
        """NEDCScorer raises ImportError if nedc-bench not found"""
        import src.brain_brr.eval.nedc_wrapper as wrapper_module

        fake_path = tmp_path / "nonexistent"
        monkeypatch.setattr(wrapper_module, "NEDC_BENCH_PATH", fake_path)

        with pytest.raises(ImportError, match="NEDC-BENCH not found"):
            NEDCScorer()


class TestNEDCScorerScoring:
    """Test NEDCScorer scoring functionality (FILE-LEVEL API)"""

    @pytest.fixture
    def scorer(self):
        """Create NEDCScorer instance"""
        return NEDCScorer()

    @pytest.fixture
    def sample_csv_bi_ref(self, tmp_path):
        """
        Create sample reference .csv_bi file WITH # HEADERS.

        Recording: test_001.csv_bi
        Duration: 300s
        Events: 10-30s, 100-150s
        """
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_file = ref_dir / "test_001.csv_bi"
        ref_file.write_text(csv_bi)
        return ref_dir

    @pytest.fixture
    def sample_csv_bi_hyp_perfect(self, tmp_path):
        """Hypothesis matching reference perfectly (100% TP)"""
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    @pytest.fixture
    def sample_csv_bi_hyp_partial(self, tmp_path):
        """
        Hypothesis with partial match:
        - Detects 10-30s (TP)
        - Misses 100-150s (FN)
        - False alarm 200-220s (FP)
        """
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,200.0000,220.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    def test_score_predictions_perfect_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect
    ):
        """
        Score perfect predictions (100% match).

        Expected:
        - Recall: 100% (2/2 seizures detected)
        - Precision: 100% (2/2 detections correct)
        - F1: 1.0
        - TP: 2, FP: 0, FN: 0
        """
        metrics = scorer.score_predictions(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithm="overlap",
        )

        assert metrics.algorithm == "overlap"
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.tp > 0
        assert metrics.fp == 0
        assert metrics.fn == 0

    def test_score_predictions_partial_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_partial
    ):
        """
        Score partial predictions (1 TP, 1 FN, 1 FP).

        Reference: 10-30s, 100-150s
        Hypothesis: 10-30s, 200-220s

        Expected:
        - TP: 1, FN: 1, FP: 1
        - Recall: 50%, Precision: 50%
        """
        metrics = scorer.score_predictions(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_partial,
            algorithm="overlap",
        )

        assert metrics.tp > 0
        assert metrics.fn > 0
        assert metrics.fp > 0
        assert 0 < metrics.recall < 1
        assert 0 < metrics.precision < 1

    def test_score_predictions_batch(self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect):
        """Score with multiple algorithms simultaneously"""
        results = scorer.score_predictions_batch(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithms=["overlap", "epoch"],
        )

        assert isinstance(results, dict)
        assert "overlap" in results
        assert "epoch" in results


class TestNEDCScorerErrors:
    """Test error handling"""

    @pytest.fixture
    def scorer(self):
        return NEDCScorer()

    def test_score_predictions_dir_not_found(self, scorer, tmp_path):
        """NEDCScorer raises FileNotFoundError for missing directories"""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            scorer.score_predictions(
                reference_dir=nonexistent,
                hypothesis_dir=nonexistent,
                algorithm="overlap",
            )

    def test_score_predictions_no_files(self, scorer, tmp_path):
        """NEDCScorer raises ValueError if no .csv_bi files found"""
        empty_ref = tmp_path / "empty_ref"
        empty_hyp = tmp_path / "empty_hyp"
        empty_ref.mkdir()
        empty_hyp.mkdir()

        with pytest.raises(ValueError, match="No .csv_bi files found"):
            scorer.score_predictions(
                reference_dir=empty_ref,
                hypothesis_dir=empty_hyp,
                algorithm="overlap",
            )


class TestNEDCScorerValidation:
    """Test CSV_BI validation"""

    @pytest.fixture
    def scorer(self):
        return NEDCScorer()

    def test_validate_csv_bi_format_valid(self, scorer, tmp_path):
        """validate_csv_bi_format() returns True for valid file"""
        valid_csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
"""
        csv_bi_file = tmp_path / "valid.csv_bi"
        csv_bi_file.write_text(valid_csv_bi)

        assert scorer.validate_csv_bi_format(csv_bi_file) is True

    def test_validate_csv_bi_format_invalid(self, scorer, tmp_path):
        """validate_csv_bi_format() returns False for missing file"""
        invalid_file = tmp_path / "nonexistent.csv_bi"

        assert scorer.validate_csv_bi_format(invalid_file) is False
