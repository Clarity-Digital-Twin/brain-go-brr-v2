"""
NEDCScorer - Direct Python integration with nedc-bench for NEDC v6.0.0 scoring.

NO Docker! NO subprocess! Just sys.path.insert() + Python imports!

CRITICAL: nedc-bench has FILE-LEVEL API, not directory-level.
Must loop over matched file pairs and accumulate counts.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

NEDC_BENCH_PATH = Path(__file__).resolve().parents[3] / "reference_repos" / "nedc-bench" / "src"


@dataclass
class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    CRITICAL: NEDC API only provides counts (tp/fp/fn).
    We compute precision/recall/f1 from counts.
    sensitivity_at_*FA must be computed separately (threshold sweep).
    """

    algorithm: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    total_recording_duration_sec: float
    num_files: int


AlgorithmType = Literal["overlap", "taes", "dp", "epoch", "ira"]


class NEDCScorer:
    """
    Direct Python integration with nedc-bench for official NEDC v6.0.0 scoring.

    Uses sys.path.insert() to import nedc-bench modules directly.
    NO Docker overhead, NO subprocess complexity!

    CRITICAL: nedc-bench has FILE-LEVEL API (evaluate_overlap(ref_file, hyp_file)).
    We must loop over file pairs and accumulate counts.
    """

    def __init__(self) -> None:
        """
        Initialize NEDC scorer with BetaPipeline.

        Raises:
            ImportError: If nedc-bench not found at reference_repos/
            RuntimeError: If nedc-bench import fails
        """
        if not NEDC_BENCH_PATH.exists():
            raise ImportError(
                f"NEDC-BENCH not found at {NEDC_BENCH_PATH}. "
                f"Clone from https://github.com/Clarity-Digital-Twin/nedc-bench"
            )

        sys.path.insert(0, str(NEDC_BENCH_PATH))

        try:
            from nedc_bench.models.annotations import (
                AnnotationFile,  # type: ignore[import-not-found]
            )
            from nedc_bench.orchestration.dual_pipeline import (
                BetaPipeline,  # type: ignore[import-not-found]
            )

            self.BetaPipeline = BetaPipeline
            self.AnnotationFile = AnnotationFile
            self.beta = BetaPipeline()
            logger.info("NEDC-BENCH imported successfully")
        except ImportError as e:
            raise RuntimeError(f"Failed to import nedc-bench: {e}") from e

    def score_predictions(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithm: AlgorithmType = "overlap",
    ) -> NEDCMetrics:
        """
        Score predictions using NEDC-BENCH Python API (FILE-LEVEL).

        CRITICAL: nedc-bench API is FILE-LEVEL, not directory-level!
        We loop over matched file pairs and accumulate counts.

        Args:
            reference_dir: Directory with ground truth .csv_bi files
            hypothesis_dir: Directory with model prediction .csv_bi files
            algorithm: NEDC scoring algorithm (overlap recommended)

        Returns:
            NEDCMetrics with aggregated NEDC scores

        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If no .csv_bi files found
            RuntimeError: If NEDC scoring fails
        """
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
        if not hypothesis_dir.exists():
            raise FileNotFoundError(f"Hypothesis directory not found: {hypothesis_dir}")

        ref_files = sorted(reference_dir.glob("*.csv_bi"))
        hyp_files = sorted(hypothesis_dir.glob("*.csv_bi"))

        if len(ref_files) == 0:
            raise ValueError(f"No .csv_bi files found in reference directory: {reference_dir}")
        if len(hyp_files) == 0:
            raise ValueError(f"No .csv_bi files found in hypothesis directory: {hypothesis_dir}")

        ref_dict = {f.stem: f for f in ref_files}
        hyp_dict = {f.stem: f for f in hyp_files}
        common_files = set(ref_dict.keys()) & set(hyp_dict.keys())

        if len(common_files) == 0:
            raise ValueError("No matching file pairs found between reference and hypothesis")

        logger.info(f"Scoring {len(common_files)} file pairs with '{algorithm}' algorithm...")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_duration = 0.0

        for file_id in sorted(common_files):
            ref_file = ref_dict[file_id]
            hyp_file = hyp_dict[file_id]

            if algorithm == "overlap":
                result = self.beta.evaluate_overlap(ref_file, hyp_file)
            elif algorithm == "taes":
                result = self.beta.evaluate_taes(ref_file, hyp_file)
            elif algorithm == "dp":
                result = self.beta.evaluate_dp(ref_file, hyp_file)
            elif algorithm == "epoch":
                result = self.beta.evaluate_epoch(ref_file, hyp_file)
            elif algorithm == "ira":
                result = self.beta.evaluate_ira(ref_file, hyp_file)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            total_tp += getattr(result, "total_hits", 0)
            total_fp += getattr(result, "total_false_alarms", 0)
            total_fn += getattr(result, "total_misses", 0)

            ref_ann = self.AnnotationFile.from_csv_bi(ref_file)
            total_duration += ref_ann.duration

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"NEDC scoring complete: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        return NEDCMetrics(
            algorithm=algorithm,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=precision,
            recall=recall,
            f1=f1,
            total_recording_duration_sec=total_duration,
            num_files=len(common_files),
        )

    def score_predictions_batch(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithms: list[AlgorithmType],
    ) -> dict[str, NEDCMetrics]:
        """Score predictions with multiple algorithms."""
        results: dict[str, NEDCMetrics] = {}
        for algorithm in algorithms:
            logger.info(f"Scoring with algorithm: {algorithm}")
            results[algorithm] = self.score_predictions(reference_dir, hypothesis_dir, algorithm)
        return results

    def validate_csv_bi_format(self, csv_bi_path: Path) -> bool:
        """
        Validate CSV_BI file format using nedc-bench parser.

        Returns:
            True if valid, False otherwise
        """
        try:
            self.AnnotationFile.from_csv_bi(csv_bi_path)
            return True
        except Exception as e:
            logger.error(f"Invalid CSV_BI format in {csv_bi_path}: {e}")
            return False
