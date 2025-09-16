"""ExCa infrastructure wrapper for experiment caching and orchestration.

This module provides a clean interface to ExCa's experiment infrastructure,
handling caching, artifact storage, and run provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from exca import TaskInfra

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ExperimentInfra:
    """Wrapper around ExCa TaskInfra for ML experiments.

    Provides convenient methods for:
    - Creating cached experiment stages
    - Saving artifacts with metadata
    - Recording experiment runs
    - Managing cache lifecycle
    """

    cache_dir: Path
    experiment_name: str
    _task_infra: Optional[TaskInfra] = None

    def __post_init__(self):
        """Initialize the infrastructure after creation."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def task_infra(self) -> TaskInfra:
        """Lazy-loaded TaskInfra instance."""
        if self._task_infra is None:
            self._task_infra = TaskInfra(folder=self.cache_dir)
        return self._task_infra

    def cached_stage(
        self, stage_name: str
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to cache expensive computation stages.

        Args:
            stage_name: Name of the processing stage (used in cache key)

        Returns:
            Decorator that caches function results

        Example:
            @infra.cached_stage("data_preprocessing")
            def preprocess_data(raw_data):
                # Expensive preprocessing
                return processed_data
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            original_func = func
            cached_stage_name = stage_name

            def cached_func(*args, **kwargs) -> T:
                # Ensure TaskInfra is initialized (tracks folder/provenance)
                _ = self.task_infra

                # Allow callers to pass an optional context for cache keying
                cache_context = kwargs.pop("_cache_context", None)

                key_payload = {
                    "experiment": self.experiment_name,
                    "stage": cached_stage_name,
                    "context": cache_context,
                }
                try:
                    key_json = json.dumps(key_payload, sort_keys=True, default=str)
                except Exception:
                    key_json = str(key_payload)

                cache_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]
                cache_base = f"{cached_stage_name}_{cache_hash}"
                cache_file = self.cache_dir / f"{cache_base}.pkl"
                meta_file = self.cache_dir / f"{cache_base}.json"

                # Try loading from cache
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            result = pickle.load(f)
                        logger.info(
                            f"Loaded cached result for {cached_stage_name} ({cache_hash})"
                        )
                        return result
                    except Exception as e:
                        logger.warning(
                            f"Failed to load cache for {cached_stage_name}: {e}. Recomputing."
                        )

                # No cache hit: compute and optionally persist
                logger.info(f"Running stage: {cached_stage_name}")
                start_time = time.time()
                result = original_func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Stage {cached_stage_name} completed in {duration:.2f}s (miss)"
                )

                # Try to cache the result (skip silently if not serializable)
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(result, f)
                    with open(meta_file, "w") as f:
                        json.dump(
                            {
                                "key": key_payload,
                                "timestamp": time.time(),
                                "duration": duration,
                                "cache_file": str(cache_file),
                            },
                            f,
                            indent=2,
                            default=str,
                        )
                    logger.info(
                        f"Cached result for {cached_stage_name} at {cache_file.name}"
                    )
                except Exception as e:
                    # Still record meta for observability
                    try:
                        with open(meta_file, "w") as f:
                            json.dump(
                                {
                                    "key": key_payload,
                                    "timestamp": time.time(),
                                    "duration": duration,
                                    "cacheable": False,
                                    "error": str(e),
                                },
                                f,
                                indent=2,
                                default=str,
                            )
                    except Exception:
                        pass
                    logger.info(
                        f"Result not cacheable for {cached_stage_name}: {e}. Proceeding without cache."
                    )

                return result

            cached_func.__name__ = f"{func.__name__}_{stage_name}"
            return cached_func

        return decorator

    def save_artifact(
        self, artifact: Any, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save an artifact with metadata to the cache directory.

        Args:
            artifact: Object to save (will be pickled)
            name: Name for the artifact file
            metadata: Optional metadata dict to save alongside

        Returns:
            Path where the artifact was saved
        """
        import pickle

        # Save the main artifact
        artifact_path = self.cache_dir / f"{name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)

        # Save metadata if provided
        if metadata:
            metadata_path = self.cache_dir / f"{name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved artifact: {artifact_path}")
        return artifact_path

    def load_artifact(self, name: str) -> Any:
        """Load a previously saved artifact.

        Args:
            name: Name of the artifact (without .pkl extension)

        Returns:
            The loaded artifact object

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        import pickle

        artifact_path = self.cache_dir / f"{name}.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)

        logger.info(f"Loaded artifact: {artifact_path}")
        return artifact

    def save_metrics(self, metrics: Dict[str, float], stage: str = "final") -> Path:
        """Save metrics to a JSON file.

        Args:
            metrics: Dictionary of metric names to values
            stage: Stage name (e.g., 'training', 'validation', 'final')

        Returns:
            Path where metrics were saved
        """
        metrics_path = self.cache_dir / f"metrics_{stage}.json"

        # Add timestamp and experiment metadata
        full_metrics = {
            "experiment_name": self.experiment_name,
            "stage": stage,
            "timestamp": time.time(),
            "metrics": metrics,
        }

        with open(metrics_path, "w") as f:
            json.dump(full_metrics, f, indent=2)

        logger.info(f"Saved {len(metrics)} metrics to {metrics_path}")
        return metrics_path

    def save_model(
        self, model: Any, name: str, metrics: Optional[Dict[str, float]] = None
    ) -> Path:
        """Save a trained model with optional metrics.

        Args:
            model: Model object to save
            name: Name for the model file
            metrics: Optional performance metrics

        Returns:
            Path where model was saved
        """
        model_path = self.cache_dir / f"{name}.pt"

        # Handle different model types
        if hasattr(model, "state_dict"):  # PyTorch model
            import torch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": model.__class__.__name__,
                    "metrics": metrics or {},
                    "timestamp": time.time(),
                },
                model_path,
            )
        else:  # Sklearn or other
            import pickle

            with open(model_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(
                    {
                        "model": model,
                        "metrics": metrics or {},
                        "timestamp": time.time(),
                    },
                    f,
                )
            model_path = model_path.with_suffix(".pkl")

        logger.info(f"Saved model: {model_path}")
        return model_path

    def record_run(self, config: Dict[str, Any], results: Dict[str, Any]) -> Path:
        """Record a complete experiment run.

        Args:
            config: Experiment configuration dictionary
            results: Experiment results dictionary

        Returns:
            Path to the run record file
        """
        run_record = {
            "experiment_name": self.experiment_name,
            "timestamp": time.time(),
            "config": config,
            "results": results,
            "cache_dir": str(self.cache_dir),
        }

        # Generate unique run ID
        run_id = f"run_{int(time.time())}"
        run_path = self.cache_dir / f"{run_id}.json"

        with open(run_path, "w") as f:
            json.dump(run_record, f, indent=2, default=str)

        logger.info(f"Recorded experiment run: {run_path}")
        return run_path

    def list_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """List all artifacts in the cache directory.

        Returns:
            Dictionary mapping artifact names to metadata
        """
        artifacts = {}

        for pkl_file in self.cache_dir.glob("*.pkl"):
            if pkl_file.name.endswith("_metadata.json"):
                continue

            name = pkl_file.stem
            info = {
                "path": str(pkl_file),
                "size_mb": pkl_file.stat().st_size / (1024 * 1024),
                "modified": pkl_file.stat().st_mtime,
            }

            # Check for metadata file
            metadata_file = pkl_file.with_name(f"{name}_metadata.json")
            if metadata_file.exists():
                with open(metadata_file) as f:
                    info["metadata"] = json.load(f)

            artifacts[name] = info

        return artifacts

    def clean_cache(self, keep_recent: int = 5) -> int:
        """Clean old cache files, keeping only the most recent ones.

        Args:
            keep_recent: Number of recent files to keep per type

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        # Group files by type
        file_types = {
            "models": list(self.cache_dir.glob("*.pt"))
            + list(self.cache_dir.glob("*.pkl")),
            "metrics": list(self.cache_dir.glob("metrics_*.json")),
            "runs": list(self.cache_dir.glob("run_*.json")),
        }

        for file_type, files in file_types.items():
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Delete old files
            for old_file in files[keep_recent:]:
                old_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old {file_type} file: {old_file}")

        return deleted_count


def create_infra(cache_dir: Path, experiment_name: str) -> ExperimentInfra:
    """Factory function to create ExperimentInfra instance.

    Args:
        cache_dir: Directory for caching artifacts
        experiment_name: Name of the experiment

    Returns:
        Configured ExperimentInfra instance
    """
    return ExperimentInfra(cache_dir=cache_dir, experiment_name=experiment_name)


# Convenience decorator for quick caching
def cache_result(cache_dir: Path, stage_name: str):
    """Standalone decorator for caching function results.

    Args:
        cache_dir: Directory for caching
        stage_name: Name of the processing stage

    Example:
        @cache_result(Path("cache"), "preprocessing")
        def preprocess_data(data):
            return expensive_preprocessing(data)
    """
    infra = create_infra(cache_dir, "default")
    return infra.cached_stage(stage_name)


__all__ = ["ExperimentInfra", "create_infra", "cache_result"]
