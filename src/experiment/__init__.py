"""Experiment package for ML research with Pydantic + ExCa

This package provides a complete framework for reproducible machine learning
experiments with strong typing, validation, and caching capabilities.
"""

__version__ = "0.1.0"

from .pipeline import Experiment
from .schemas import ExperimentConfig

__all__ = ["ExperimentConfig", "Experiment"]
