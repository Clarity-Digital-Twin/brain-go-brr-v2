"""Pydantic schemas for config validation - single source of truth for all configs."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data loading and batching configuration."""

    dataset: Literal["tuh_eeg", "chb_mit"] = Field(
        default="tuh_eeg", description="Dataset to use for training"
    )
    data_dir: Path = Field(default=Path("data"), description="Root directory containing EDF files")
    cache_dir: Path = Field(default=Path("cache/data"), description="Data cache directory")
    use_balanced_sampling: bool = Field(default=True, description="Use balanced sampling")
    sampling_rate: Literal[256] = Field(
        default=256, description="Target sampling rate in Hz (fixed at 256)"
    )
    n_channels: Literal[19] = Field(
        default=19, description="Number of EEG channels (10-20 montage)"
    )
    window_size: Literal[60] = Field(
        default=60, description="Window size in seconds (fixed at 60s)"
    )
    stride: Literal[10] = Field(
        default=10, description="Stride between windows in seconds (fixed at 10s)"
    )
    num_workers: int = Field(default=0, ge=0, le=32, description="DataLoader workers")
    validation_split: float = Field(
        default=0.2, ge=0.0, le=0.5, description="Fraction of data for validation"
    )
    max_samples: int | None = Field(
        default=None, ge=1, description="Limit samples for debugging (None = use all)"
    )
    max_hours: float | None = Field(
        default=None, gt=0, description="Limit total hours of data (None = use all)"
    )

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        """Ensure data_dir is Path object."""
        return Path(v) if not isinstance(v, Path) else v


class PreprocessingConfig(BaseModel):
    """Signal preprocessing configuration."""

    montage: Literal["10-20", "standard_1020"] = Field(
        default="10-20", description="EEG montage standard"
    )
    bandpass: tuple[float, float] = Field(
        default=(0.5, 120.0), description="Bandpass filter range in Hz"
    )
    notch_freq: Literal[50, 60] = Field(
        default=60, description="Powerline frequency to notch (50 EU, 60 US)"
    )
    normalize: bool = Field(default=True, description="Apply per-channel z-score normalization")
    use_mne: bool = Field(default=True, description="Use MNE for EDF loading")

    @field_validator("bandpass")
    @classmethod
    def validate_bandpass(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Ensure bandpass range is valid."""
        low, high = v
        if low >= high:
            raise ValueError(f"Bandpass low {low} must be < high {high}")
        if low < 0.1 or high > 200:
            raise ValueError("Bandpass must be in range [0.1, 200] Hz")
        return v


class EncoderConfig(BaseModel):
    """U-Net encoder configuration."""

    stages: Literal[4] = Field(default=4, description="Number of encoder stages")
    channels: list[int] = Field(
        default=[64, 128, 256, 512], description="Channel progression per stage"
    )
    kernel_size: int = Field(default=5, ge=3, le=9, description="Convolution kernel size")
    downsample_factor: Literal[2] = Field(default=2, description="Downsampling factor per stage")

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: list[int]) -> list[int]:
        """Ensure channel progression matches our architecture."""
        if v != [64, 128, 256, 512]:
            raise ValueError("Channels must be [64, 128, 256, 512] for our architecture")
        return v


class ResCNNConfig(BaseModel):
    """ResCNN stack configuration."""

    n_blocks: Literal[3] = Field(default=3, description="Number of ResCNN blocks")
    kernel_sizes: list[int] = Field(default=[3, 5, 7], description="Multi-scale kernel sizes")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")

    @field_validator("kernel_sizes")
    @classmethod
    def validate_kernels(cls, v: list[int]) -> list[int]:
        """Ensure kernel sizes are odd and valid."""
        for k in v:
            if k % 2 == 0:
                raise ValueError(f"Kernel size {k} must be odd")
            if k < 3 or k > 11:
                raise ValueError(f"Kernel size {k} must be in [3, 11]")
        return v


class MambaConfig(BaseModel):
    """Bi-Mamba-2 configuration."""

    n_layers: int = Field(default=6, ge=1, le=12, description="Number of Mamba layers")
    d_model: Literal[512] = Field(default=512, description="Model dimension")
    d_state: Literal[16] = Field(default=16, description="SSM state dimension")
    conv_kernel: int = Field(default=5, ge=3, le=9, description="Mamba convolution kernel")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")


class DecoderConfig(BaseModel):
    """U-Net decoder configuration."""

    stages: Literal[4] = Field(default=4, description="Number of decoder stages")
    kernel_size: int = Field(default=2, ge=2, le=8, description="Transpose conv kernel size")


class ModelConfig(BaseModel):
    """Complete model architecture configuration."""

    name: Literal["seizure_detector"] = Field(default="seizure_detector", description="Model name")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    rescnn: ResCNNConfig = Field(default_factory=ResCNNConfig)
    mamba: MambaConfig = Field(default_factory=MambaConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)


class HysteresisConfig(BaseModel):
    """Hysteresis thresholding configuration."""

    tau_on: float = Field(default=0.86, ge=0.5, le=1.0, description="Upper threshold")
    tau_off: float = Field(default=0.78, ge=0.5, le=1.0, description="Lower threshold")
    min_onset_samples: int = Field(
        default=128, ge=1, description="Min samples above tau_on to enter"
    )
    min_offset_samples: int = Field(
        default=256, ge=1, description="Min samples below tau_off to exit"
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "HysteresisConfig":
        """Ensure tau_on > tau_off."""
        if self.tau_on <= self.tau_off:
            raise ValueError(f"tau_on {self.tau_on} must be > tau_off {self.tau_off}")
        return self


class MorphologyConfig(BaseModel):
    """Morphological operations configuration."""

    opening_kernel: int = Field(default=11, ge=1, description="Opening kernel size (samples)")
    closing_kernel: int = Field(default=31, ge=1, description="Closing kernel size (samples)")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration if available")

    @field_validator("opening_kernel", "closing_kernel")
    @classmethod
    def validate_odd(cls, v: int) -> int:
        """Ensure kernel sizes are odd."""
        if v % 2 == 0:
            raise ValueError(f"Kernel size {v} must be odd")
        return v


class DurationConfig(BaseModel):
    """Event duration filtering configuration."""

    min_duration_s: float = Field(
        default=3.0, ge=0.0, description="Minimum event duration (seconds)"
    )
    max_duration_s: float = Field(
        default=600.0, gt=0.0, description="Maximum event duration (seconds)"
    )

    @model_validator(mode="after")
    def validate_durations(self) -> "DurationConfig":
        """Ensure max >= min."""
        if self.max_duration_s < self.min_duration_s:
            raise ValueError(f"max_duration_s must be >= min_duration_s")
        return self


class EventsConfig(BaseModel):
    """Event merging and confidence configuration."""

    tau_merge: float = Field(default=2.0, ge=0.0, description="Max gap to merge events (seconds)")
    confidence_method: Literal["mean", "peak", "percentile"] = Field(
        default="mean", description="Method for confidence scoring"
    )
    confidence_percentile: float = Field(
        default=0.75, gt=0.0, lt=1.0, description="Percentile for confidence if method='percentile'"
    )


class StitchingConfig(BaseModel):
    """Window stitching configuration."""

    method: Literal["overlap_add", "overlap_add_weighted", "max"] = Field(
        default="overlap_add", description="Stitching method"
    )
    window_size: int = Field(default=15360, ge=1, description="Window size (samples)")
    stride: int = Field(default=2560, ge=1, description="Window stride (samples)")


class PostprocessingConfig(BaseModel):
    """Post-processing pipeline configuration."""

    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    duration: DurationConfig = Field(default_factory=DurationConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    stitching: StitchingConfig = Field(default_factory=StitchingConfig)

    # Backward compatibility - will be removed after migration
    min_duration: float = Field(
        default=3.0, ge=1.0, description="[Deprecated] Use duration.min_duration_s"
    )


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: Literal["cosine", "linear", "constant"] = Field(
        default="cosine", description="Scheduler type"
    )
    warmup_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Warmup fraction of total steps"
    )


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    patience: int = Field(default=5, ge=1, le=50, description="Patience epochs")
    metric: str = Field(
        default="sensitivity_at_10fa",
        description="Metric to monitor (e.g., sensitivity_at_10fa)",
    )
    mode: Literal["max", "min"] = Field(default="max", description="Maximize or minimize metric")


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(default=1, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=256, description="Batch size")
    learning_rate: float = Field(
        default=3e-4, ge=1e-6, le=1e-2, description="Initial learning rate"
    )
    weight_decay: float = Field(default=0.05, ge=0.0, le=0.2, description="AdamW weight decay")
    optimizer: Literal["adamw", "adam", "sgd"] = Field(
        default="adamw", description="Optimizer type"
    )
    resume: bool = Field(default=False, description="Resume from last checkpoint")
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gradient_clip: float = Field(
        default=1.0, ge=0.0, description="Gradient clipping value (0 = disabled)"
    )
    mixed_precision: bool = Field(default=True, description="Use automatic mixed precision (AMP)")
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class EvaluationConfig(BaseModel):
    """Evaluation and metrics configuration."""

    metrics: list[str] = Field(
        default=["taes", "sensitivity", "specificity", "auroc"],
        description="Metrics to compute",
    )
    fa_rates: list[float] = Field(
        default=[10, 5, 2.5, 1], description="False alarm rates per 24h for TAES"
    )
    save_predictions: bool = Field(default=False, description="Save prediction outputs")
    save_plots: bool = Field(default=True, description="Generate evaluation plots")


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    enabled: bool = Field(default=False, description="Enable W&B logging")
    project: str = Field(default="seizure-detection", description="W&B project name")
    entity: str | None = Field(default=None, description="W&B entity/team name")
    tags: list[str] = Field(default_factory=list, description="Run tags")


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""

    name: str = Field(default="debug", description="Experiment name for tracking")
    description: str = Field(default="default experiment", description="Experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: Literal["auto", "cuda", "cpu", "mps"] = Field(
        default="auto", description="Device selection"
    )
    output_dir: Path = Field(default=Path("results"), description="Output directory for results")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging verbosity"
    )
    save_model: bool = Field(default=False, description="Save trained model")
    save_best_only: bool = Field(default=True, description="Only save best checkpoint")
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: Path) -> Path:
        """Ensure cache_dir is Path object."""
        return Path(v) if not isinstance(v, Path) else v


class LoggingConfig(BaseModel):
    """Training logging configuration."""

    log_every_n_steps: int = Field(default=10, ge=1, description="Steps between logging")
    log_gradients: bool = Field(default=False, description="Log gradient norms")
    log_weights: bool = Field(default=False, description="Log weight histograms")


class ResourcesConfig(BaseModel):
    """Compute resource configuration."""

    max_memory_gb: float | None = Field(default=None, gt=0, description="Maximum GPU memory to use")
    distributed: bool = Field(default=False, description="Use distributed training")
    mixed_precision: bool = Field(default=True, description="Use mixed precision training")


class Config(BaseModel):
    """Complete configuration schema for seizure detection pipeline."""

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    resources: ResourcesConfig | None = Field(default=None)

    @model_validator(mode="after")
    def validate_device_resources(self) -> "Config":
        """Ensure GPU settings are consistent."""
        if self.experiment.device == "cpu" and self.training.mixed_precision:
            raise ValueError("Mixed precision requires GPU (cuda/mps)")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load config from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def validate_for_phase(self, phase: str) -> None:
        """Validate config is appropriate for given phase."""
        if phase == "data":
            # Phase 1 only needs data + preprocessing
            assert self.data.sampling_rate == 256, "Must use 256 Hz"
            assert self.data.n_channels == 19, "Must use 19 channels"
            assert self.data.window_size == 60, "Must use 60s windows"
            assert self.data.stride == 10, "Must use 10s stride"
        elif phase == "model":
            # Phase 2 needs model config
            assert len(self.model.encoder.channels) == 4, "Must have 4 encoder stages"
            assert self.model.mamba.d_model == 512, "Mamba d_model must be 512"
        elif phase == "training":
            # Phase 3 needs full config
            assert self.training.epochs > 0, "Must have positive epochs"
            assert self.training.learning_rate > 0, "Must have positive LR"
