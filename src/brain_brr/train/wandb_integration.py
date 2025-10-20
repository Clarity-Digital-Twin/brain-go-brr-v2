"""Weights & Biases integration for Brain-Go-Brr V3."""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

# Module logger
logger = logging.getLogger(__name__)

try:
    # wandb is optional dependency (may not be installed)
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """W&B logging wrapper with graceful fallback."""

    def __init__(self, config: Any, resume: bool = False):
        """Initialize W&B if enabled and available.

        Args:
            config: Training configuration
            resume: Whether to resume existing W&B run (from config.training.resume)
        """
        self.enabled = False
        self.run = None

        # Check if W&B is enabled in config
        if not hasattr(config.experiment, "wandb") or not config.experiment.wandb.enabled:
            return

        # Check if W&B is installed
        if not WANDB_AVAILABLE:
            logger.warning("W&B enabled but not installed. Install with: pip install wandb")
            return

        # Check for API key
        if not os.getenv("WANDB_API_KEY"):
            logger.warning("W&B enabled but WANDB_API_KEY not set")
            return

        # Initialize W&B
        try:
            out_dir = Path(config.experiment.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id_path = out_dir / ".wandb_run_id"
            resume_existing = False

            # Only resume if explicitly requested AND run ID file exists
            if resume and run_id_path.exists():
                run_id = run_id_path.read_text().strip()
                if run_id:
                    resume_existing = True
                    logger.info(f"[W&B] Resuming existing run: {run_id}")
                else:
                    run_id = uuid.uuid4().hex
                    logger.info(f"[W&B] Empty run ID file, creating new run: {run_id}")
            else:
                run_id = uuid.uuid4().hex
                if resume and not run_id_path.exists():
                    logger.info(
                        f"[W&B] Resume requested but no run ID found, creating new run: {run_id}"
                    )
                elif not resume and run_id_path.exists():
                    logger.info(
                        f"[W&B] Fresh run requested, deleting old run ID and creating new: {run_id}"
                    )
                else:
                    logger.info(f"[W&B] Creating new run: {run_id}")

            # Atomic write of run ID (temp file + rename to prevent torn writes)
            tmp_path = run_id_path.with_suffix(".tmp")
            tmp_path.write_text(run_id)
            os.replace(str(tmp_path), str(run_id_path))

            # Guard optional graph config once to avoid repeated hasattr(None, ...)
            g = getattr(config.model, "graph", None)

            self.run = wandb.init(
                project=config.experiment.wandb.project,
                entity=config.experiment.wandb.entity,
                name=config.experiment.name,
                config={
                    "learning_rate": config.training.learning_rate,
                    "epochs": config.training.epochs,
                    "batch_size": config.training.batch_size,
                    "model": "TCN + Bi-Mamba-2 + GNN (V3)",
                    "optimizer": config.training.optimizer,
                    "scheduler": config.training.scheduler.type
                    if config.training.scheduler
                    else None,
                    "seed": config.experiment.seed,
                    "mixed_precision": config.training.mixed_precision,
                    "gradient_clip": config.training.gradient_clip,
                    "weight_decay": config.training.weight_decay,
                    "warmup_ratio": config.training.scheduler.warmup_ratio
                    if config.training.scheduler
                    else None,
                    "early_stopping_patience": config.training.early_stopping.patience,
                    "early_stopping_metric": config.training.early_stopping.metric,
                    # Model config
                    "architecture": config.model.architecture,
                    "tcn_num_layers": config.model.tcn.num_layers,
                    "tcn_kernel_size": config.model.tcn.kernel_size,
                    "tcn_stride_down": config.model.tcn.stride_down,
                    "mamba_layers": config.model.mamba.n_layers,
                    "mamba_d_model": config.model.mamba.d_model,
                    "mamba_d_state": config.model.mamba.d_state,
                    # Graph config (V3)
                    "graph_enabled": bool(g and getattr(g, "enabled", False)),
                    "graph_edge_features": getattr(g, "edge_features", None),
                    "graph_edge_top_k": getattr(g, "edge_top_k", None),
                    "graph_edge_threshold": getattr(g, "edge_threshold", None),
                    "graph_edge_mamba_layers": getattr(g, "edge_mamba_layers", None),
                    "graph_k_eigenvectors": getattr(g, "k_eigenvectors", None),
                    # Data config
                    "window_size": config.data.window_size,
                    "stride": config.data.stride,
                    "sampling_rate": config.data.sampling_rate,
                    # Post-processing
                    "tau_on": config.postprocessing.hysteresis.tau_on,
                    "tau_off": config.postprocessing.hysteresis.tau_off,
                },
                tags=config.experiment.wandb.tags,
                dir=str(config.experiment.output_dir),
                id=run_id,
                resume="allow",
            )
            self.enabled = True
            status = "resumed" if resume_existing else "created"
            if self.run:
                logger.info(f"[W&B] Run {status}: {self.run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B."""
        if self.enabled and self.run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B logging error: {e}")

    def log_model(self, checkpoint_path: Path, name: str = "model") -> None:
        """Log model checkpoint to W&B."""
        if self.enabled and self.run:
            try:
                artifact = wandb.Artifact(
                    name=name,
                    type="model",
                    description=f"Checkpoint from {checkpoint_path}",
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B artifact logging error: {e}")

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B finish error: {e}")
