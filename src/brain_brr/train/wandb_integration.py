"""Weights & Biases integration for Brain-Go-Brr v2."""

import os
import uuid
from pathlib import Path
from typing import Any

try:
    import wandb  # type: ignore[import-not-found]

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """W&B logging wrapper with graceful fallback."""

    def __init__(self, config: Any):
        """Initialize W&B if enabled and available."""
        self.enabled = False
        self.run = None

        # Check if W&B is enabled in config
        if not hasattr(config.experiment, "wandb") or not config.experiment.wandb.enabled:
            return

        # Check if W&B is installed
        if not WANDB_AVAILABLE:
            print("W&B enabled but not installed. Install with: pip install wandb", flush=True)
            return

        # Check for API key
        if not os.getenv("WANDB_API_KEY"):
            print("W&B enabled but WANDB_API_KEY not set", flush=True)
            return

        # Initialize W&B
        try:
            out_dir = Path(config.experiment.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id_path = out_dir / ".wandb_run_id"
            if run_id_path.exists():
                run_id = run_id_path.read_text().strip() or uuid.uuid4().hex
            else:
                run_id = uuid.uuid4().hex
                run_id_path.write_text(run_id)

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
            print(f"W&B run initialized: {wandb.run.url}", flush=True)
        except Exception as e:
            print(f"Failed to initialize W&B: {e}", flush=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B."""
        if self.enabled and self.run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"W&B logging error: {e}", flush=True)

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
                print(f"W&B artifact logging error: {e}", flush=True)

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
            except Exception as e:
                print(f"W&B finish error: {e}", flush=True)
