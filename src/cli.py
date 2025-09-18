"""CLI entry point for seizure detection pipeline."""

import sys
from pathlib import Path

import click
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.experiment.schemas import Config

console = Console()


@click.group()
def cli() -> None:
    """Brain-Go-Brr v2: Bi-Mamba-2 + U-Net + ResCNN seizure detection."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--phase", type=click.Choice(["data", "model", "training"]), default=None)
def validate(config_path: Path, phase: str | None) -> None:
    """Validate a YAML configuration file against schemas.

    Args:
        config_path: Path to YAML config file
        phase: Optional phase to validate for (data/model/training)
    """
    try:
        # Load and parse YAML
        console.print(f"[cyan]Validating config:[/cyan] {config_path}")

        with open(config_path) as f:
            yaml_content = f.read()
            data = yaml.safe_load(yaml_content)

        # Validate against schema
        config = Config(**data)

        # Phase-specific validation if requested
        if phase:
            console.print(f"[cyan]Validating for phase:[/cyan] {phase}")
            config.validate_for_phase(phase)

        # Success!
        console.print("[green]✅ Config validation successful![/green]")

        # Pretty print summary
        _print_config_summary(config)

        sys.exit(0)

    except FileNotFoundError as e:
        console.print(f"[red]❌ File not found:[/red] {e}")
        sys.exit(1)

    except yaml.YAMLError as e:
        console.print(f"[red]❌ Invalid YAML:[/red] {e}")
        sys.exit(1)

    except ValidationError as e:
        console.print("[red]❌ Validation failed:[/red]")
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            console.print(f"  [yellow]{loc}:[/yellow] {error['msg']}")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]❌ Unexpected error:[/red] {e}")
        sys.exit(1)


def _print_config_summary(config: Config) -> None:
    """Pretty print config summary."""

    # Create summary table
    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Section", style="cyan")
    table.add_column("Key Settings", style="white")

    # Data settings
    data_summary = (
        f"Dataset: {config.data.dataset}\n"
        f"Batch size: {config.training.batch_size}\n"
        f"Window: {config.data.window_size}s @ {config.data.sampling_rate}Hz\n"
        f"Channels: {config.data.n_channels} (10-20)"
    )
    table.add_row("Data", data_summary)

    # Model settings
    model_summary = (
        f"Encoder: {config.model.encoder.stages} stages {config.model.encoder.channels}\n"
        f"ResCNN: {config.model.rescnn.n_blocks} blocks, kernels {config.model.rescnn.kernel_sizes}\n"
        f"Mamba: {config.model.mamba.n_layers} layers, d_model={config.model.mamba.d_model}"
    )
    table.add_row("Model", model_summary)

    # Training settings
    training_summary = (
        f"Epochs: {config.training.epochs}\n"
        f"LR: {config.training.learning_rate:.1e}\n"
        f"Optimizer: {config.training.optimizer}\n"
        f"Mixed precision: {config.training.mixed_precision}"
    )
    table.add_row("Training", training_summary)

    # Post-processing
    post_summary = (
        f"Hysteresis: τ_on={config.postprocessing.hysteresis.tau_on}, "
        f"τ_off={config.postprocessing.hysteresis.tau_off}\n"
        f"Min duration: {config.postprocessing.min_duration}s"
    )
    table.add_row("Post-processing", post_summary)

    # Experiment
    exp_summary = (
        f"Name: {config.experiment.name}\n"
        f"Device: {config.experiment.device}\n"
        f"Seed: {config.experiment.seed}"
    )
    table.add_row("Experiment", exp_summary)

    console.print(table)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
def train(config_path: Path, resume: bool, device: str) -> None:
    """Train seizure detection model.

    Args:
        config_path: Path to training configuration YAML
        resume: Resume from last checkpoint
        device: Device to train on (auto/cpu/cuda)
    """
    try:
        # Load config
        console.print(f"[cyan]Loading config:[/cyan] {config_path}")
        config = Config.from_yaml(config_path)

        # Override CLI options
        if resume:
            config.training.resume = True
        if device != "auto":
            config.experiment.device = device  # type: ignore[assignment]

        # Import here to avoid circular dependency
        from src.experiment.pipeline import main as train_main

        console.print("[green]Starting training...[/green]")

        # Mock sys.argv for pipeline.main()
        import sys

        old_argv = sys.argv
        sys.argv = ["train", "--config", str(config_path)]
        if resume:
            sys.argv.append("--resume")

        try:
            train_main()
        finally:
            sys.argv = old_argv

    except ValidationError as e:
        console.print(f"[red]Config validation error:[/red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Training error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Config to use")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--output-json", type=click.Path(path_type=Path), help="Save metrics to JSON file")
@click.option(
    "--output-csv-bi", type=click.Path(path_type=Path), help="Export events in CSV_BI format"
)
def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    config: Path | None,
    device: str,
    output_json: Path | None,
    output_csv_bi: Path | None,
) -> None:
    """Evaluate model on test data.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to test data directory
        config: Optional config file (uses checkpoint's config by default)
        device: Device to evaluate on
        output_json: Optional path to save metrics JSON
        output_csv_bi: Optional path to export events in CSV_BI format
    """
    try:
        import json
        from datetime import datetime

        import torch
        from torch.utils.data import DataLoader

        from src.experiment.data import EEGWindowDataset
        from src.experiment.evaluate import batch_probs_to_events
        from src.experiment.events import SeizureEvent
        from src.experiment.export import export_csv_bi
        from src.experiment.models import SeizureDetector
        from src.experiment.pipeline import validate_epoch

        console.print(f"[cyan]Loading checkpoint:[/cyan] {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config
        if config:
            cfg = Config.from_yaml(config)
        elif "config" in checkpoint:
            cfg = Config(**checkpoint["config"])
        else:
            console.print("[red]No config found in checkpoint or provided[/red]")
            sys.exit(1)

        # Create model
        model = SeizureDetector.from_config(cfg.model)
        model.load_state_dict(checkpoint["model_state_dict"])

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Load test data
        edf_files = list(data_path.glob("**/*.edf"))
        console.print(f"[cyan]Found {len(edf_files)} EDF files[/cyan]")

        dataset = EEGWindowDataset(
            edf_files,
            cache_dir=Path(cfg.data.cache_dir) / "test",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=(device == "cuda"),
        )

        # Evaluate
        console.print("[green]Running evaluation...[/green]")
        metrics = validate_epoch(
            model,
            dataloader,
            cfg.postprocessing,
            device=device,
            fa_rates=cfg.evaluation.fa_rates,
        )

        # Print results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            elif key == "thresholds" and isinstance(value, dict):
                for fa_target, threshold in value.items():
                    if isinstance(threshold, (int, float)):
                        table.add_row(f"τ_on @ {fa_target} FA/24h", f"{threshold:.4f}")

        console.print(table)

        # Export metrics to JSON if requested
        if output_json:
            # Add metadata
            metrics["metadata"] = {
                "checkpoint": str(checkpoint_path),
                "data_path": str(data_path),
                "timestamp": datetime.now().isoformat(),
                "device": device,
            }

            output_json.parent.mkdir(parents=True, exist_ok=True)
            with output_json.open("w") as f:
                json.dump(metrics, f, indent=2, default=str)
            console.print(f"[green]✅ Metrics saved to:[/green] {output_json}")

        # Export events to CSV_BI if requested
        if output_csv_bi:
            console.print("[cyan]Generating predictions for CSV_BI export...[/cyan]")

            # Run inference to get probabilities
            model.eval()
            all_probs = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch["window"].to(device)
                    outputs = model(inputs)
                    all_probs.append(outputs.cpu())

            probs = torch.cat(all_probs, dim=0)

            # Convert to events using best threshold
            thresholds_dict = metrics.get("thresholds", {})
            if isinstance(thresholds_dict, dict):
                best_threshold = thresholds_dict.get("10", 0.86)
            else:
                best_threshold = 0.86
            cfg_for_export = cfg.postprocessing
            cfg_for_export.hysteresis.tau_on = best_threshold
            cfg_for_export.hysteresis.tau_off = max(0.0, best_threshold - 0.08)

            pred_events = batch_probs_to_events(probs, cfg_for_export, cfg.data.sampling_rate)

            # Convert to SeizureEvent objects for export
            seizure_events = []
            for record_events in pred_events:
                for start_s, end_s in record_events:
                    seizure_events.append(
                        SeizureEvent(
                            start_s=start_s,
                            end_s=end_s,
                            confidence=0.9,  # Default confidence
                        )
                    )

            # Export to CSV_BI
            total_duration = len(probs) * 60.0  # Assuming 60s windows
            export_csv_bi(
                seizure_events,
                output_csv_bi,
                patient_id="test",
                recording_id="eval",
                duration_s=total_duration,
            )
            console.print(f"[green]✅ Events exported to:[/green] {output_csv_bi}")

    except Exception as e:
        console.print(f"[red]Evaluation error:[/red] {e}")
        sys.exit(1)


@cli.command()
def list_configs() -> None:
    """List available configuration files."""
    config_dir = Path("configs")

    if not config_dir.exists():
        console.print("[yellow]No configs directory found[/yellow]")
        return

    configs = sorted(config_dir.glob("*.yaml"))

    if not configs:
        console.print("[yellow]No config files found[/yellow]")
        return

    table = Table(title="Available Configurations")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")

    for cfg_path in configs:
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            Config(**data)
            status = "✅ Valid"
        except Exception:
            status = "❌ Invalid"

        table.add_row(cfg_path.name, status)

    console.print(table)


# Provide hyphenated alias for convenience
cli.add_command(list_configs, name="list-configs")


def main() -> int:
    """Main entry point."""
    return cli(standalone_mode=False) or 0


if __name__ == "__main__":
    sys.exit(main())
