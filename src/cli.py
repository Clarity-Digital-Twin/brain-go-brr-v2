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
        f"Batch size: {config.data.batch_size}\n"
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
