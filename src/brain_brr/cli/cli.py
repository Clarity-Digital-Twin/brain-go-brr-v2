"""CLI entry point for seizure detection pipeline."""

import sys
from pathlib import Path

import click
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.brain_brr.config.schemas import Config
from src.brain_brr.data.cache_utils import check_cache_completeness, scan_existing_cache

console = Console()


@click.group()
def cli() -> None:
    """Brain-Go-Brr v2: TCN + Bi-Mamba seizure detection."""
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

        # Enforce presence of core sections to catch under-specified configs
        if not isinstance(data, dict):
            console.print("[red]❌ Invalid YAML structure (expected mapping)[/red]")
            sys.exit(1)
        required_sections = ["data", "model", "training", "postprocessing"]
        missing_sections = [k for k in required_sections if k not in data]
        if missing_sections:
            console.print(f"[red]❌ Missing required sections:[/red] {', '.join(missing_sections)}")
            sys.exit(1)

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
    arch = config.model.architecture if hasattr(config.model, "architecture") else "v3"
    model_summary = (
        f"Architecture: {arch.upper()} ({'V3 dual-stream' if arch == 'v3' else arch})\n"
        f"TCN: layers={config.model.tcn.num_layers}, "
        f"k={config.model.tcn.kernel_size}, stride_down={config.model.tcn.stride_down}\n"
        f"Mamba: layers={config.model.mamba.n_layers}, d_model={config.model.mamba.d_model}, "
        f"d_state={config.model.mamba.d_state}, conv_kernel={config.model.mamba.conv_kernel}"
    )

    # Add graph config if enabled
    if config.model.graph and config.model.graph.enabled:
        model_summary += (
            f"\nGraph (V3): edge_features={config.model.graph.edge_features}, "
            f"edge_top_k={config.model.graph.edge_top_k}, "
            f"k_eigenvectors={config.model.graph.k_eigenvectors}"
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
        f"Min duration: {config.postprocessing.duration.min_duration_s}s"
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
        from src.brain_brr.train.loop import main as train_main

        console.print("[green]Starting training...[/green]")

        # Mock sys.argv for pipeline.main()
        old_argv = sys.argv
        sys.argv = ["train", str(config_path)]  # Positional argument now
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


@cli.command("build-cache")
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path), required=True)
@click.option("--validation-split", type=float, default=0.2)
@click.option("--split", type=click.Choice(["train", "dev", "val"]), default="train")
@click.option(
    "--limit-files", type=int, default=None, help="Limit number of files to process (for testing)"
)
def build_cache_cmd(
    data_dir: Path, cache_dir: Path, validation_split: float, split: str, limit_files: int | None
) -> None:
    """Build cache for a chosen split under DATA_DIR into CACHE_DIR."""
    try:
        from src.brain_brr.data import EEGWindowDataset
        from src.brain_brr.utils.env import EnvConfig

        edf_files_all = sorted(Path(data_dir).glob("**/*.edf"))

        # Apply limit from CLI option first, then env var as fallback
        if limit_files is not None:
            edf_files_all = edf_files_all[:limit_files]
            console.print(f"[yellow]Limiting to {limit_files} files (from --limit-files)[/yellow]")
        else:
            limit_from_env = EnvConfig.limit_files()
            if limit_from_env is not None and limit_from_env > 0:
                edf_files_all = edf_files_all[:limit_from_env]
                console.print(
                    f"[yellow]Limiting to {limit_from_env} files (from BGB_LIMIT_FILES)[/yellow]"
                )

        # For official TUSZ usage, point --data-dir at a single split (train/ or dev/)
        # and select --split accordingly. We no longer re-split here; we build the
        # cache for all files under the provided directory. 'val' is accepted as an
        # alias for 'dev' for backward compatibility.
        if split == "val":
            split = "dev"
        edf_files = edf_files_all
        label_files = [p.with_suffix(".csv") for p in edf_files]

        console.print(
            f"[cyan]Building cache for {split} split ({len(edf_files)} files) from[/cyan] {data_dir}"
        )

        status = check_cache_completeness(edf_files, cache_dir)
        console.print(
            f"[cyan]Cache status:[/cyan] {status.cached_files}/{status.total_files} present, "
            f"missing {status.missing_files}"
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        _ = EEGWindowDataset(edf_files, label_files=label_files, cache_dir=cache_dir)
        status2 = check_cache_completeness(edf_files, cache_dir)
        if status2.missing_files == 0:
            try:
                manifest = scan_existing_cache(cache_dir)
                console.print(
                    "[green]✅ Cache build complete + manifest:[/green] "
                    f"partial={len(manifest['partial_seizure'])}, "
                    f"full={len(manifest['full_seizure'])}, "
                    f"none={len(manifest['no_seizure'])}"
                )
                if (
                    len(manifest.get("partial_seizure", [])) == 0
                    and len(manifest.get("full_seizure", [])) == 0
                ):
                    console.print(
                        "[red]❌ No seizures found in cache. Refusing to proceed to training.[/red]"
                    )
                    sys.exit(2)
            except Exception as e:
                console.print(f"[yellow]Built cache but manifest scan failed: {e}[/yellow]")
        else:
            console.print(
                f"[yellow]⚠️ Cache incomplete: {status2.missing_files} missing after build[/yellow]"
            )
    except Exception as e:
        console.print(f"[red]Cache build error:[/red] {e}")
        sys.exit(1)


@cli.command("scan-cache")
@click.option("--cache-dir", type=click.Path(exists=True, path_type=Path), required=True)
def scan_cache_cmd(cache_dir: Path) -> None:
    """Scan NPZ cache and build a seizure-category manifest."""
    try:
        manifest = scan_existing_cache(cache_dir)
        console.print(
            "[green]✅ Manifest created:[/green] "
            f"partial={len(manifest['partial_seizure'])}, "
            f"full={len(manifest['full_seizure'])}, "
            f"none={len(manifest['no_seizure'])} at {cache_dir / 'manifest.json'}"
        )
        if (
            len(manifest.get("partial_seizure", [])) == 0
            and len(manifest.get("full_seizure", [])) == 0
        ):
            console.print(
                "[red]❌ No seizures detected across cache — investigate CSV parsing/paths"
            )
            sys.exit(2)
    except Exception as e:
        console.print(f"[red]Manifest scan error:[/red] {e}")
        sys.exit(1)


@cli.command("evaluate")
@click.argument(
    "checkpoint_path", type=click.Path(exists=False, path_type=Path)
)  # Allow non-existent for dry-run
@click.argument(
    "data_path", type=click.Path(exists=False, path_type=Path)
)  # Allow non-existent for dry-run
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Config to use")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--output-json", type=click.Path(path_type=Path), help="Save metrics to JSON file")
@click.option(
    "--output-csv-bi", type=click.Path(path_type=Path), help="Export events in CSV_BI format"
)
@click.option("--dry-run", is_flag=True, help="Validate args and emit empty reports")
def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    config: Path | None,
    device: str,
    output_json: Path | None,
    output_csv_bi: Path | None,
    dry_run: bool,
) -> None:
    """Evaluate model on test data.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to test data directory
        config: Optional config file (uses checkpoint's config by default)
        device: Device to evaluate on
        output_json: Optional path to save metrics JSON
        output_csv_bi: Optional path to export events in CSV_BI format
        dry_run: If True, validate args and emit empty reports without loading model
    """
    # Handle dry-run mode
    if dry_run:
        console.print("[cyan]Running in dry-run mode[/cyan]")

        # Create empty reports if requested
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            metrics = {
                "checkpoint": str(checkpoint_path),
                "data_dir": str(data_path),
                "events": [],
                "sensitivity": 0.0,
                "specificity": 0.0,
                "auc": 0.0,
            }
            import json

            with output_json.open("w") as f:
                json.dump(metrics, f, indent=2)
            console.print(f"[green]✅ Created empty JSON:[/green] {output_json}")

        if output_csv_bi:
            output_csv_bi.parent.mkdir(parents=True, exist_ok=True)
            output_csv_bi.write_text("record,start_s,end_s,confidence\n")
            console.print(f"[green]✅ Created empty CSV:[/green] {output_csv_bi}")

        console.print("[green]Dry-run evaluation complete[/green]")
        return

    # Check files exist for real evaluation
    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint_path}")
        sys.exit(1)
    if not data_path.exists():
        console.print(f"[red]Data path not found:[/red] {data_path}")
        sys.exit(1)

    try:
        import json
        from datetime import datetime

        import torch
        from torch.utils.data import DataLoader

        from src.brain_brr.data import EEGWindowDataset
        from src.brain_brr.eval.metrics import batch_probs_to_events
        from src.brain_brr.events import SeizureEvent
        from src.brain_brr.events.export import export_csv_bi
        from src.brain_brr.models import SeizureDetector
        from src.brain_brr.train.loop import validate_epoch

        console.print(f"[cyan]Loading checkpoint:[/cyan] {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config
        if config:
            cfg = Config.from_yaml(config)
        elif "config" in checkpoint and checkpoint["config"] is not None:
            cfg = Config(**checkpoint["config"])
        else:
            console.print(
                "[red]No config found in checkpoint or provided. Please provide --config[/red]"
            )
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

        if len(edf_files) == 0:
            console.print("[red]No EDF files found under data path[/red]")
            sys.exit(1)

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
                for windows, _labels in dataloader:
                    inputs = windows.to(device)
                    logits = model(inputs)
                    probs_batch = torch.sigmoid(logits)
                    all_probs.append(probs_batch.cpu())

            probs = torch.cat(all_probs, dim=0)

            # Convert to events using best threshold
            thresholds_dict = metrics.get("thresholds", {})
            if isinstance(thresholds_dict, dict):
                # Be robust to key types: accept "10", 10, or 10.0
                if "10" in thresholds_dict:
                    best_threshold = thresholds_dict["10"]
                elif 10 in thresholds_dict:
                    best_threshold = thresholds_dict[10]
                elif 10.0 in thresholds_dict:
                    best_threshold = thresholds_dict[10.0]
                else:
                    best_threshold = 0.86
            else:
                best_threshold = 0.86
            cfg_for_export = cfg.postprocessing
            cfg_for_export.hysteresis.tau_on = best_threshold
            cfg_for_export.hysteresis.tau_off = max(0.0, best_threshold - 0.08)

            pred_events = batch_probs_to_events(probs, cfg_for_export, cfg.data.sampling_rate)

            # Convert to SeizureEvent objects for export with proper timing
            # Windows overlap with 10s stride: window_i starts at i * stride_seconds
            stride_s = cfg.data.stride  # 10 seconds
            window_s = cfg.data.window_size  # 60 seconds
            seizure_events = []
            for window_idx, record_events in enumerate(pred_events):
                window_start_s = window_idx * stride_s
                for start_s, end_s in record_events:
                    # Adjust event times relative to the window's actual position
                    seizure_events.append(
                        SeizureEvent(
                            start_s=window_start_s + start_s,
                            end_s=window_start_s + end_s,
                            confidence=0.9,  # Default confidence
                        )
                    )

            # Export to CSV_BI
            # Correct total duration calculation accounting for stride
            total_duration = (len(probs) - 1) * stride_s + window_s if len(probs) > 0 else 0.0
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

    # Search recursively for YAML files in subdirectories
    configs = sorted(config_dir.rglob("*.yaml"))

    if not configs:
        console.print("[yellow]No config files found[/yellow]")
        return

    table = Table(title="Available Configurations")
    table.add_column("Environment", style="magenta")
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

        # Get relative path from configs directory
        relative_path = cfg_path.relative_to(config_dir)

        # Extract environment (local/modal) from path
        parts = relative_path.parts
        if len(parts) > 1:
            environment = parts[0]
            filename = parts[-1]
        else:
            environment = "root"
            filename = parts[0]

        table.add_row(environment, filename, status)

    console.print(table)


# Provide hyphenated alias for convenience
cli.add_command(list_configs, name="list-configs")


def main() -> int:
    """Main entry point."""
    return cli(standalone_mode=False) or 0


if __name__ == "__main__":
    sys.exit(main())
