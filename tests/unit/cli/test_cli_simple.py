"""Simple CLI tests that actually work."""

from pathlib import Path

from click.testing import CliRunner

from src.brain_brr.cli.cli import cli


def test_cli_help():
    """Test main help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Brain-Go-Brr v2" in result.output


def test_validate_help():
    """Test validate help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output.lower()


def test_train_help():
    """Test train help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output.lower()


def test_evaluate_help():
    """Test evaluate help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output.lower()


def test_list_configs():
    """Test list-configs command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list-configs"])
    # Won't fail even if no configs found
    assert result.exit_code == 0


def test_validate_with_valid_config(tmp_path: Path):
    """Test config validation with valid config."""
    config = tmp_path / "test.yaml"
    config.write_text("""
experiment:
  name: test
  seed: 42
data:
  dataset: tuh_eeg
  data_dir: /tmp/data
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  num_workers: 0
model:
  encoder:
    channels: [64, 128, 256, 512]
    stages: 4
  rescnn:
    n_blocks: 3
    kernel_sizes: [3, 5, 7]
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
  decoder:
    stages: 4
    kernel_size: 4
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  optimizer: adamw
postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
  duration:
    min_duration_s: 1.0
    max_duration_s: 300.0
evaluation:
  fa_rates: [10, 5, 1]
""")
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(config)])
    assert result.exit_code == 0
    assert "✅" in result.output or "valid" in result.output.lower()


def test_validate_with_invalid_config(tmp_path: Path):
    """Test config validation with invalid config."""
    config = tmp_path / "bad.yaml"
    config.write_text("""
experiment:
  name: test
# Missing required sections
""")
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(config)])
    assert result.exit_code != 0
    assert "missing" in result.output.lower() or "error" in result.output.lower()


def test_validate_nonexistent_file():
    """Test validation with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_train_missing_config():
    """Test train with missing config."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_evaluate_missing_checkpoint():
    """Test evaluate with missing checkpoint."""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "missing.pt", "/tmp/data"])
    assert result.exit_code != 0
