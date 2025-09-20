"""Comprehensive CLI command testing for Brain-Go-Brr v2."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from src.brain_brr.cli.cli import cli


class TestCLIValidateCommand:
    """Test the validate command."""

    def test_validate_success(self, cli_runner: CliRunner, valid_config_yaml: Path):
        """Test successful config validation."""
        result = cli_runner.invoke(cli, ["validate", str(valid_config_yaml)])
        assert result.exit_code == 0
        assert "✅" in result.output or "valid" in result.output.lower()

    def test_validate_with_phase(self, cli_runner: CliRunner, valid_config_yaml: Path):
        """Test validation with specific phase."""
        result = cli_runner.invoke(cli, ["validate", str(valid_config_yaml), "--phase", "training"])
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self, cli_runner: CliRunner):
        """Test validation with non-existent file."""
        result = cli_runner.invoke(cli, ["validate", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_validate_malformed_yaml(self, cli_runner: CliRunner, tmp_path: Path):
        """Test validation with malformed YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{ this is: not valid yaml [}", encoding="utf-8")

        result = cli_runner.invoke(cli, ["validate", str(bad_yaml)])
        assert result.exit_code != 0

    def test_validate_missing_required_fields(self, cli_runner: CliRunner, tmp_path: Path):
        """Test validation with missing required fields."""
        incomplete_yaml = tmp_path / "incomplete.yaml"
        incomplete_yaml.write_text(
            """
        experiment:
          name: test
        # Missing data, model, training sections
        """,
            encoding="utf-8",
        )

        result = cli_runner.invoke(cli, ["validate", str(incomplete_yaml)])
        assert result.exit_code != 0

    def test_validate_invalid_values(self, cli_runner: CliRunner, tmp_path: Path):
        """Test validation with invalid values."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text(
            """
        experiment:
          name: test
          seed: -1  # Invalid negative seed
        data:
          dataset: tuh_eeg
          data_dir: /tmp
          sampling_rate: -256  # Invalid negative
          n_channels: 19
          window_size: 60
          stride: 10
          num_workers: 0
        model:
          encoder: {channels: [64, 128, 256, 512], stages: 4}
          rescnn: {n_blocks: 3, kernel_sizes: [3, 5, 7]}
          mamba: {n_layers: 6, d_model: 512, d_state: 16}
          decoder: {stages: 4, kernel_size: 4}
        training:
          epochs: -1  # Invalid negative
          batch_size: 2
          learning_rate: 1e-3
          optimizer: adamw
        postprocessing:
          hysteresis: {tau_on: 0.86, tau_off: 0.78}
        """,
            encoding="utf-8",
        )

        result = cli_runner.invoke(cli, ["validate", str(invalid_yaml)])
        assert result.exit_code != 0


class TestCLITrainCommand:
    """Test the train command."""

    @patch("src.brain_brr.train.loop.main")
    def test_train_basic(self, mock_train_main, cli_runner: CliRunner, valid_config_yaml: Path):
        """Test basic training invocation."""
        mock_train_main.return_value = None

        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml)])
        assert result.exit_code == 0
        mock_train_main.assert_called_once()

    @patch("src.brain_brr.train.loop.main")
    def test_train_with_resume(
        self, mock_train_main, cli_runner: CliRunner, valid_config_yaml: Path
    ):
        """Test training with resume flag."""
        mock_train_main.return_value = None

        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml), "--resume"])
        assert result.exit_code == 0
        mock_train_main.assert_called_once()

    @patch("src.brain_brr.train.loop.main")
    def test_train_with_device_selection(
        self, mock_train_main, cli_runner: CliRunner, valid_config_yaml: Path
    ):
        """Test training with device selection."""
        mock_train_main.return_value = None

        # Test CPU
        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml), "--device", "cpu"])
        assert result.exit_code == 0

        # Test CUDA
        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml), "--device", "cuda"])
        assert result.exit_code == 0

        # Test auto
        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml), "--device", "auto"])
        assert result.exit_code == 0

    def test_train_invalid_config(self, cli_runner: CliRunner):
        """Test training with invalid config."""
        result = cli_runner.invoke(cli, ["train", "nonexistent.yaml"])
        assert result.exit_code != 0

    @patch("src.brain_brr.train.loop.main")
    def test_train_exception_handling(
        self, mock_train_main, cli_runner: CliRunner, valid_config_yaml: Path
    ):
        """Test training exception handling."""
        mock_train_main.side_effect = RuntimeError("Training failed")

        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml)])
        assert result.exit_code != 0


class TestCLIEvaluateCommand:
    """Test the evaluate command."""

    def test_evaluate_basic(self, cli_runner: CliRunner, temp_checkpoint: Path, tmp_path: Path):
        """Test basic evaluation - simplified to just check args parsing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Just test that the command parses args correctly with dry-run
        result = cli_runner.invoke(
            cli, ["evaluate", str(temp_checkpoint), str(data_dir), "--dry-run"]
        )
        # Dry-run should succeed without loading actual checkpoint
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()

    def test_evaluate_with_json_output(
        self, cli_runner: CliRunner, temp_checkpoint: Path, tmp_path: Path
    ):
        """Test evaluation with JSON output."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_json = tmp_path / "metrics.json"

        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                str(temp_checkpoint),
                str(data_dir),
                "--output-json",
                str(output_json),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0  # dry-run should succeed
        assert output_json.exists()
        import json

        with output_json.open() as f:
            data = json.load(f)
            assert "checkpoint" in data
            assert "events" in data

    def test_evaluate_with_csv_export(
        self, cli_runner: CliRunner, temp_checkpoint: Path, tmp_path: Path
    ):
        """Test evaluation with CSV-BI export."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_csv = tmp_path / "events.csv"

        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                str(temp_checkpoint),
                str(data_dir),
                "--output-csv-bi",
                str(output_csv),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0  # dry-run should succeed
        assert output_csv.exists()
        assert "record,start_s,end_s,confidence" in output_csv.read_text()

    def test_evaluate_with_config_override(
        self,
        cli_runner: CliRunner,
        temp_checkpoint: Path,
        tmp_path: Path,
        valid_config_yaml: Path,
    ):
        """Test evaluation with config override."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                str(temp_checkpoint),
                str(data_dir),
                "--config",
                str(valid_config_yaml),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0  # dry-run should succeed

    def test_evaluate_missing_checkpoint(self, cli_runner: CliRunner, tmp_path: Path):
        """Test evaluation with missing checkpoint."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = cli_runner.invoke(cli, ["evaluate", "missing.pt", str(data_dir)])
        assert result.exit_code != 0

    def test_evaluate_missing_data(self, cli_runner: CliRunner, temp_checkpoint: Path):
        """Test evaluation with missing data directory."""
        result = cli_runner.invoke(cli, ["evaluate", str(temp_checkpoint), "/nonexistent/data"])
        assert result.exit_code != 0

    @patch("torch.load")  # Just patch torch.load to avoid the actual loading
    def test_evaluate_device_selection(
        self, mock_evaluate, cli_runner: CliRunner, temp_checkpoint: Path, tmp_path: Path
    ):
        """Test evaluation with different devices."""
        mock_evaluate.return_value = {"auc": 0.95}

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Test CPU
        result = cli_runner.invoke(
            cli, ["evaluate", str(temp_checkpoint), str(data_dir), "--device", "cpu"]
        )
        assert result.exit_code != 0  # evaluate command not yet implemented

        # Test auto
        result = cli_runner.invoke(
            cli, ["evaluate", str(temp_checkpoint), str(data_dir), "--device", "auto"]
        )
        assert result.exit_code != 0  # evaluate command not yet implemented


class TestCLIListConfigs:
    """Test the list-configs command."""

    def test_list_configs(self, cli_runner: CliRunner):
        """Test listing available configs."""
        result = cli_runner.invoke(cli, ["list-configs"])
        assert result.exit_code == 0
        # Should list configs from configs/ directory
        assert "configs" in result.output.lower() or "available" in result.output.lower()


class TestCLIHelp:
    """Test help output."""

    def test_main_help(self, cli_runner: CliRunner):
        """Test main help output."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "validate" in result.output
        assert "train" in result.output
        assert "evaluate" in result.output

    def test_validate_help(self, cli_runner: CliRunner):
        """Test validate command help."""
        result = cli_runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "config_path" in result.output.lower() or "config" in result.output.lower()

    def test_train_help(self, cli_runner: CliRunner):
        """Test train command help."""
        result = cli_runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "resume" in result.output
        assert "device" in result.output

    def test_evaluate_help(self, cli_runner: CliRunner):
        """Test evaluate command help."""
        result = cli_runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.output.lower()
        assert "output-json" in result.output or "json" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_unknown_command(self, cli_runner: CliRunner):
        """Test unknown command handling."""
        result = cli_runner.invoke(cli, ["unknown-command"])
        assert result.exit_code != 0

    def test_missing_required_args(self, cli_runner: CliRunner):
        """Test missing required arguments."""
        # Validate without config
        result = cli_runner.invoke(cli, ["validate"])
        assert result.exit_code != 0

        # Train without config
        result = cli_runner.invoke(cli, ["train"])
        assert result.exit_code != 0

        # Evaluate without checkpoint
        result = cli_runner.invoke(cli, ["evaluate"])
        assert result.exit_code != 0

    def test_invalid_option_values(self, cli_runner: CliRunner, valid_config_yaml: Path):
        """Test invalid option values."""
        # Invalid device
        result = cli_runner.invoke(cli, ["train", str(valid_config_yaml), "--device", "invalid"])
        assert result.exit_code != 0

    @patch("src.brain_brr.cli.cli.sys.exit")
    def test_keyboard_interrupt_handling(
        self, mock_exit, cli_runner: CliRunner, valid_config_yaml: Path
    ):
        """Test keyboard interrupt handling."""
        with patch("src.brain_brr.train.loop.main", side_effect=KeyboardInterrupt):
            cli_runner.invoke(cli, ["train", str(valid_config_yaml)])
            # Should handle gracefully
