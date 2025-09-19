# Brain-Go-Brr v2 Test Suite Hardening Plan

## Executive Summary
Professional-grade test suite refactoring to achieve 95%+ coverage for clinical EEG seizure detection system.
- Current state (measured): 70% coverage; CLI 0%, export 0%
- Target state: 95%+ coverage, hierarchical structure, clinical validation
- Timeline: 4 weeks
- Priority: FDA-submission-ready testing infrastructure

## Current Coverage Snapshot (grounded)
- Overall: 70% (1885 stmts, 573 miss)
- CLI: src/brain_brr/cli/cli.py 0%, src/brain_brr/cli/__init__.py 0%
- Export: src/brain_brr/events/export.py 0%
- Training loop: src/brain_brr/train/loop.py 72%
- Data pipeline:
  - datasets.py 59%, io.py 71%, preprocess.py 86%, windows.py 93%
- Models:
  - detector.py 97%, unet.py 95%, rescnn.py 98%, mamba.py 81%, layers.py 100%
- Post-processing: post/postprocess.py 80%
- Utils: utils/pick_utils.py 49%

Target thresholds and priority
```
Component                     Current     Target    Priority
-----------------------------------------------------------
CLI Interface                 0%          95%       CRITICAL
Export Functions              0%          95%       CRITICAL
Training Loop                 72%         95%       HIGH
Data Pipeline (avg ~72%)      59–93%      90%       HIGH
Models (avg ~91%)             81–100%     95%       MEDIUM
Post-processing               80%         95%       MEDIUM
Utils                         49%         80%       LOW
```

## Phase 1: Test Structure Refactoring (Week 1)

### 1.1 New Directory Structure
```
tests/
├── unit/                      # Isolated component tests
│   ├── cli/
│   │   ├── test_cli_commands.py
│   │   ├── test_cli_validate.py
│   │   └── test_cli_errors.py
│   ├── config/
│   │   ├── test_schemas.py
│   │   └── test_validation.py
│   ├── data/
│   │   ├── test_datasets.py
│   │   ├── test_io.py
│   │   ├── test_preprocess.py
│   │   └── test_windows.py
│   ├── models/
│   │   ├── test_detector.py
│   │   ├── test_mamba.py
│   │   ├── test_unet.py
│   │   ├── test_rescnn.py
│   │   └── test_layers.py
│   ├── post/
│   │   └── test_postprocess.py
│   ├── events/
│   │   ├── test_events.py
│   │   └── test_export.py
│   ├── train/
│   │   └── test_loop.py
│   └── utils/
│       └── test_pick_utils.py
├── integration/              # Component interaction tests
│   ├── test_full_pipeline.py
│   ├── test_model_assembly.py
│   ├── test_data_flow.py
│   └── test_training_cycle.py
├── performance/              # Performance benchmarks
│   ├── test_memory.py
│   ├── test_latency.py
│   └── test_throughput.py
├── clinical/                 # Clinical validation
│   ├── test_taes_metrics.py
│   ├── test_channel_order.py
│   └── test_export_formats.py
├── fixtures/                 # Shared test resources
│   ├── sample_data.py
│   ├── mock_models.py
│   └── test_configs/
└── conftest.py              # Root fixtures
```

### 1.2 Migration Strategy
1. Keep existing tests running during migration
2. Move tests to new structure incrementally
3. Update imports after structure is complete
4. Remove old test files only after validation

## Phase 2: Critical Gap Coverage (Week 1-2)

### 2.1 CLI Testing Suite (0% → 95%)
Notes that reflect the actual CLI:
- Click group: `src.brain_brr.cli.cli:cli`
- validate: positional `config_path` (optional `--phase`)
- train: positional `config_path`, flags `--resume`, `--device` (auto/cpu/cuda)
- evaluate: positional `checkpoint_path` and `data_path`; options `--config`, `--device`, `--output-json`, `--output-csv-bi`

```python
# tests/unit/cli/test_cli_commands.py
import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.brain_brr.cli.cli import cli


class TestCLICommands:
    def test_validate_success(self, tmp_path: Path):
        cfg = tmp_path / "ok.yaml"
        cfg.write_text("""
        experiment: {name: t, seed: 1}
        data: {dataset: tuh_eeg, data_dir: tests, sampling_rate: 256, n_channels: 19, window_size: 60, stride: 10, num_workers: 0}
        model: {encoder: {channels: [64,128], stages: 2}, rescnn: {n_blocks: 1, kernel_sizes: [3,5,7]}, mamba: {n_layers: 1, d_model: 64, d_state: 16}}
        training: {epochs: 0, batch_size: 2, learning_rate: 1e-3, optimizer: adam}
        postprocessing: {hysteresis: {tau_on: 0.86, tau_off: 0.78}, min_duration: 1.0}
        evaluation: {fa_rates: [10,5,1]}
        """, encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(cfg)])
        assert result.exit_code == 0

    def test_validate_failures(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_train_invocation(self, tmp_path: Path):
        # Just exercise argument plumbing; do not actually train
        cfg = tmp_path / "train.yaml"
        cfg.write_text("experiment: {name: t} \n data: {dataset: tuh_eeg, data_dir: tests, sampling_rate: 256, n_channels: 19, window_size: 60, stride: 10, num_workers: 0}\n model: {encoder: {channels: [64,128], stages: 2}, rescnn: {n_blocks: 1, kernel_sizes: [3,5,7]}, mamba: {n_layers: 1, d_model: 64, d_state: 16}}\n training: {epochs: 0, batch_size: 2, learning_rate: 1e-3, optimizer: adam}\n postprocessing: {hysteresis: {tau_on: 0.86, tau_off: 0.78}, min_duration: 1.0}\n evaluation: {fa_rates: [10,5,1]}\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, ["train", str(cfg), "--resume"], catch_exceptions=True)
        assert result.exit_code in (0, 1, 130)  # Training may early-exit in tests

    def test_evaluate_json_export_args_only(self, tmp_path: Path):
        # Evaluate requires checkpoint + data dir; validate CLI surfaces proper error when missing
        runner = CliRunner()
        output = tmp_path / "metrics.json"
        result = runner.invoke(cli, ["evaluate", "checkpoint.pt", str(tmp_path), "--output-json", str(output)])
        assert result.exit_code != 0  # No real checkpoint present; just validates error path
```

### 2.2 Export Functionality (0% → 95%)
Grounded in src/brain_brr/events/export.py (export_csv_bi, export_json, export_batch_csv_bi, validate_csv_bi).

```python
# tests/unit/events/test_export.py
from pathlib import Path
import tempfile
from src.brain_brr.events import SeizureEvent
from src.brain_brr.events.export import (
    export_csv_bi, export_json, export_batch_csv_bi, validate_csv_bi,
)


def test_csv_bi_minimal_valid_file(tmp_path: Path) -> None:
    events = [SeizureEvent(1.0, 2.5, 0.9)]
    out = tmp_path / "P001_R001.csv"
    export_csv_bi(events, out, patient_id="P001", recording_id="R001", duration_s=60.0)
    ok, errors = validate_csv_bi(out)
    assert ok, f"CSV_BI invalid: {errors}"


def test_export_json_roundtrip(tmp_path: Path) -> None:
    events = [SeizureEvent(0.0, 1.0, 0.5), SeizureEvent(2.0, 3.0, 0.8)]
    out = tmp_path / "events.json"
    export_json(events, out, metadata={"src": "unit"})
    data = out.read_text(encoding="utf-8")
    assert "\"events\"" in data and "\"num_events\"" in data


def test_batch_export_file_names(tmp_path: Path) -> None:
    batch = [[SeizureEvent(0.0, 0.5, 0.7)], [SeizureEvent(1.0, 1.5, 0.9)]]
    pats = ["P1", "P2"]
    recs = ["R1", "R2"]
    durs = [10.0, 20.0]
    export_batch_csv_bi(batch, tmp_path, pats, recs, durs)
    for p, r in zip(pats, recs, strict=False):
        assert (tmp_path / f"{p}_{r}.csv").exists()


def test_batch_export_length_mismatch(tmp_path: Path) -> None:
    try:
        export_batch_csv_bi([[SeizureEvent(0.0, 1.0, 0.5)]], tmp_path, ["P1"], ["R1"], [])
        assert False, "Expected ValueError for length mismatch"
    except ValueError:
        pass
```

### 2.3 Training Loop Coverage (75% → 95%)
```python
# tests/unit/train/test_loop.py
class TestTrainingLoop:
    """Comprehensive training loop testing"""

    def test_checkpoint_saving_and_loading(self, tmp_path, model, optimizer):
        """Test checkpoint persistence"""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save
        save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path)
        assert checkpoint_path.exists()

        # Modify model
        original_weight = model.encoder.conv1.weight.clone()
        model.encoder.conv1.weight.data.fill_(0)

        # Load
        load_checkpoint(model, optimizer, checkpoint_path)
        assert torch.allclose(model.encoder.conv1.weight, original_weight)

    def test_early_stopping_triggers(self, mock_trainer):
        """Test early stopping with patience"""
        mock_trainer.config.training.early_stopping.patience = 3

        # Simulate no improvement
        for epoch in range(5):
            mock_trainer.val_metric = 0.5  # No improvement
            should_stop = mock_trainer.check_early_stopping()

            if epoch < 3:
                assert not should_stop
            else:
                assert should_stop
                break

    @pytest.mark.parametrize("scheduler_type", ["cosine", "linear", "constant"])
    def test_lr_scheduling(self, scheduler_type):
        """Test all LR scheduler configurations"""
        scheduler = create_scheduler(scheduler_type, optimizer, epochs=10)

        lrs = []
        for epoch in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # Verify schedule shape
        if scheduler_type == "cosine":
            assert lrs[0] > lrs[5] > lrs[9]  # Decreasing
        elif scheduler_type == "constant":
            assert len(set(lrs)) == 1  # All same
```

## Phase 3: Clinical-Grade Testing (Week 2-3)

### 3.1 Channel Order Invariants
```python
# tests/clinical/test_channel_order.py
CANONICAL_MONTAGE = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
                     "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8",
                     "T4", "T6", "O2"]

class TestChannelOrdering:
    """Ensure 10-20 montage preservation throughout pipeline"""

    @pytest.mark.parametrize("stage", [
        "after_load", "after_preprocess", "after_window",
        "after_model", "after_postprocess"
    ])
    def test_channel_order_preserved(self, edf_file, stage):
        """Channel order must never change"""
        data = load_edf(edf_file)

        if stage == "after_preprocess":
            data = preprocess(data)
        elif stage == "after_window":
            data = create_windows(data)
        # ... etc

        assert data.ch_names == CANONICAL_MONTAGE

    def test_channel_synonym_mapping(self):
        """Test T7→T3, T8→T4, P7→T5, P8→T6 mappings"""
        data_with_new_names = create_test_data(
            channels=["T7", "T8", "P7", "P8"]
        )
        mapped = map_channel_synonyms(data_with_new_names)
        assert "T3" in mapped.ch_names
        assert "T7" not in mapped.ch_names
```

### 3.2 Numerical Stability Tests
```python
# tests/integration/test_numerical_stability.py
class TestNumericalStability:
    """Test model behavior under extreme conditions"""

    @pytest.mark.parametrize("input_type", [
        "zeros", "ones", "large_values", "small_values",
        "inf", "nan", "mixed"
    ])
    def test_model_handles_edge_inputs(self, model, input_type):
        """Model must handle edge cases gracefully"""
        if input_type == "zeros":
            x = torch.zeros(2, 19, 15360)
        elif input_type == "large_values":
            x = torch.ones(2, 19, 15360) * 1e6
        elif input_type == "nan":
            x = torch.full((2, 19, 15360), float('nan'))
        # ... etc

        # Expected behavior today: forward returns finite tensors; training code should
        # sanitize inputs earlier. Guard that outputs are finite for valid inputs.
        output = model(x)
        assert torch.isfinite(output).all()

    def test_gradient_flow_stability(self, model):
        """Ensure stable gradient flow"""
        x = torch.randn(2, 19, 15360, requires_grad=True)

        for _ in range(100):  # Multiple iterations
            output = model(x)
            loss = output.mean()
            loss.backward()

            # Check gradient health
            assert not torch.isnan(x.grad).any()
            assert not torch.isinf(x.grad).any()
            assert x.grad.abs().max() < 100  # No explosion

            x.grad.zero_()
```

### 3.3 TAES Metrics Validation
```python
# tests/clinical/test_taes_metrics.py
class TestTAESMetrics:
    """Validate Time-Aligned Event Scoring metrics"""

    @pytest.mark.parametrize("fa_rate,expected_sens", [
        (10, 0.95),  # >95% sensitivity at 10 FA/24h
        (5, 0.90),   # >90% sensitivity at 5 FA/24h
        (1, 0.75),   # >75% sensitivity at 1 FA/24h
    ])
    def test_taes_thresholds(self, trained_model, test_data, fa_rate, expected_sens):
        """Verify clinical performance targets"""
        predictions = trained_model(test_data)
        metrics = evaluate_predictions(predictions, test_data.labels, fa_rate)

        assert metrics['sensitivity'] >= expected_sens, \
               f"Failed target: {metrics['sensitivity']:.2%} < {expected_sens:.0%} at {fa_rate} FA/24h"
```

## Phase 4: Performance Testing (Week 3)

### 4.1 Memory Profiling
```python
# tests/performance/test_memory.py
class TestMemoryUsage:
    """Profile memory consumption"""

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_inference_memory(self, model, batch_size):
        """Memory should scale linearly with batch size"""
        import tracemalloc

        tracemalloc.start()
        x = torch.randn(batch_size, 19, 15360)

        snapshot_before = tracemalloc.take_snapshot()
        output = model(x)
        snapshot_after = tracemalloc.take_snapshot()

        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_mb = sum(stat.size_diff for stat in stats) / 1024 / 1024

        # Envelope guard; enforce a generous budget per sample
        assert total_mb / batch_size < 500
```

### 4.2 Latency Requirements
```python
# tests/performance/test_latency.py
class TestInferenceLatency:
    """Ensure real-time processing capability"""

    @pytest.mark.performance
    def test_single_window_latency(self, model):
        """Single 60s window must process in <100ms"""
        x = torch.randn(1, 19, 15360)

        # Warmup
        for _ in range(10):
            _ = model(x)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - start)

        p95_latency = np.percentile(times, 95) * 1000  # ms
        assert p95_latency < 100, f"P95 latency {p95_latency:.1f}ms exceeds 100ms target"
```

## Phase 5: Test Infrastructure (Week 4)

### 5.1 Shared Fixtures
```python
# tests/conftest.py
import pytest
from pathlib import Path
import torch
import numpy as np
from click.testing import CliRunner

@pytest.fixture(scope="session")
def sample_edf_data():
    """Generate valid 19-channel EDF test data"""
    from src.brain_brr.constants import CANONICAL_CHANNELS

    # Create synthetic but realistic EEG
    duration = 600  # 10 minutes
    fs = 256
    n_samples = duration * fs

    # Generate with 1/f characteristics
    data = np.random.randn(19, n_samples)
    for i in range(19):
        # Add realistic frequency components
        data[i] += 10 * np.sin(2 * np.pi * 10 * np.arange(n_samples) / fs)  # Alpha
        data[i] += 5 * np.sin(2 * np.pi * 20 * np.arange(n_samples) / fs)   # Beta

    return {
        'data': data,
        'channels': CANONICAL_CHANNELS,
        'fs': fs
    }

@pytest.fixture
def trained_model(tmp_path):
    """Lightweight pre-trained model for testing"""
    from src.brain_brr.models import SeizureDetector

    model = SeizureDetector(
        in_channels=19,
        base_channels=32,  # Smaller for tests
        encoder_depth=2,    # Shallower
        mamba_layers=2,     # Fewer layers
    )

    # Initialize with known weights for reproducibility
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    return model

@pytest.fixture
def cli_runner():
    """Click CLI test runner"""
    return CliRunner()

"""
Optional S3 fixtures (boto3/moto) can be added if we later test modal deploy flows.
Keep out of default suite to avoid adding heavy deps; gate under a marker.
"""
```

### 5.2 Test Utilities
```python
# tests/fixtures/test_helpers.py
def assert_tensor_close(a, b, rtol=1e-5, atol=1e-8):
    """Helper for comparing tensors with tolerance"""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, rtol=rtol, atol=atol)

def create_mock_config(**overrides):
    """Create test config with overrides"""
    from src.brain_brr.config.schemas import Config

    base = {
        'experiment': {'name': 'test', 'seed': 42},
        'data': {'data_dir': 'tests/fixtures/data'},
        'model': {...},
        'training': {'epochs': 1, 'batch_size': 2},
        # ... minimal valid config
    }

    # Deep merge overrides
    deep_update(base, overrides)
    return Config(**base)

class TempConfig:
    """Context manager for temporary configs"""
    def __init__(self, **config):
        self.config = config
        self.path = None

    def __enter__(self):
        import tempfile
        import yaml

        self.tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.tmp)
        self.tmp.close()
        return Path(self.tmp.name)

    def __exit__(self, *args):
        Path(self.tmp.name).unlink()
```

## Implementation Timeline

### Week 1: Foundation
- [x] Create TEST_PLAN.md (this document)
- [ ] Set up new test directory structure (unit/integration/performance/clinical)
- [ ] Migrate existing flat tests into unit/ subfolders (no behavior change)
- [ ] Create shared fixtures and utilities
- [ ] Begin CLI tests (validate/train/evaluate/list-configs)

### Week 2: Critical Coverage
- [ ] Complete CLI testing (0% → 95%)
- [ ] Complete export testing (0% → 95%)
- [ ] Improve training loop coverage (72% → 90%)
- [ ] Add integration test suite (evaluate path w/ CSV_BI)

### Week 3: Clinical Validation
- [ ] Implement property-based tests
- [ ] Add numerical stability tests
- [ ] Create TAES validation suite
- [ ] Add performance benchmarks

### Week 4: Polish & Optimization
- [ ] Reach 95% overall coverage
- [ ] Optimize test execution time (<5 min)
- [ ] Documentation and examples
- [ ] CI/CD integration updates

## Success Metrics
- ✅ Overall coverage ≥95%
- ✅ Critical paths (CLI, export) at 100%
- ✅ All tests pass in <5 minutes
- ✅ Property tests for all invariants
- ✅ Performance benchmarks documented
- ✅ Hierarchical structure matching src/
- ✅ CI/CD gates enforced

## Risk Mitigation
1. **No Breaking Changes**: All refactoring preserves existing functionality
2. **Incremental Migration**: Move tests gradually, validate at each step
3. **Mock Heavy Operations**: Use mocks for I/O, network, and GPU operations
4. **Parallel Execution**: Leverage pytest-xdist for speed
5. **Test Markers**: Use @pytest.mark.slow for optional long tests

## Review Checklist for Senior Sign-off
- [ ] Coverage meets 95% target
- [ ] All critical paths fully tested
- [ ] Test structure mirrors src/ hierarchy
- [ ] Property-based tests for invariants
- [ ] Performance tests with benchmarks
- [ ] Clinical validation (TAES metrics)
- [ ] Documentation complete
- [ ] CI/CD integration validated
- [ ] Test execution time <5 minutes
- [ ] No flaky tests

## Appendix: Coverage Commands
```bash
# Run full test suite with coverage
make test

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v -m integration

# Run performance tests
pytest tests/performance/ -v -m performance

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/cli/test_cli_commands.py -v

# Run with specific markers
pytest -m "not slow" -v

# Parallel execution (faster)
pytest -n auto

# Watch mode for development
ptw -- -x -v
```

---

**Approvals**: Requires senior engineering review before implementation.

**Notes & Dependencies**
- Property-based tests require `hypothesis` (add to dev dependencies) if adopted.
- CSV_BI format tests do not require pandas; use csv/text parsing to avoid extra deps. If using pandas, sync with `uv sync -E eval`.
- Performance tests are gated by `@pytest.mark.performance`.

**Last Updated**: 2025-09-19
