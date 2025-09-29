# Logging Migration Plan: From Print to Production

**Date**: September 28, 2025
**Status**: PLANNING
**Estimated Effort**: 2-3 days
**Priority**: HIGH - Critical for production deployment and debugging

## 📊 Current State Analysis

### Print Statement Statistics
- **Total print() statements**: 387
  - `src/`: 247 occurrences across 11 files
  - `deploy/`: 140 occurrences across 3 files
- **Real-time prints with `flush=True`**: 147 total
  - `train/loop.py`: 87 instances
  - `deploy/modal/app.py`: 42 instances
  - Others: 18 instances across 5 files
- **Rich console.print() calls**: 47 in `cli/cli.py` (DO NOT MIGRATE - user-facing)
- **Files already using logging**: 5
  - `data/io.py:19` - logger = logging.getLogger(__name__)
  - `models/clamp_utils.py:12` - logger = logging.getLogger(__name__)
  - `models/tcn.py:16` - logger = logging.getLogger(__name__)
  - `models/mamba.py:19` - logger = logging.getLogger(__name__)
  - `literature/pdf_to_markdown.py:42` - logger = logging.getLogger(__name__)
- **No central logging configuration exists**

### File Breakdown
| File | Print Count | Priority | Category |
|------|------------|----------|----------|
| `src/brain_brr/train/loop.py` | 147 (87 flush) | CRITICAL | Training progress, metrics, NaN detection |
| `deploy/modal/app.py` | 114 (42 flush) | HIGH | Deployment status, cache operations |
| `src/brain_brr/cli/cli.py` | 47 print + 47 console.print | HIGH | User-facing CLI output (KEEP console.print) |
| `deploy/modal/cleanup_volume.py` | 16 | MEDIUM | Volume maintenance |
| `src/brain_brr/data/tusz_splits.py` | 13 | HIGH | Data validation, patient splits |
| `src/brain_brr/data/datasets.py` | 11 (6 flush) | HIGH | Dataset loading, sampling |
| `deploy/modal/inspect_volume.py` | 10 | LOW | Diagnostic tool |
| `src/brain_brr/train/wandb_integration.py` | 7 | MEDIUM | W&B logging |
| `src/brain_brr/data/cache_utils.py` | 6 | MEDIUM | Cache operations |
| `src/brain_brr/models/tcn.py` | 5 | MEDIUM | Model initialization |
| `src/brain_brr/models/debug_utils.py` | 5 | LOW | Debug utilities |
| Others | <5 each | LOW | Various |

### Print Categories Identified
1. **Training Progress** (40%): Epoch/batch progress, loss values, metrics
2. **Data Operations** (20%): Cache building, dataset loading, sampling
3. **Debug/Diagnostic** (15%): NaN detection, gradient checks, tensor values
4. **Model Status** (10%): Initialization, layer info, complexity
5. **Deployment/Modal** (10%): S3 sync, volume operations, resource usage
6. **User-facing CLI** (5%): Commands, help text, status updates

## 🎯 Migration Strategy

### Phase 1: Infrastructure (Day 1 Morning)
1. Create central logging configuration module
2. Set up structured logging with rich handlers
3. Define log levels and categories
4. Create logging utilities and helpers

### Phase 2: Critical Path (Day 1 Afternoon)
1. Migrate `train/loop.py` (147 prints, 87 with flush)
2. Migrate `deploy/modal/app.py` (114 prints, 42 with flush)
3. Test training and deployment flows

### Phase 3: Data Pipeline (Day 2 Morning)
1. Migrate data loading/preprocessing files
2. Migrate cache utilities
3. Test data pipeline end-to-end

### Phase 4: Models & Utils (Day 2 Afternoon)
1. Migrate model files (already partially done)
2. Migrate debug utilities
3. Migrate remaining utility files

### Phase 5: CLI & Polish (Day 3)
1. Migrate CLI (special handling for user output)
2. Integration testing
3. Documentation update
4. Performance validation

## 🏗️ Proposed Architecture

### 1. Central Logging Configuration (`src/brain_brr/utils/logging_config.py`)
```python
import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
import os

# Log levels based on environment
LOG_LEVEL = os.getenv("BGB_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("BGB_LOG_FILE", None)
LOG_FORMAT = os.getenv("BGB_LOG_FORMAT", "rich")  # "rich", "simple", "json"

def setup_logging(
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = None,
    format_style: str = LOG_FORMAT,
    force_setup: bool = False
) -> None:
    """Configure logging for entire application."""

    # Only configure once unless forced
    if logging.getLogger().handlers and not force_setup:
        return

    # Clear existing handlers
    logging.getLogger().handlers = []

    # Console handler with rich formatting
    if format_style == "rich":
        console = Console(stderr=True, force_terminal=True)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
            )
        )

    # Set up root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
    )

    # File handler if specified
    if log_file or LOG_FILE:
        file_handler = logging.FileHandler(log_file or LOG_FILE)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)

    # Suppress noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)

# IMPORTANT: Do NOT auto-configure on import.
# Call setup_logging() explicitly from entrypoints (CLI/train/deploy).
```

### 2. Logging Categories & Conventions
```python
# Standard logger pattern for all modules
import logging
from src.brain_brr.utils.logging_config import setup_logging

# Get module logger
logger = logging.getLogger(__name__)

# Usage patterns:
# - logger.debug() - Detailed diagnostic info (only in debug mode)
# - logger.info() - General informational messages
# - logger.warning() - Warning messages (recoverable issues)
# - logger.error() - Error messages (failures)
# - logger.critical() - Critical failures (system-level issues)
```

### 3. Special Handlers

#### Training Progress Logger (optional, Phase 2)
```python
class TrainingLogger:
    """Specialized logger for training metrics with real-time updates."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.console = Console()
        self.progress = None

    def start_epoch(self, epoch: int, total_epochs: int):
        """Start epoch progress tracking."""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")

    def log_batch(self, batch_idx: int, loss: float, **metrics):
        """Log batch with metrics - replaces print(..., flush=True)."""
        # Use rich for pretty terminal output (refreshing less frequently)
        self.console.print(
            f"[cyan]Batch {batch_idx}[/cyan] | "
            f"Loss: {loss:.4f} | " +
            " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )

    def log_nan_detection(self, location: str, tensor_info: dict):
        """Specialized NaN logging with context."""
        self.logger.warning(
            f"NaN detected at {location}",
            extra={"tensor_info": tensor_info}
        )
```

#### CLI Output Handler (optional, Phase 3)
```python
class CLILogger:
    """User-facing CLI output - maintains rich formatting."""

    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    def status(self, message: str, style: str = "cyan"):
        """Status messages for user."""
        self.console.print(f"[{style}]{message}[/{style}]")
        self.logger.info(message)  # Also log for debugging

    def success(self, message: str):
        """Success messages."""
        self.console.print(f"[green]✓ {message}[/green]")
        self.logger.info(f"SUCCESS: {message}")

    def error(self, message: str):
        """Error messages."""
        self.console.print(f"[red]✗ {message}[/red]")
        self.logger.error(message)
```

## 🔧 Implementation Details

### Migration Rules
1. **Direct replacements**:
   - `print(f"[TAG] {msg}")` → `logger.info(f"[TAG] {msg}")`
   - `print(f"WARNING: {msg}")` → `logger.warning(msg)`
   - `print(f"ERROR: {msg}")` → `logger.error(msg)`

2. **Flush=True prints** → Replace with logger calls; console `StreamHandler` flushes per record. For high-frequency inner-loop messages, gate by `step % N == 0` and/or log level.

3. **Debug prints** → `logger.debug()` (only shown with BGB_LOG_LEVEL=DEBUG)

4. **User-facing CLI** → Keep `console.print()` for rich formatting; optionally mirror important messages with `logger.info()` for audit trails (avoid duplicate spam).

5. **Modal/deployment** → Structured logging with context

### Environment Variables
```bash
# New logging controls
BGB_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR|CRITICAL  # Default: INFO
BGB_LOG_FILE=/path/to/logfile.log                # Optional file output
BGB_LOG_FORMAT=rich|simple                       # Output format (JSON optional in Phase 2)
BGB_LOG_EVERY_N_STEPS=50                         # Gate per-batch logs (default 50)

# Existing debug controls (integrate with logging)
BGB_NAN_DEBUG=1        # Sets logger to DEBUG for NaN messages
BGB_DEBUG_FINITE=1     # Enables finite checks with DEBUG logging
```

### Backwards Compatibility
- Keep existing environment variables working
- Gradual migration: coexistence of print + logging initially
- Add `--verbose` and `--quiet` CLI flags

## 📋 Validation Criteria

### Success Metrics
- [ ] All 387 print statements migrated or consciously retained (47 CLI console.print kept)
- [ ] No loss of real-time monitoring capability
- [ ] Training progress remains visible
- [ ] CLI output maintains rich formatting
- [ ] Log files properly structured
- [ ] Performance impact < 1%
- [ ] All tests pass

### Testing Plan
1. Unit tests for logging configuration (levels, file handler, warnings capture)
2. Integration test for training loop (INFO emits epoch-level only by default; DEBUG emits per-batch when gated)
3. CLI output validation (unchanged `console.print` output via Click runner)
4. Modal deployment test (logs visible in Modal app logs; no excessive spam)
5. Log file format verification (rich vs simple)
6. Performance benchmarking (<1% overhead; `isEnabledFor` guards heavy formatting)

## 🔩 Change Map (exact entrypoints and hot spots)

- Configure logging from entrypoints only:
  - `src/brain_brr/cli/cli.py:592` main(): call `setup_logging(level=os.getenv("BGB_LOG_LEVEL","INFO"), force_setup=True)` before returning `cli(...)`.
  - `src/brain_brr/train/loop.py:1403` main(): call `setup_logging(...)` at top; also `logging.captureWarnings(True)`.
  - `deploy/modal/app.py` for each `@app.function` (5 functions total):
    - Line 147: `populate_cache()` - SSD cache population
    - Line 256: `clean_cache()` - Volume cleanup
    - Line 297: `test_mamba_cuda()` - CUDA kernel validation
    - Line 360: `train()` - Main training function
    - Line 650: `evaluate()` - Model evaluation
    First line in each function body should call `setup_logging(...)` (Modal streams stderr to logs).

- High-volume replacements (gate by N and/or DEBUG):
  - `src/brain_brr/train/loop.py` (147 prints, 87 with flush):
    - Sampler creation/progress: INFO for start/end; DEBUG every N windows.
    - Dataset stats banners: INFO once per phase.
    - Critical “FATAL/CRITICAL/SMOKE TEST” banners: WARNING/ERROR.
    - Per-batch metrics: DEBUG every `BGB_LOG_EVERY_N_STEPS`.
  - `deploy/modal/app.py` (114 prints; 42 with flush):
    - Resource/config banners: INFO.
    - Non-fatal count deviations: WARNING (still proceed).
    - One-time success/failure messages: INFO/ERROR.

- Medium-volume replacements:
  - `src/brain_brr/data/tusz_splits.py` (13): INFO for summaries, WARNING for violations.
  - `src/brain_brr/data/datasets.py` (11): INFO for cache creation, WARNING on fallbacks.
  - `src/brain_brr/train/wandb_integration.py` (7): INFO for init/success, WARNING/ERROR for failures.
  - `src/brain_brr/data/cache_utils.py` (6): INFO/WARNING.

- Low-volume replacements:
  - `src/brain_brr/models/tcn.py` (5 prints): Already has logger, migrate prints
  - `src/brain_brr/models/debug_utils.py` (5 prints, 2 flush): DEBUG level for all
  - `src/brain_brr/models/mamba.py` (2 prints): Already has logger, convert remaining
  - `src/brain_brr/models/gnn_pyg.py` (2 prints): INFO for complexity messages
  - `src/brain_brr/eval/metrics.py` (2 prints): INFO level

## 🧪 Test Impact & Adjustments

### Tests Using Output Capture
- **CLI tests** (`tests/unit/cli/*`): Use `CliRunner` and assert on `result.output`
  - NO CHANGES NEEDED - CLI keeps `console.print()` for user output
  - Tests: `test_cli_simple.py`, `test_cli_commands.py` continue working
- **Logging tests** (`tests/unit/models/test_interpolation.py:157-204`):
  - Already uses `caplog` fixture correctly
  - Sets level with `caplog.set_level(logging.WARNING)`
  - Asserts on `caplog.records`

### New Test Requirements
- Add test for `BGB_LOG_EVERY_N_STEPS` gating in `train/loop.py`
- Verify logging configuration from entrypoints
- Test log file creation when `BGB_LOG_FILE` is set
- Ensure performance tests run at INFO level by default

## ⚠️ Critical Integration Points

### W&B Integration Considerations
- W&B captures stdout/stderr by default → potential duplicate logs
- W&B has its own logging (wandb.log) → don't confuse with Python logging
- Solutions:
  1. Set W&B to not capture stdout: `wandb.init(settings=wandb.Settings(console="off"))`
  2. OR: Use WARNING level for W&B-related logs to reduce noise
  3. OR: Create separate logger for W&B with different handler

### Files Already Using Logging (Handle Carefully)
- `data/io.py:19` - Has logger, just remove prints
- `models/clamp_utils.py:12` - Has logger, ensure consistency
- `models/tcn.py:16` - Has logger, migrate 5 remaining prints
- `models/mamba.py:19` - Has logger, migrate 2 remaining prints
- `literature/pdf_to_markdown.py:42` - Has logger (not in main codebase)
- DON'T break existing logger usage patterns

### Smoke Test Mode Handling
- `BGB_SMOKE_TEST=1` affects verbosity expectations
- In smoke mode: More verbose logging acceptable (DEBUG level)
- Production mode: INFO level with gated inner loops
- Consider: `if env.smoke_test(): logger.setLevel(logging.DEBUG)`

### Environment Variable Completeness
All 25+ BGB_* variables from `src/brain_brr/utils/env.py`:
- **Debug/NaN**: BGB_NAN_DEBUG, BGB_DEBUG_FINITE, BGB_NAN_DEBUG_MAX
- **Sanitization**: BGB_SANITIZE_INPUTS, BGB_SANITIZE_GRADS, BGB_SKIP_OPT_STEP_ON_NAN
- **Safety**: BGB_SAFE_CLAMP, BGB_SAFE_CLAMP_MIN/MAX
- **Control**: BGB_SMOKE_TEST, BGB_LIMIT_FILES, BGB_DISABLE_TQDM, BGB_DISABLE_TB
- **Model**: SEIZURE_MAMBA_FORCE_FALLBACK, BGB_FORCE_TCN_EXT
- **Performance**: BGB_PERF_ALLOW_GPU, BGB_PERF_THREADS, BGB_PERF_TOLERANCE_FACTOR
- **Training**: BGB_MID_EPOCH_MINUTES, BGB_MID_EPOCH_KEEP
- **Other**: BGB_FORCE_MANIFEST_REBUILD, BGB_ANOMALY_DETECT

## 🧱 Design Guardrails (revisions to plan)

- No auto-setup on import. Only entrypoints configure logging.
- Prefer parameterized logging to avoid formatting cost when disabled:
  - `logger.debug("Loss: %.4f", loss)` instead of f-strings in inner loops.
- Gate noisy paths explicitly: `if logger.isEnabledFor(logging.DEBUG) and step % N == 0:`.
- Capture Python warnings into logging: `logging.captureWarnings(True)` in entrypoints.
- Suppress noisy third-party loggers at WARNING+ (torch/matplotlib/PIL already listed).
- JSON logs: defer to Phase 2 (optional). If needed, use a simple custom `JsonFormatter` or add `python-json-logger` as an optional extra.

## 🧰 Minimal Code Inserts (ready-to-paste snippets)

- `src/brain_brr/cli/cli.py:592`:
```python
from src.brain_brr.utils.logging_config import setup_logging

def main() -> int:
    setup_logging()
    return cli(standalone_mode=False) or 0
```

- `src/brain_brr/train/loop.py:1403`:
```python
import logging
from src.brain_brr.utils.logging_config import setup_logging

def main() -> None:
    setup_logging()
    logging.captureWarnings(True)
    # ... existing code ...
```

- `deploy/modal/app.py` (first line inside each @app.function):
```python
from src.brain_brr.utils.logging_config import setup_logging
setup_logging()
```

## 🚀 Rollout Plan

### Day 1: Infrastructure + Critical Path
- Morning: Create logging infrastructure
- Afternoon: Migrate train/loop.py and modal/app.py
- Evening: Test training run

### Day 2: Data + Models
- Morning: Migrate data pipeline
- Afternoon: Migrate model files
- Evening: Integration testing

### Day 3: CLI + Documentation
- Morning: Migrate CLI with special handling
- Afternoon: Update documentation
- Evening: Final validation

## 📝 Code Examples

### Before (current):
```python
# train/loop.py
print(f"[TRAIN] Epoch {epoch}/{num_epochs}", flush=True)
print(f"[TRAIN] Loss: {loss:.4f}", flush=True)
if math.isnan(loss):
    print("WARNING: NaN loss detected!", flush=True)
```

### After (migrated):
```python
# train/loop.py
from src.brain_brr.utils.logging_config import setup_logging
from src.brain_brr.utils.training_logger import TrainingLogger

setup_logging()
logger = logging.getLogger(__name__)
train_logger = TrainingLogger(logger)

train_logger.start_epoch(epoch, num_epochs)
train_logger.log_batch(batch_idx, loss=loss)
if math.isnan(loss):
    train_logger.log_nan_detection("loss_computation", {"loss": loss})
```

## 🎁 Benefits

### Immediate Benefits
1. **Structured logging**: Consistent format across codebase
2. **Log levels**: Control verbosity via environment
3. **File output**: Persistent logs for debugging
4. **Performance**: Async logging possible
5. **Rich formatting**: Better terminal output

### Long-term Benefits
1. **Production ready**: Proper logging for deployment
2. **Debugging**: Historical logs for issue analysis
3. **Monitoring**: Integration with log aggregation services
4. **Metrics**: Structured logs enable metrics extraction
5. **Compliance**: Audit trails for clinical deployment

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | HIGH | Benchmark before/after, use async handlers |
| Lost real-time visibility | HIGH | Ensure flush behavior maintained |
| Breaking changes | MEDIUM | Gradual migration, backwards compatibility |
| Complex debugging | LOW | Keep BGB_NAN_DEBUG working as-is |

## 📅 Timeline

- **Start Date**: TBD
- **Estimated Duration**: 2-3 days
- **Testing**: 1 additional day
- **Total Effort**: 3-4 days

## 🔄 Alternative Approaches Considered

1. **Minimal change**: Just add `logging.basicConfig()` and replace prints
   - Pros: Quick, simple
   - Cons: No structure, loses rich formatting
   - Decision: Rejected - need proper architecture

2. **External library** (loguru, structlog):
   - Pros: Feature-rich, structured logging
   - Cons: New dependency, learning curve
   - Decision: Rejected - stdlib is sufficient

3. **Keep prints + add logging**:
   - Pros: No breaking changes
   - Cons: Duplicate output, maintenance burden
   - Decision: Rejected - clean migration better

## ✅ Approval & Sign-off

- [ ] Technical Lead Review
- [ ] Testing Plan Approved
- [ ] Migration Schedule Confirmed
- [ ] Documentation Updated
- [ ] Team Notified

---

**Recommendation**: This is a necessary and valuable investment. The current print-based approach is not sustainable for production deployment. The proposed architecture balances simplicity with functionality, maintaining the developer experience while adding production-grade capabilities.

**Next Steps**:
1. Review and approve this plan
2. Create feature branch `feature/logging-migration`
3. Implement Phase 1 infrastructure
4. Begin systematic migration

## 📊 Final Statistics Summary

### Accurate Counts (Verified)
- **Total print() calls**: 387 (247 src + 140 deploy)
- **Prints with flush=True**: 147 total
- **Rich console.print()**: 47 in CLI (KEEP as-is)
- **Files with logging**: 5 existing
- **Environment variables**: 25+ BGB_* variables to integrate
- **Modal functions needing setup**: 5 functions
- **Tests using output capture**: ~10 CLI tests, 1 caplog test

### Migration Scope
- **Must migrate**: 340 print statements (387 - 47 CLI console.print)
- **High priority**: 261 prints (loop.py + app.py)
- **Already logging**: 5 files need print cleanup only
- **Test changes**: Minimal (CLI tests unchanged)

The 387 print statements represent significant technical debt. However, with proper gating (BGB_LOG_EVERY_N_STEPS) and level control, we can maintain performance while gaining production logging capabilities. This plan provides a precise, actionable path forward.
