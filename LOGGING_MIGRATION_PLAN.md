# Logging Migration Plan: From Print to Production

**Date**: September 28, 2025
**Status**: PLANNING
**Estimated Effort**: 2-3 days
**Priority**: HIGH - Critical for production deployment and debugging

## 📊 Current State Analysis

### Print Statement Statistics
- **Total print statements**: 387
  - `src/`: 247 occurrences across 11 files
  - `deploy/`: 140 occurrences across 3 files
- **Real-time monitoring prints**: 153 with `flush=True`
- **Existing logging**: 5 files using `logging.getLogger(__name__)`
- **No central logging configuration**

### File Breakdown
| File | Print Count | Priority | Category |
|------|------------|----------|----------|
| `src/brain_brr/train/loop.py` | 147 | CRITICAL | Training progress, metrics, NaN detection |
| `deploy/modal/app.py` | 114 | HIGH | Deployment status, cache operations |
| `deploy/modal/cleanup_volume.py` | 16 | MEDIUM | Volume maintenance |
| `src/brain_brr/data/tusz_splits.py` | 13 | HIGH | Data validation, patient splits |
| `src/brain_brr/data/datasets.py` | 11 | HIGH | Dataset loading, sampling |
| `deploy/modal/inspect_volume.py` | 10 | LOW | Diagnostic tool |
| `src/brain_brr/cli/cli.py` | 47 | HIGH | User-facing CLI output |
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
1. Migrate `train/loop.py` (147 prints)
2. Migrate `deploy/modal/app.py` (114 prints)
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

# Auto-setup on import for convenience
setup_logging()
```

### 2. Logging Categories & Conventions
```python
# Standard logger pattern for all modules
import logging
from src.brain_brr.utils.logging_config import setup_logging

# Ensure logging is configured
setup_logging()

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

#### Training Progress Logger
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
        # Use rich for pretty terminal output
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

#### CLI Output Handler
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

2. **Flush=True prints** → Use specialized handlers or `logger.info()` with console handler

3. **Debug prints** → `logger.debug()` (only shown with BGB_LOG_LEVEL=DEBUG)

4. **User-facing CLI** → Keep `console.print()` for rich formatting + logger.info()

5. **Modal/deployment** → Structured logging with context

### Environment Variables
```bash
# New logging controls
BGB_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR|CRITICAL  # Default: INFO
BGB_LOG_FILE=/path/to/logfile.log                # Optional file output
BGB_LOG_FORMAT=rich|simple|json                  # Output format

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
- [ ] All 387 print statements migrated
- [ ] No loss of real-time monitoring capability
- [ ] Training progress remains visible
- [ ] CLI output maintains rich formatting
- [ ] Log files properly structured
- [ ] Performance impact < 1%
- [ ] All tests pass

### Testing Plan
1. Unit tests for logging configuration
2. Integration test for training loop
3. CLI output validation
4. Modal deployment test
5. Log file format verification
6. Performance benchmarking

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

The 387 print statements represent technical debt that should be addressed before the next major release. This plan provides a clear path to production-ready logging while maintaining all current functionality.