# Coding Standards

- Python 3.11+ with full type hints
- Ruff line length 100, 4-space indent
- Import order: stdlib → third-party → first-party (sorted)
- Follow patterns from neighboring files

## Logging Conventions

- Use `logger = logging.getLogger(__name__)` for module-level loggers
- Configure logging only from entrypoints via `setup_logging()` (never on module import)
- Use parameterized logging for performance: `logger.debug("Loss: %.4f", loss)` not f-strings in inner loops
- Gate high-frequency logs: `if logger.isEnabledFor(logging.DEBUG) and step % N == 0:`
- Log levels:
  - DEBUG: Detailed diagnostics, inner-loop metrics (gated by `BGB_LOG_EVERY_N_STEPS`)
  - INFO: General progress, epoch/phase boundaries
  - WARNING: Recoverable issues, unexpected but handled conditions
  - ERROR: Failures, exceptions
- Rich console.print() retained for CLI user-facing output only
