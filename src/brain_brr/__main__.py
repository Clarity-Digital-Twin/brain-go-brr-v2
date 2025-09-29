"""Main module entry point for Brain-Go-Brr V3.

This allows running the package with `python -m brain_brr` without warnings.
"""


# Use relative import since we're inside the package
from .cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
