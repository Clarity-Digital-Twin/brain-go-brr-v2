"""CLI entrypoint for SzCORE Docker submissions.

Usage:
  python -m src.brain_brr.szcore /data/input.edf /output/output.tsv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .infer import run_szcore


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(argv) != 3:
        print("Usage: python -m src.brain_brr.szcore <input.edf> <output.tsv>")
        return 2

    input_edf = Path(argv[1])
    output_tsv = Path(argv[2])
    run_szcore(input_edf, output_tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

