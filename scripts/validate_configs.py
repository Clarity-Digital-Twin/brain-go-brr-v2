#!/usr/bin/env python3
"""Validate YAML configs match constants.py defaults.

Usage:
    python scripts/validate_configs.py
    make config-check
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_brr import constants


CRITICAL_CONSTANTS = {
    "hysteresis.tau_on": constants.HYSTERESIS_TAU_ON,
    "hysteresis.tau_off": constants.HYSTERESIS_TAU_OFF,
    "hysteresis.delta": constants.HYSTERESIS_DELTA,
    "focal.alpha": constants.FOCAL_ALPHA_PRODUCTION,
    "focal.gamma": constants.FOCAL_GAMMA_PRODUCTION,
    "data.sampling_rate": constants.SAMPLING_RATE,
    "data.n_channels": constants.N_CHANNELS,
    "data.window_size": constants.WINDOW_SIZE_SEC,
    "data.stride": constants.STRIDE_SIZE_SEC,
}


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def validate_config(config_path: Path) -> list[str]:
    """Validate one config file against constants.

    Returns list of error messages (empty if valid).
    """
    errors = []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for path, expected_value in CRITICAL_CONSTANTS.items():
        actual_value = get_nested_value(config, path)

        if actual_value is None:
            continue

        if actual_value != expected_value:
            errors.append(
                f"{config_path.name}: {path} = {actual_value}, "
                f"expected {expected_value} (from constants.py)"
            )

    return errors


def main() -> int:
    """Validate all YAML configs."""
    config_dir = Path(__file__).parent.parent / "configs"

    all_errors = []

    for config_path in config_dir.rglob("*.yaml"):
        if config_path.name.startswith("."):
            continue

        print(f"Checking {config_path.relative_to(config_dir)}...")
        errors = validate_config(config_path)
        all_errors.extend(errors)

    if all_errors:
        print("\n❌ CONFIG VALIDATION FAILED\n")
        for error in all_errors:
            print(f"  - {error}")
        print(
            f"\nFound {len(all_errors)} config/constants mismatches. "
            "Update configs to match constants.py or vice versa."
        )
        return 1

    print("\n✅ All configs match constants.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
