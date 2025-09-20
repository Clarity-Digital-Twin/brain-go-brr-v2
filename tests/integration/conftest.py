"""Mark integration tests that need isolation."""

import pytest


# Auto-mark all tests in this directory as integration
def pytest_collection_modifyitems(items):
    for item in items:
        # Mark memory-intensive tests as serial
        if "memory" in item.name.lower() or "deterministic" in item.name:
            item.add_marker(pytest.mark.serial)
