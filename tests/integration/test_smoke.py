import sys


def test_smoke_environment() -> None:
    # Basic sanity checks for environment and Python version
    assert sys.version_info.major == 3
    assert isinstance(__name__, str)
