"""
Configuration for specster.
"""
import os
from pathlib import Path

TEST_PATH = Path(__file__).absolute().parent

TEST_DATA_PATH = TEST_PATH / "test_data"


def pytest_sessionstart(session):
    """
    Hook to run before any other tests.
    """
    # If running in CI make sure to turn off matplotlib.
    # If running in CI make sure to turn off matplotlib.
    if os.environ.get("CI", False):
        pass
        # TODO need to set logic to load/cache/compile specfem
