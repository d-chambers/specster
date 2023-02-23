"""
Configuration for specster.
"""
import os
from functools import cache
from pathlib import Path

import pytest

import specster

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


@cache
def get_data_directories():
    """Get a list of data directories in examples cases."""
    specster.settings.assert_specfem_available()
    example_path = specster.settings.spec_path / "EXAMPLES"
    out = tuple(x for x in example_path.rglob("DATA"))
    return out


@pytest.fixture(scope="session", params=get_data_directories())
def data_dir_path(request):
    """Fixture to iterate over data directory and return examples."""
    return Path(request.param)


@pytest.fixture(scope="session")
def par_file_path(data_dir_path):
    """Fixture to iterate over data directory and return examples."""
    par_path = data_dir_path / "Par_file"
    if not par_path.exists() or "not_ready_yet" in str(data_dir_path):
        pytest.skip(f"{data_dir_path} has not par file.")
    return par_path


@pytest.fixture()
def receiver_2d_line():
    """
    A snippet of a par file containing receiver lines (these are hard to parse)
    """
    return (TEST_DATA_PATH / "receiver_2d_lines").read_text()
