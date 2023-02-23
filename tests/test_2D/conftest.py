"""
Configuration for specster 2D tests.
"""
import os
from functools import cache
from pathlib import Path

import pytest

import specster
import specster.d2.io.parfile as pf

TEST_PATH_2D = Path(__file__).absolute().parent

TEST_DATA_2D_PATH = TEST_PATH_2D / "test_data"

specster.settings.assert_specfem_available()

# Path to the example directory
EXAMPLE_PATH = specster.settings.spec_path / "EXAMPLES"

PAR_FILES = list(EXAMPLE_PATH.rglob("Par_file*"))

SOURCE_FILES = list(EXAMPLE_PATH.rglob("SOURCE*"))


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

    out = tuple(x for x in EXAMPLE_PATH.rglob("DATA"))
    return out


@pytest.fixture(scope="session", params=get_data_directories())
def data_dir_path(request):
    """Fixture to iterate over data directory and return examples."""
    return Path(request.param)


@pytest.fixture(scope="session", params=PAR_FILES)
def par_file_path(request):
    """Fixture to iterate over data directory and return examples."""
    # these strings are used to match on bad (malformed) files
    par_path = request.param
    path_str = str(par_path)
    not_ready = "not_ready_yet" in path_str
    rec_checker = par_path.name.endswith("rec_checker")
    doesnt_exist = not par_path.exists()
    not_data = "DATA" not in path_str

    if any([not_ready, doesnt_exist, not_data, rec_checker]):
        pytest.skip(f"{data_dir_path} has not par file.")
    return par_path


@pytest.fixture(scope="class")
def par_dicts_2d(par_file_path):
    """Return dictionaries of parameters for 2D test cases."""
    return pf.parse_parfile(par_file_path)


@pytest.fixture(scope="class")
def run_parameters_2d(par_dicts_2d) -> pf.RunParameters:
    """Return dictionaries of parameters for 2D test cases."""
    return pf.RunParameters.init_from_dict(par_dicts_2d)


@pytest.fixture()
def receiver_2d_line():
    """
    A snippet of a par file containing receiver lines (these are hard to parse)
    """
    return (TEST_DATA_2D_PATH / "receiver_2d_lines").read_text()