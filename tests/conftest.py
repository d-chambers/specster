"""
Configuration for specster 2D tests.
"""
import os
import shutil
from functools import cache
from pathlib import Path

import pytest

import specster
import specster.d2.io.parfile as pf
from specster.utils.misc import find_file_startswith

TEST_PATH = Path(__file__).absolute().parent

TEST_DATA_PATH = TEST_PATH / "test_data"

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


@pytest.fixture(scope="session")
def test_data_path():
    """Return the test data path for the whole directory."""
    return TEST_DATA_PATH


@cache
def get_data_directories():
    """Get a list of data directories in examples cases."""

    out = tuple(x for x in EXAMPLE_PATH.rglob("DATA"))
    return out


@pytest.fixture(scope="session")
def default_data_path(tmp_path_factory) -> Path:
    """Return the path to the default data directory."""
    new = tmp_path_factory.mktemp("Default_DATA")
    path = specster.settings.spec_path / "DATA"
    shutil.copytree(path, new / "DATA")
    return new


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
def run_parameters_2d(par_dicts_2d) -> pf.SpecParameters2D:
    """Return dictionaries of parameters for 2D test cases."""
    return pf.SpecParameters2D.init_from_dict(par_dicts_2d)


@pytest.fixture(scope="class")
def control_2d(data_dir_path):
    """2D control instances."""
    try:
        find_file_startswith(data_dir_path, "Par_file")
    except FileNotFoundError:
        pytest.skip("Parfile doesn't exist")
    if "not_ready_yet" in str(data_dir_path):
        pytest.skip("not ready yet")
    spec = specster.Control2d(data_dir_path)
    return spec


@pytest.fixture(scope="session")
def control_2d_default(default_data_path) -> specster.Control2d:
    """Return a default control 2D"""
    return specster.Control2d(default_data_path)


@pytest.fixture()
def receiver_2d_line():
    """
    A snippet of a par file containing receiver lines (these are hard to parse)
    """
    return (TEST_DATA_2D_PATH / "receiver_2d_lines").read_text()
