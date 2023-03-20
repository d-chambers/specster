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
from specster.core.misc import find_file_startswith

TEST_PATH = Path(__file__).absolute().parent

TEST_DATA_PATH = TEST_PATH / "test_data"

TEST_PATH_2D = Path(__file__).absolute().parent / "test_2D"

TEST_DATA_2D_PATH = TEST_PATH_2D / "test_data"

# Path to the example directory
if specster.settings.specfem2d_path is None:
    msg = "Cannot run 2D tests until specfem2d path is set!"
    raise ValueError(msg)

EXAMPLE_PATH = specster.settings.specfem2d_path / "EXAMPLES"

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
    path = specster.settings.specfem2d_path / "DATA"
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


@pytest.fixture(scope="class")
def control_2d_default() -> specster.Control2d:
    """Return a default control 2D"""
    return specster.Control2d()


@pytest.fixture(scope="session")
def modified_control(tmp_path_factory):
    """Create a control class, perform several modifications."""
    tmp_path = tmp_path_factory.mktemp("end_to_end")
    control = specster.Control2d().copy(tmp_path)
    # make simulation shorter
    control.time_steps = 600
    # first change model params
    mods = control.models
    mods[0].rho *= 1.01
    mods[0].Vp *= 1.01
    control.models = mods
    # Then remove all but 1 stations
    station = control.stations[0]
    station.xs, station.xz = 2200, 2200
    control.stations = [station]
    # Then add a source
    new_source = control.sources[0].copy()
    new_source.xs, new_source.zs = 2100, 2100
    control.sources = control.sources + [new_source]
    return control


@pytest.fixture(scope="session")
def modified_control_ran(modified_control):
    """Run the modified control"""
    modified_control.run().validate()
    return modified_control


@pytest.fixture()
def receiver_2d_line():
    """
    A snippet of a par file containing receiver lines (these are hard to parse)
    """
    return (TEST_DATA_2D_PATH / "receiver_2d_lines").read_text()
