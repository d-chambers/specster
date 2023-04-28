"""
Configuration for specster 2D tests.
"""
import os
import copy
import shutil
from functools import cache
from pathlib import Path

import pytest

import specster
import specster.d2.io.parfile as pf
from specster.core.parse import read_ascii_kernels
from specster.core.misc import find_file_startswith
from specster.core.misc import cache_file_or_dir, load_cache

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


def pytest_collection_modifyitems(config, items):
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_mymarker = pytest.mark.skip(reason='slow not selected')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_mymarker)


@pytest.fixture(scope="session")
def test_data_path():
    """Return the test data path for the whole directory."""
    return TEST_DATA_PATH


@pytest.fixture(scope='session')
def kernel_2d_dir_path():
    """Return a path to the kernel directory."""
    return Path(__file__).parent / "test_2D" / "test_data" / 'kernels'

@pytest.fixture(scope='class')
def weights_kernel(kernel_2d_dir_path):
    return read_ascii_kernels(kernel_2d_dir_path, "weights")

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


@pytest.fixture(scope="session")
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
    station.xs, station.zs = 2200, 2200
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


@pytest.fixture(scope='session')
def control_2d_default_3_sources(control_2d_default, tmp_path_factory):
    """Tests for running control 2d with 3 sources in parallel."""
    cache_name = "control_2d_default_3_sources"
    if not load_cache(cache_name):
        control = control_2d_default.copy(tmp_path_factory.mktemp("3source"))
        # there are some issues with visco elastic models; just
        # set the value to a small vs
        control.models[1].Vs = control.models[1].Vp / 2
        sources = control.sources
        new1 = copy.deepcopy(sources[0])
        new1.xs = 1000
        new1.zs = 1000
        new2 = copy.deepcopy(sources[0])
        new2.xs = 2000
        new2.zs = 2000
        control.sources += [new1, new2]
        # ps images are huge; don't do them.
        control.par.visualizations.postscript.output_postscript_snapshot = False
        control.run_each_source()

        cache_file_or_dir(control.base_path, cache_name)
    else:
        control = specster.Control2d(load_cache(cache_name))
    return control


@pytest.fixture(scope="class")
def initial_control(control_2d_default_3_sources):
    """Get the initial control and generate waveforms."""
    cache_name = "initial_control_fwi"
    control_true = control_2d_default_3_sources
    if not load_cache(cache_name):
        control = control_true.copy()
        # make salt a bit slower and run.
        material = control.par.material_models.models[3]
        material.Vp *= 0.97
        material.Vs *= 0.97
        control.write()
        control.run_each_source()
        cache_file_or_dir(control.base_path, cache_name)
    else:
        control = specster.Control2d(load_cache(cache_name))
    return control


@pytest.fixture()
def receiver_2d_line():
    """
    A snippet of a par file containing receiver lines (these are hard to parse)
    """
    return (TEST_DATA_2D_PATH / "receiver_2d_lines").read_text()
