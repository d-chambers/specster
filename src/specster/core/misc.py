"""
Misc small utilities.
"""
import copy
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import pooch
from jinja2 import Template
from pydantic import BaseModel

import specster
from specster.constants import special_dirs


def get_directory_path(base_path: Path, directory_name: str) -> Path:
    """Get the directory path, make it if it isn't there."""
    out = base_path / directory_name
    out.mkdir(exist_ok=True, parents=True)
    return out


def find_file_startswith(path: Path, startswith="Par_file"):
    """Try to find a file that starts with file_start in a directory."""
    if path.is_file() and startswith in path.name:
        return path
    parfiles = sorted(path.glob(f"{startswith}*"))
    if len(parfiles):
        return parfiles[0]
    msg = f"Unable to find {startswith} file in {path}"
    raise FileNotFoundError(msg)


def find_data_path(path):
    """Look for the data path."""
    if path.name.startswith("DATA"):
        return path
    return path / "DATA"


def find_base_path(path):
    """find the base path"""
    if path.name in special_dirs:
        return path.parent
    return path


def match_between(text, start, end="$", default=None):
    """
    Scan through text and find text between start/end string.

    Parameters
    ----------
    text
        The string to search
    start
        The starting string
    end
        The ending string, default matches on line ends.
    """

    regex = f"{start}(.*?){end}"
    out = re.search(regex, text, re.MULTILINE)
    if out is None and default is not None:
        return default
    assert out is not None, "Match returned nothing!"
    return out.group(1).replace("=", "").strip()


@cache
def load_templates_text_from_directory(path: Path) -> dict:
    """Load all templates in directory."""
    assert path.exists() and path.is_dir()

    out = {}
    for path in path.glob("*"):
        with path.open("r") as fi:
            name = path.name.lower().rsplit(".", 1)[0]
            out[name] = fi.read()
    return out


@cache
def load_templates_from_directory(path: Path) -> Dict[str, Template]:
    """Load all templates in directory."""
    text_dict = load_templates_text_from_directory(path)
    out = {i: Template(v) for i, v in text_dict.items()}
    return out


def assert_floats_nearly_equal(val1, val2, tolerance=0.0001):
    """Ensure floats are nearly equal"""
    if val1 != val2:
        out = (val2 - val1) / (np.mean([val2, val1]))
        assert np.abs(out) < tolerance


def assert_models_equal(model1, model2):
    """Walk the models and assert they are equal (helps find unequal parts)"""
    if hasattr(model1, "__fields__") and hasattr(model2, "__fields__"):
        f1, f2 = set(model1.__fields__), set(model2.__fields__)
        assert set(f1) == set(f2)
        # iterate each key, recursively apply model equals
        for key in f1:
            val1, val2 = getattr(model1, key), getattr(model2, key)
            if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                for sub_val1, sub_val2 in zip(val1, val2):
                    assert_models_equal(sub_val1, sub_val2)
            else:
                assert_models_equal(val1, val2)
    # special handling for floats
    elif isinstance(model1, float) and isinstance(model2, float):
        assert_floats_nearly_equal(model1, model2)
    # simply recurse for other models
    elif isinstance(model1, BaseModel) and isinstance(model2, BaseModel):
        assert_models_equal(model1, model2)
    else:  # anything else should be equal
        assert model1 == model2


def get_control_default_path(control: Literal["2D", "3D", None] = "2D") -> Path:
    """Get the path to the default control files."""
    if control == "2D":
        return specster.settings.package_path / "d2" / "data" / "base_case_2d"
    else:
        msg = "other controls not yet supported"
        raise ValueError(msg)


def write_model_data(self, key: Optional[str] = None):
    """Write the model data."""
    param_list = [self.get_formatted_str(x) for x in self.__fields__]
    return " ".join(param_list)


def copy_directory_contents(old, new, exclude=None):
    """
    Copy all contents of old directory to new directory.
    """
    exclude = [] if not exclude else exclude
    old, new = Path(old), Path(new)
    assert old.exists() and old.is_dir()
    for path in old.rglob("*"):
        has_exclude = [x in str(path) for x in exclude]
        if path.is_dir() or any(has_exclude):
            continue
        base = path.relative_to(old)
        new_path = new / base
        new_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path, new_path)


class SequenceDescriptor:
    """
    A descriptor for returning/setting nested values in par structure.

    Set_functions and get_functions will be called after setting and
    getting values. These are useful for updating related values.
    Both of these should take the instance as the first argument.
    """

    def __init__(self, attribute: str, set_functions=(), get_functions=()):
        self._attributes = attribute.split(".")
        self.set_functions = set_functions
        self.get_functions = get_functions

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        out = instance
        for attr in self._attributes:
            out = getattr(out, attr)
        for func in self.get_functions:
            func(instance, out)
        return out

    def __set__(self, instance, value):
        base_attr = instance
        for attr in self._attributes[:-1]:
            base_attr = getattr(base_attr, attr)
        setattr(base_attr, self._attributes[-1], value)
        for func in self.set_functions:
            func(instance, value)


def grid(x, y, z, resX=100, resY=100):
    """
    Converts 3 column data to matplotlib grid
    """
    # Can be found in ./utils/Visualization/plot_kernel.py
    from scipy.interpolate import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)

    # scipy version
    Z = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")

    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z


def cache_file_or_dir(path, name):
    """Create a persistent cache to a path.

    This is mainly useful for testing.
    """
    # delete old cache
    cache_loc = Path(pooch.os_cache("specster")) / name
    if cache_loc.exists() and cache_loc.is_dir():
        shutil.rmtree(cache_loc)
    shutil.copytree(path, cache_loc)


def load_cache(name) -> Union[Path, None]:
    """Get the path to a cache or raise not found Error."""
    cache_loc = Path(pooch.os_cache("specster")) / name
    if not cache_loc.exists():
        return None
    return cache_loc


def clear_cache():
    """Delete the entire specster cache."""
    cache_loc = Path(pooch.os_cache("specster"))
    shutil.rmtree(cache_loc)


@cache
def get_executor():
    """Return an executor for running things in parallel."""
    return ProcessPoolExecutor()


def call_it(call):
    """Dummy function to call thing passed in"""
    return call()


def parallel_call(funcs):
    """Call each function in parallel."""
    client = get_executor()
    futures = [client.submit(call_it, func) for func in funcs]
    wait(futures)
    exceptions = [x.exception() for x in futures]
    # assert not any(exceptions), "Exception raised in subprocess!"
    first_exception = [x for x in exceptions if x]
    if first_exception:
        raise first_exception[0]


@contextmanager
def run_new_par(control, supress_output=False):
    """Return a new parameter object which is overwritten then run
    control and switch back.
    """
    par_old = copy.deepcopy(control.par)
    new_par = copy.deepcopy(control.par)
    yield new_par
    control.par = new_par
    control.write(overwrite=True)
    try:
        control.run(supress_output=supress_output)
    except Exception as e:
        control.par = par_old
        control.write(overwrite=True)
        raise e
    control.par = par_old
    control.write(overwrite=True)
