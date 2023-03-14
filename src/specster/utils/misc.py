"""
Misc small utilities.
"""
import copy
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
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
def load_templates_from_directory(path: Path) -> dict:
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
        return specster.settings.package_path / "d2" / "base_case_2d"
    else:
        msg = "other controls not yet supported"
        raise ValueError(msg)


class SequenceDescriptor:
    """A descriptor for returning/setting discriptors."""

    def __init__(self, attribute: str):
        self._attributes = attribute.split(".")

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        out = instance
        for attr in self._attributes:
            out = getattr(out, attr)
        return copy.deepcopy(out)

    def __set__(self, instance, value):
        base_attr = instance
        for attr in self._attributes[:-1]:
            base_attr = getattr(base_attr, attr)
        setattr(base_attr, self._attributes[-1], value)


class SizeOfDescriptor:
    """
    A descriptor which returns the size of a sequence attached to instance.
    """

    def __init__(self, sequence_attr):
        self._sequence_attr = sequence_attr

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        seq = getattr(instance, self._sequence_attr)
        return len(seq)

    def __set__(self, instance, value):
        pass  # allow passing so values can be parsed.
