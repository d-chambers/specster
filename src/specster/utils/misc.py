"""
Misc small utilities.
"""
from functools import cache
from pathlib import Path

import numpy as np
from jinja2 import Template
from pydantic import BaseModel

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
    if isinstance(val1, float) and isinstance(val2, float):
        out = (val2 - val1) / (np.mean([val2, val1]))
        assert np.abs(out) < tolerance
    else:
        assert val1 == val2


def assert_models_equal(model1, model2):
    """Walk the models and assert they are equal (helps find unequal parts)"""
    if hasattr(model1, "__fields__") and hasattr(model2, "__fields__"):
        f1, f2 = set(model1.__fields__), set(model2.__fields__)
        assert set(f1) == set(f2)
        for key in f1:
            val1, val2 = getattr(model1, key), getattr(model2, key)
            if isinstance(val2, BaseModel) and isinstance(val1, BaseModel):
                assert_models_equal(val1, val2)
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                for sub_val1, sub_val2 in zip(val1, val2):
                    assert_models_equal(sub_val1, sub_val2)
            else:
                assert_floats_nearly_equal(model1, model2)
    else:
        assert_floats_nearly_equal(model1, model2)
