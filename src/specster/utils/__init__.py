"""
Generic Utilities for specster.
"""
from functools import cache
from pathlib import Path
from typing import Union

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from specster.constants import _SUB_VALUES


class SpecsterModel(BaseModel):
    """Abstract model in case we need to modify base behavior."""

    _key_space = 31
    _value_space = 15
    _key_val_deliminator = " = "

    @classmethod
    def read_line(cls, params):
        """Read params from a sequence into pydantic model."""
        if isinstance(params, str):
            params = params.split()
        field_names = list(cls.__fields__)
        assert len(params) == len(field_names), "names should match args"
        input_dict = {i: v for i, v in zip(field_names, params)}
        return cls(**input_dict)

    class Config:
        """Configuration for models."""

        validate_assignment = True  # validators run on assignment

    def _write_data(self, key: str):
        """Write the data contained in key to a string."""
        value = getattr(self, key)
        # handle special types that need formatting
        field = self.__fields__.get(key, None)
        if field and field.type_ in _FORMATTERS:
            value = _FORMATTERS[field.type_](value)
        # handles recursive case
        if hasattr(value, "_write_data"):
            out = value._write_data(key)
        else:
            padded_key = key.ljust(self._key_space, " ")
            str_value = str(value).ljust(self._value_space, " ")
            out = padded_key + self._key_val_deliminator + str_value
        return out

    @property
    def disp(self):
        """Return a displayer for nicely rendering contents."""
        return Displayer(self)


class SimpleValidator:
    """
    A custom class for getting simple validation behavior in pydantic.

    Subclass, then define function to be used as validator. func
    """

    @classmethod
    def func(cls, value):
        """A method to overwrite with custom validation."""
        return value

    @classmethod
    def __get_validators__(cls):
        """Hook used by pydantic."""
        yield cls.validate

    @classmethod
    def validate(cls, validator):
        """Simply call func."""
        return cls.func(validator)


class SpecFloat(float, SimpleValidator):
    """Validator to convert specfem float str to python float"""

    @staticmethod
    def func(value):
        """Remove silly d, cast to float"""
        if "d" in value:
            value = value.replace("d", "e")
        return float(value)

    @staticmethod
    def format_value(value):
        """Format the value back to d rather than e"""
        fmt_str = f"{value:e}".replace("e", "d")
        return fmt_str


class Displayer:
    """
    A class to produce outputs from the various parameter classes.

    This is essentially just a convinence class for accessing/writing
    string output of various parameters via getattr.
    """

    def __init__(self, model: SpecsterModel):
        """Init new display for model.

        Parameters
        ----------
        model
            The model which will get parameters from.
        """
        self._model = model

    def __getattr__(self, item):
        key = item.lower()
        # hand case where key is another spec class, need to get new disp
        value = getattr(self._model, key)
        if hasattr(value, "disp"):
            return value.disp
        out = self._model._write_data(key)
        return out


def number_to_spec_str(value: Union[int, float]) -> str:
    """
    Write float to string.

    This uses specfem's d notation, eg (2700.d0) means (2700e0)
    """
    # handle "special cases" which are special enough to break the rules
    if value == 0:
        return "0"
    elif value == 9999:
        return "9999"
    e_str = f"{value:.04e}"
    exp = e_str.split("e")[1]
    num = float(e_str.split("e")[0])
    return f"{num}d{int(exp)}"


def dict_to_description(param_name, somedict):
    """Convert a dict of enums to a description string."""
    param_str = "".join([f"{i}: {v}\n" for i, v in somedict.items()])
    out = f"{param_name} supports several params including:\n" f"{param_str}"
    return out


def extract_parline_key_value(line):
    """Extract key/value pairs from a single line of the par file."""
    key_value = line.split("=")
    key = key_value[0].strip().lower()
    value = key_value[1].split("#")[0].strip()
    return key, _SUB_VALUES.get(value, value)


def iter_file_lines(path, ignore="#"):
    """Read lines of a file, dont include comment lines."""
    with open(path, "r") as fi:
        for line in fi.readlines():
            stripped = line.strip()
            if stripped.startswith(ignore) or not stripped:
                continue
            yield line


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


def format_bool(bool_like):
    """Format boolean value to specfem style."""
    return ".true." if bool_like else ".false."


_FORMATTERS = {SpecFloat: SpecFloat.format_value, bool: format_bool}


@cache
def get_env(template_path):
    """Get the template environment."""
    template_path = Path(template_path)
    env = Environment(loader=FileSystemLoader(template_path))
    return env


@cache
def get_template(template_path, name):
    """Get the template for rendering tables."""
    env = get_env(template_path)
    template = env.get_template(name)
    return template
