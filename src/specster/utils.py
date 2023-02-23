"""
Generic Utilities for specster.
"""
from typing import Union

from pydantic import BaseModel

from .constants import _SUB_VALUES


class SpecsterModel(BaseModel):
    """Abstract model in case we need to modify base behavior."""

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
