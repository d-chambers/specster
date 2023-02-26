"""
Module for models.
"""
import copy
import shutil
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Optional, Self

import obspy
from pydantic import BaseModel

import specster
from specster.utils.callout import run_command
from specster.utils.waveforms import read_ascii_stream


class SpecsterModel(BaseModel):
    """Abstract model in case we need to modify base behavior."""

    _key_space = 31
    _value_space = 15
    _key_val_deliminator = " = "

    class Config:
        """Configuration for models."""

        validate_assignment = True  # validators run on assignment
        keep_untouched = (cached_property,)

    @classmethod
    def read_line(cls, params):
        """Read params from a sequence into pydantic model."""
        if isinstance(params, str):
            params = params.split()
        field_names = list(cls.__fields__)
        assert len(params) == len(field_names), "names should match args"
        input_dict = {i: v for i, v in zip(field_names, params)}
        return cls(**input_dict)

    @cached_property
    def disp(self):
        """Return a displayer for nicely rendering contents."""
        from specster.utils.render import Displayer

        return Displayer(self)

    def _write_data(self, key: str):
        """Write the data contained in key to a string."""

        value = getattr(self, key)
        # handle special types that need formatting
        field = self.__fields__.get(key, None)
        formatter_dict = self._parser_dict
        if field and field.type_ in formatter_dict:
            value = formatter_dict[field.type_](value)
        # handles recursive case
        if hasattr(value, "_write_data"):
            out = value._write_data(key)
        else:
            padded_key = key.ljust(self._key_space, " ")
            str_value = str(value).ljust(self._value_space, " ")
            out = padded_key + self._key_val_deliminator + str_value
        return out

    @cached_property
    def _parser_dict(self):
        """Return the dict used for parsing."""
        from specster.utils.render import format_bool, number_to_spec_str

        out = {bool: format_bool, SpecFloat: number_to_spec_str}
        return out


class AbstractParameterModel(SpecsterModel):
    """Abstract class for defining specfem parameter models."""

    @classmethod
    def init_from_dict(cls, data_dict):
        """Init class, and subclasses, from a dict of values"""
        my_fields = set(cls.__fields__)
        nested_models = {
            k: v.type_
            for k, v in cls.__fields__.items()
            if hasattr(v.type_, "init_from_dict")
            # if the key is already the right type we skip it
            and not isinstance(data_dict.get(k), v.type_)
        }
        # get inputs for this model
        needed_inputs = {k: v for k, v in data_dict.items() if k in my_fields}
        # add nested models
        for field_name, model in nested_models.items():
            needed_inputs[field_name] = model.init_from_dict(data_dict)
        return cls(**needed_inputs)


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


class BaseControl:
    """Base control class."""

    base_path: Path = None
    # True when the current state has been writen to disk.
    _writen: bool = False
    _template_path = None
    _spec_parameters = SpecsterModel

    def __init__(self, base_path: Path, spec_bin_path: Optional[Path] = None):
        base_path = Path(base_path)
        self.base_path = base_path
        self._data_path = self._find_data_path(base_path)
        self._spec_bin_path = Path(spec_bin_path or specster.settings.spec_bin_path)
        self._par = self._spec_parameters.from_file(self._data_path)
        self._writen = True

    @staticmethod
    def _find_data_path(path):
        """Look for the data path."""
        if path.name.startswith("DATA"):
            return path
        elif (path / "DATA").exists():
            return path / "DATA"

    def copy(self, path: Optional[Path] = None) -> Self:
        """Copy control2D and specify a new path.

        Parameters
        ----------
        Path
            The Path to a new directory. If None use a temp
            directory.
        """
        path = path or Path(tempfile.TemporaryDirectory().name)
        assert not path.is_file(), "must pass a directory."
        new = copy.deepcopy(self)
        new._writen = False
        new.base_path = path
        return new

    def write(self, overwrite=False):
        """
        Write the control contents to disk.

        Parameters
        ----------
        overwrite
            Squash existing input files. If not overwrite and files
            already exist do nothing.
        """
        # first ensure directory and output files exist
        # data_path = get_directory_path(self.base_path, 'DATA')
        # get_directory_path(self.base_path, 'OUTPUT_FILES')
        # disp = self._par.disp
        # for name, path in self._load_templates().items():
        #     temp = get_template(path.parent, name)
        #     out = temp.render(dis=disp)
        #     if 'Par' in name:
        #         breakpoint()
        # self._par.write_data(data_path)
        # self._writen = True

    @property
    def empty_output(self):
        """Return True if the output directory is empty"""
        out = self.get_output_path()
        return len(list(out.glob("*"))) == 0

    def _load_templates(self) -> dict[str, Path]:
        """
        Load templates.
        """
        out = {}
        for p in self._template_path.glob("*"):
            out[p.name] = p
        return out

    def get_output_path(self):
        """Get the output directory path, create it if it isn't there."""
        expected = self.base_path / "OUTPUT_FILES"
        expected.mkdir(exist_ok=True, parents=True)
        return expected

    def clear_outputs(self):
        """Remove output directory."""
        path = self.get_output_path()
        shutil.rmtree(path)

    def _write_output_file(self, text, name):
        """Write text to the output directory"""
        out_path = self.get_output_path() / name
        if not isinstance(text, str):
            text = "\n".join(text)
        # dont write empty file
        if text:
            with open(out_path, "w") as fi:
                fi.write(text)

    def _run_spec_command(self, command: str, print_=True):
        """Run a specfem command."""
        self.get_output_path()
        bin = self._spec_bin_path / command
        assert bin.exists()
        out = run_command(str(bin), cwd=self.base_path, print_=print_)
        # write ouput
        if print_:
            self._write_output_file(out["stdout"], f"{command}_stdout.txt")
            self._write_output_file(out["stderr"], f"{command}_stderr.txt")
        return out

    def get_waveforms(self) -> obspy.Stream:
        """Read all waveforms in the output."""
        return read_ascii_stream(self.get_output_path())
