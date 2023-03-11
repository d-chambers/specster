"""
Control class.
"""
import abc
import copy
import shutil
import tempfile
from functools import cache
from pathlib import Path
from typing import Optional, Self

import obspy

import specster
from specster.utils.callout import run_command
from specster.utils.misc import (
    find_base_path,
    find_data_path,
    load_templates_from_directory,
)
from specster.utils.models import SpecsterModel
from specster.utils.waveforms import read_ascii_stream


class BaseControl:
    """Base control class."""

    base_path: Path = None
    # True when the current state has been writen to disk.
    _writen: bool = False
    _template_path = None
    _spec_parameters = SpecsterModel

    def __init__(self, base_path: Path, spec_bin_path: Optional[Path] = None):
        self.base_path = find_base_path(Path(base_path))
        self._spec_bin_path = Path(spec_bin_path or specster.settings.spec_bin_path)
        self.par = self._spec_parameters.from_file(self._data_path)
        self._writen = True

    @property
    def _data_path(self):
        out = find_data_path(self.base_path)
        out.mkdir(exist_ok=True, parents=True)
        return out

    @abc.abstractmethod
    def get_file_paths(self) -> dict[str, Path]:
        """
        Return a dict of important paths.

        The names should match the template file names.
        """

    @property
    def _par_file_path(self):
        return self._data_path / "Par_file"

    @property
    def _stations_path(self):
        return self._data_path / "STATIONS"

    @property
    def _source_path(self):
        return self._data_path / "SOURCE"

    @property
    def _interfaces_path(self):
        return self._data_path / "interfaces.dat"

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

    def write(self, path: Optional[Path] = None, overwrite: bool = False) -> Self:
        """
        Write the control contents to disk.

        Parameters
        ----------
        path
            An optional path to write files to. If None, use current
            base_path.
        overwrite
            Squash existing input files. If not overwrite and files
            already exist do nothing.

        Returns
        -------
        A new instance of Control with basepath updated.
        """
        if path is not None:
            self = self.copy(path)  # NOQA
        templates = load_templates_from_directory(self._template_path)
        paths = self.get_file_paths()
        assert set(paths).issubset(set(templates))
        disp = self.par.disp
        for name, template in templates.items():
            path = paths[name]
            self._render_template(template, disp, path, overwrite)
        return self

    def _render_template(self, temp, disp, path, overwrite=False):
        """Render the template."""
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(exist_ok=True, parents=True)
        text = temp.render(dis=disp)
        with path.open("w") as fi:
            fi.write(text)

    @property
    def empty_output(self):
        """Return True if the output directory is empty"""
        out = self.get_output_path()
        return len(list(out.glob("*"))) == 0

    @cache
    def _load_templates(self) -> dict[str, Path]:
        """
        Load templates.
        """
        out = {}
        for p in self._template_path.glob("*"):
            out[p.name.lower()] = p
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

    def __str__(self):
        msg = f"{self.__class__.__name__} with basepath {self.base_path}"
        return msg

    __repr__ = __str__

    def __eq__(self, other):
        """Tests for equality"""
        if not isinstance(other, BaseControl):
            return False
        return self.par == other.par
