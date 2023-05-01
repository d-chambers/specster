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

import pandas as pd

from specster.core.callout import run_command
from specster.core.misc import (
    SequenceDescriptor,
    copy_directory_contents,
    find_base_path,
    find_data_path,
    get_control_default_path,
    load_templates_from_directory,
    run_new_par,
)
from specster.core.models import SpecsterModel
from specster.core.output import BaseOutput
from specster.core.parse import (
    read_binaries_in_directory,
    write_ascii_waveforms,
    write_directory_binaries,
)
from specster.core.printer import console, program_render
from specster.core.stations import _maybe_use_station_file
from specster.exceptions import SpecFEMError


class BaseControl:
    """Base control class."""

    base_path: Path = None
    # True when the current state has been writen to disk.
    _writen: bool = False
    _read_only = False
    _template_path = None
    _spec_parameters = SpecsterModel
    _control_type: Optional[str] = None
    _each_source_path = "EACH_SOURCE"
    _coord_columns = ("x", "y", "z")

    sources = SequenceDescriptor("par.sources.sources")
    models = SequenceDescriptor("par.material_models.models")
    stations = SequenceDescriptor("par.receivers.stations")

    receiver_sets = SequenceDescriptor(
        "par.receivers.receiver_sets",
        set_functions=(_maybe_use_station_file,),
    )
    regions = SequenceDescriptor("par.internal_meshing.regions.regions")
    dt = SequenceDescriptor("par.dt")
    time_steps = SequenceDescriptor("par.nstep")

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = get_control_default_path(self._control_type)
            self._read_only = True  # don't overwrite base files
        self.base_path = find_base_path(Path(base_path))
        self.par = self._spec_parameters.from_file(self._data_path)
        self._writen = True

    # --- Properties for all control subclasses

    @property
    def _data_path(self):
        out = find_data_path(self.base_path)
        out.mkdir(exist_ok=True, parents=True)
        return out

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

    # --- Abstract methods which need to be implemented.

    @abc.abstractmethod
    def get_input_paths(self) -> dict[str, Path]:
        """
        Return a dict of input file paths.

        The names should match the template file names.
        """

    @property
    @abc.abstractmethod
    def output(self) -> BaseOutput:
        """Return an output object for working with simulation output."""

    @abc.abstractmethod
    def run(self, output_path=None) -> BaseOutput:
        """Run the simulation, optionally copy the output folder."""

    @abc.abstractmethod
    def run_each_source(self) -> BaseOutput:
        """Run the simulation separately for each source."""

    @abc.abstractmethod
    def get_source_df(self) -> pd.DataFrame:
        """Get a dataframe of sources."""

    @abc.abstractmethod
    def get_station_df(self) -> pd.DataFrame:
        """Get a dataframe of stations."""

    # --- General methods

    def copy(self, path: Optional[Path] = None, exclude=None) -> Self:
        """Copy control2D and specify a new path.

        Parameters
        ----------
        Path
            The Path to a new directory. If None, use a temp
            directory.
        exclude
            List of directories of files to exclude.
        """
        path = path or Path(tempfile.TemporaryDirectory().name)
        path = Path(path)
        assert not path.is_file(), "must pass a directory."
        new = copy.deepcopy(self)
        new._writen = False
        new._read_only = False
        new.base_path = path
        copy_directory_contents(
            self.base_path,
            new.base_path,
            exclude=exclude,
        )
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
        assert not self._read_only, "control is read only"
        if path is not None:
            self = self.copy(path)  # NOQA
        templates = load_templates_from_directory(self._template_path)
        paths = self.get_input_paths()
        assert set(set(templates)).issubset(paths)
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

    @cache
    def _load_templates(self) -> dict[str, Path]:
        """
        Load templates.
        """
        out = {}
        for p in self._template_path.glob("*"):
            out[p.name.lower()] = p
        return out

    def ensure_output_path_exists(self):
        """Ensure the output directory has been created."""
        self.output_path.mkdir(exist_ok=True, parents=True)
        return self.output_path

    @property
    def output_path(self) -> Path:
        """Get the output directory path, create it if it isn't there."""
        return self.base_path / "OUTPUT_FILES"

    @property
    def each_source_path(self) -> Path:
        """
        Get the directory to where a copy of the control for each source was
        created.
        """
        return self.base_path / self._each_source_path

    def clear_each_source_path(self):
        """
        Get the directory to where a copy of the control for each source was
        created.
        """
        path = self.base_path / self._each_source_path
        if path.exists():
            shutil.rmtree(path)

    def clear_outputs(self):
        """Remove output directory."""
        path = self.output_path
        if path.exists():
            shutil.rmtree(path)

    def _write_output_file(self, text, name, console=None):
        """Write text to the output directory"""
        out_path = self.output_path / name
        if not isinstance(text, str):
            text = "\n".join(text)
        # dont write empty file
        if text:
            with open(out_path, "w") as fi:
                fi.write(text)
            if console:
                console.print(f"writing {out_path}")

    def _run_spec_command(self, command: str, bin_path, supress=False):
        """Run a specfem command."""
        render = program_render(console, title=command, supress_output=supress)
        with render as (con, _):
            bin = bin_path / command
            assert bin.exists(), f"binary {bin} doesn't exist!"
            con.rule(
                f"[bold red]Running specfem command: {command} on "
                f"{self.base_path} with binary {bin}",
            )
            if not self._writen:
                self.write(overwrite=True)
                self._writen = True
            self.ensure_output_path_exists()
            out = run_command(str(bin), cwd=self.base_path, console=con)
            out["command"], out["path"] = command, self.base_path
            # write ouput
            self._write_output_file(out["stdout"], f"{command}_stdout.txt", con)
            self._write_output_file(out["stderr"], f"{command}_stderr.txt", con)
        # raise error if std error is not None
        if out["stderr"]:
            raise SpecFEMError(out["stderr"])
        return out

    def _copy_inputs_to_outputs(self):
        """Copy the input files to the output directory."""
        self.ensure_output_path_exists()
        output_path = self.output_path
        for _, input_file in self.get_input_paths().items():
            if not input_file.exists():
                continue
            shutil.copy2(input_file, output_path / input_file.name)

    def __str__(self):
        msg = f"{self.__class__.__name__} with basepath {self.base_path}"
        return msg

    __repr__ = __str__

    def __eq__(self, other):
        """Tests for equality"""
        if not isinstance(other, BaseControl):
            return False
        return self.par == other.par

    def prepare_fwi_forward(self) -> Self:
        """
        Prepare control structure for forward simulation in FWI workflow.
        """
        self.par.simulation_type = "1"
        self.par.save_forward = True
        self.par.mesh.setup_with_binary_database = "1"
        self.par.mesh.save_model = "binary"
        self.par.mesh.model = "default"
        self.par.adjoint_kernel.approximate_hess_kl = True
        self.write(overwrite=True)
        return self

    def prepare_fwi_adjoint(self) -> Self:
        """
        Prepare control structure for adjoint part of fwi.
        """
        self.setup_with_binary_database = "2"
        self.par.simulation_type = "3"
        self.par.save_forward = False
        self.par.mesh.model = "binary"
        # save ascii kernels because they are easier to read.
        # TODO look at using binary kernels in the future.
        self.par.adjoint_kernel.save_ascii_kernels = True
        self.par.mesh.save_model = "default"
        self._writen = False
        self.write(overwrite=True)
        return self

    def get_material_model_df(self, overwrite=False) -> pd.DataFrame:
        """
        Read the material model from disk.

        Output dataframe has index set to spatial coords (eg x,z).

        If overwrite == True, force regeneration of material model
        by running mesher and specfem.
        """
        model = read_binaries_in_directory(self._data_path)
        if len(model) and not overwrite:
            return model.set_index(list(self._coord_columns))
        with run_new_par(self, supress_output=True) as par:
            par.simulation_type = "1"
            par.save_forward = False
            par.mesh.setup_with_binary_database = "1"
            par.mesh.save_model = "binary"
            par.mesh.model = "default"
            par.nstep = 10  # limit number of steps so simulation is fast.
        model = read_binaries_in_directory(self._data_path)
        return model.set_index(list(self._coord_columns))

    def set_material_model_df(self, df):
        """
        Set the material model used by control.

        Also flips needed flags so the model will be used in next run.
        """
        # ensure cols are set to spatial coords.
        if set(df.columns) & set(self._coord_columns):
            df = df.set_index(list(self._coord_columns))
        # not index is still spatial coords here; they wont update.
        write_directory_binaries(df, self._data_path)
        self.par.mesh.setup_with_binary_database = "2"
        self.par.mesh.model = "binary"
        self.par.mesh.save_model = "default"
        self.write(overwrite=True)
        return self

    def write_adjoint_sources(self, st) -> Self:
        """Write the adjoint sources in stream to directory."""
        out_path = self.base_path / "SEM"  # path to save traces.
        out_path.mkdir(exist_ok=True, parents=True)
        for tr in st:
            stats = tr.stats
            name = f"{stats.network}.{stats.station}.{stats.channel}.adj"
            new_path = out_path / name
            write_ascii_waveforms(tr, new_path)
        return self
