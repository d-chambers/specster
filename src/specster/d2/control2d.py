"""
Main control class for interacting with specfem.
"""
from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path
from typing import List, Optional, Self

import matplotlib.pyplot as plt
import pandas as pd

import specster
from specster.core.misc import copy_directory_contents, parallel_call
from specster.core.plotting import plot_gll_data
from specster.core.stations import read_stations

from ..core.control import BaseControl
from .io import SpecParameters2D
from .output2d import OutPut2D


def _copy_set_source_run(control, source_index, base_path):
    """Process for forking for running each source in parallel."""
    base_path = Path(base_path)
    out_path = base_path / f"{int(source_index):06}"
    new = control.copy(out_path, exclude=[base_path.name])
    # ensure only one source is set.
    new.par.sources.sources = [new.par.sources.sources[source_index]]
    new.write(overwrite=True)
    new.run(supress_output=True)
    return out_path


class Control2d(BaseControl):
    """
    The main module for controlling Specfem 2D.
    """

    _spec_parameters = SpecParameters2D
    _template_path = Path(__file__).parent / "templates"
    _control_type = "2D"
    _coord_columns = ("z", "x")

    def get_input_paths(self) -> dict[str, Path]:
        """
        Return a dict of input file paths.

        The names should match the template file names.
        """
        out = dict(
            par_file=self._data_path / "Par_file",
            stations=self._data_path / "STATIONS",
            source=self._data_path / "SOURCE",
            interfaces=self._data_path / "interfaces.dat",
        )
        return out

    @property
    def output(self) -> OutPut2D:
        """return the output of the control."""
        return OutPut2D(self.output_path, self)

    @property
    def each_source_output(self) -> List[OutPut2D]:
        """Return an output for each source."""
        sorted_event_paths = sorted(x for x in self.each_source_path.iterdir())
        return [OutPut2D(x / "OUTPUT_FILES", self) for x in sorted_event_paths]

    def run(self, output_path: Optional[Path] = None, supress_output=False) -> OutPut2D:
        """Run the simulation."""
        # Determine if internal mesher should be run
        use_stations = self.par.receivers.use_existing_stations
        use_external = self.par.external_meshing.read_external_mesh
        if not (use_external and use_stations):
            self.xmeshfem2d(supress_output=supress_output)
        self.xspecfem2d(supress_output=supress_output)
        self._copy_inputs_to_outputs()
        default_output = self.output_path
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copytree(default_output, output_path)
        return OutPut2D(output_path or default_output, self)

    def run_each_source(self) -> Self:
        """
        Run the simulation separately for each source.

        This is done to prepare for an FWI workflow, and as such,
        some parameters will be set.
        """
        self._prep_many_source_run()
        sources = self.par.sources.sources
        path = self._each_source_path
        base_path = self.base_path / path
        callables = [
            partial(
                _copy_set_source_run, control=self, source_index=i, base_path=base_path
            )
            for i in range(len(sources))
        ]
        parallel_call(callables)
        return self

    def prepare_fwi_forward(self, *args, **kwargs) -> Self:
        """Prepare for forward run in FWI."""
        self = super().prepare_fwi_forward(*args, **kwargs)
        # Not sure why but other examples set number of control els to 9;
        # just follow suit here. May need to change for 3D.
        self.par.mesh.ngnod = "9"
        return self

    def xmeshfem2d(self, supress_output=False):
        """Run the 2D mesher."""
        bin_path = specster.settings.get_specfem2d_binary_path()
        return self._run_spec_command("xmeshfem2D", bin_path, supress=supress_output)

    def xspecfem2d(self, supress_output=False):
        """Run 2D specfem."""
        bin_path = specster.settings.get_specfem2d_binary_path()
        return self._run_spec_command("xspecfem2D", bin_path, supress=supress_output)

    def _read_stations(self):
        """
        Return a list of station objects.

        May be different from internal data if receiver sets were used.
        """
        assert self._stations_path.exists(), "no stations to read!"
        return read_stations(True, self._stations_path)

    def _prep_many_source_run(self):
        """
        Makes sure the model is ready for a many-source run.

        Effectively, we need the binary models to be produce and subsequent
        models to read the material models from binaries so they can be
        updated.
        """
        # if len(read_binaries_in_directory(self._data_path)):
        #     return self
        self.prepare_fwi_forward()
        # since we just need the material models no need run whole thing.
        nstep_old = self.par.nstep
        self.par.nstep = 10  # we just need the velocity model to
        self.write(overwrite=True)
        self.run()  # should be fast
        # reset
        self.par.nstep = nstep_old
        self.write(overwrite=True)
        # self.clear_outputs()
        return self

    def get_source_df(self):
        """Get a dataframe of sources."""
        data = [x.dict() for x in self.sources]
        return pd.DataFrame(data)

    def get_station_df(self):
        """Get a dataframe of stations."""
        data = [x.dict() for x in self._read_stations()]
        return pd.DataFrame(data)

    def plot_geometry(self, kernel=None, overwrite=False):
        """Make a plot of the material models, stations, and sources."""
        material_df = self.get_material_model_df(overwrite=overwrite)
        fig, axes = plot_gll_data(material_df, alpha=0.5, kernel=kernel)
        station_df = self.get_station_df()
        source_df = self.get_source_df()
        # need to plot
        for ax in axes:
            ax.plot(station_df["xs"], station_df["zs"], "v", color="k")
            ax.plot(source_df["xs"], source_df["zs"], "*", color="r")
        plt.tight_layout()
        return fig, axes


def load_2d_example(name, new_path=None, base_path=None) -> Control2d:
    """
    Load an example from specsters 2d data directory, or, if it doesn't
    exist, specfem's example directory.

    Parameters
    ----------
    name
        Example name, should be a directory in specster/d2/data or
        specfem2d/Examples.
    new_path
        The new path to copy the control files to (so original examples
        aren't modified). If none, use a temp path.
    """
    spec_path = specster.settings.package_path / "d2" / "data" / name
    base_path = specster.settings.specfem2d_path / "EXAMPLES" / name
    path_exists = [x for x in [spec_path, base_path] if x.exists()]
    if not path_exists:
        msg = f"example with name {name} doesn't exist in {(base_path.parent)}"
        raise NotADirectoryError(msg)
    base_path = path_exists[0]
    # copy all files in old base_path, not just standard files in case exteranl
    # files are used (e.g., named interfaces)
    control = Control2d(base_path).copy(new_path)
    copy_directory_contents(base_path, control.base_path)
    return control
