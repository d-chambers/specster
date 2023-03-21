"""
Main control class for interacting with specfem.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Self

import pandas as pd

import specster
from specster.core.misc import copy_directory_contents
from specster.core.stations import read_stations

from ..core.control import BaseControl
from .io import SpecParameters2D
from .output2d import OutPut2D


class Control2d(BaseControl):
    """
    The main module for controlling Specfem 2D.
    """

    _spec_parameters = SpecParameters2D
    _template_path = Path(__file__).parent / "templates"
    _control_type = "2D"

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

    def run(self, output_path: Optional[Path] = None) -> OutPut2D:
        """Run the simulation."""
        # Determine if internal mesher should be run
        use_stations = self.par.receivers.use_existing_stations
        use_external = self.par.external_meshing.read_external_mesh
        if not (use_external and use_stations):
            self.xmeshfem2d()
        self.xspecfem2d()
        self._copy_inputs_to_outputs()
        default_output = self.output_path
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copytree(default_output, output_path)
        return OutPut2D(output_path or default_output, self)

    def prepare_fwi_forward(self) -> Self:
        """Prepare for forward run in FWI."""
        self = super().prepare_fwi_forward()
        # Not sure why but other examples set number of control els to 9;
        # just follow suit here. May need to change for 3D.
        self.par.mesh.ngnod = "9"
        return self

    def xmeshfem2d(self):
        """Run the 2D mesher."""
        bin_path = specster.settings.get_specfem2d_binary_path()
        return self._run_spec_command("xmeshfem2D", bin_path)

    def xspecfem2d(self):
        """Run 2D specfem."""
        bin_path = specster.settings.get_specfem2d_binary_path()
        return self._run_spec_command("xspecfem2D", bin_path)

    def _read_stations(self):
        """
        Return a list of station objects.

        May be different from internal data if receiver sets were used.
        """
        assert self._stations_path.exists(), "no stations to read!"
        return read_stations(True, self._stations_path)

    def get_source_df(self):
        """Get a dataframe of sources."""
        data = [x.dict() for x in self.sources]
        return pd.DataFrame(data)

    def get_station_df(self):
        """Get a dataframe of stations."""
        data = [x.dict() for x in self._read_stations()]
        return pd.DataFrame(data)


def load_2d_example(name, new_path=None) -> Control2d:
    """
    Load an example from the 2D specfem directory.

    Parameters
    ----------
    name
        Example name, should be a directory in specfem2d/Examples.
    new_path
        The new path to copy the control files to (so original examples
        aren't modified). If none, use a temp path.
    """
    base_path = specster.settings.specfem2d_path / "EXAMPLES" / name
    if not base_path.exists():
        msg = f"example with name {name} doesn't exist in {(base_path.parent)}"
        raise NotADirectoryError(msg)
    # copy all files in old base_path, not just standard files in case exteranl
    # files are used (e.g., named interfaces)
    control = Control2d(base_path).copy(new_path)
    copy_directory_contents(base_path, control.base_path)
    return control
