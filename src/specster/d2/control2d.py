"""
Main control class for interacting with specfem.
"""
from __future__ import annotations

from pathlib import Path

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
        return OutPut2D(self.output_path)

    def run(self) -> OutPut2D:
        """Run the simulation."""
        # Determine if internal mesher should be run
        use_stations = self.par.receivers.use_existing_stations
        use_external = self.par.external_meshing.read_external_mesh
        if not (use_external and use_stations):
            self.xmeshfem2d()
        self.xspecfem2d()
        return self.output

    def xmeshfem2d(self, print_=True):
        """Run the 2D mesher."""
        return self._run_spec_command("xmeshfem2D")

    def xspecfem2d(self, print_=True):
        """Run 2D specfem."""
        return self._run_spec_command("xspecfem2D")
