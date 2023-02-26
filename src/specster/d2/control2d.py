"""
Main control class for interacting with specfem.
"""
from __future__ import annotations

from pathlib import Path

from specster.utils.models import BaseControl

from .io import SpecParameters2D


class Control2d(BaseControl):
    """
    The main module for controlling Specfem 2D.
    """

    _spec_parameters = SpecParameters2D
    _template_path = Path(__file__).parent / "templates"

    def xmeshfem2d(self, print_=True):
        """Run the 2D mesher."""
        return self._run_spec_command("xmeshfem2D", print_=print_)

    def xspecfem2d(self, print_=True):
        """Run 2D specfem."""
        return self._run_spec_command("xspecfem2D", print_=print_)
