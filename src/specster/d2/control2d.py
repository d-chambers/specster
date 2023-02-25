"""
Main control class for interacting with specfem.
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Optional, Self

import specster

from .io import SpecParameters2D


def _find_data_path(path):
    """Look for the data path."""
    if path.name.startswith("DATA"):
        return path
    elif (path / "DATA").exists():
        return path / "DATA"


class Control2d:
    """
    The main module for controlling specfrem
    """

    base_path: Path = None

    # True when the current state has been writen to disk.
    _writen: bool = False

    def __init__(self, base_path: Path, spec_bin_path: Optional[Path] = None):
        self.base_path = base_path
        self._data_path = _find_data_path(base_path)
        self._spec_bin_path = spec_bin_path or specster.settings.spec_bin_path
        self._par = SpecParameters2D.from_file(self._data_path)
        self._writen = True

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

    def write(self):
        """Write the control contents to disk."""
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

    def _load_templates(self) -> dict[str, Path]:
        """
        Load templates.
        """
        path = Path(__file__).parent / "templates"
        out = {}
        for p in path.glob("*"):
            out[p.name] = p
        return out

    def xmeshfem2d(self):
        """Run the 2D mesher."""

    def xspecfem2d(self):
        """Run 2D specfem."""
