"""
A module for controlling the output of simulations.
"""

from __future__ import annotations

import abc
from contextlib import suppress
from pathlib import Path

import obspy

from specster.core.models import SpecsterModel
from specster.core.parse import read_ascii_stream, read_generic_trace


class BaseOutput(abc.ABC):
    """
    Class to control the output of a simulation.
    """

    validators = []
    _required_files = ()
    _optional_files = ()
    stats: SpecsterModel
    _source_time_name = "plot_source_time_function.txt"

    def __init__(self, path, control):
        self.path = Path(path)
        self._control = control

    def get_waveforms(self) -> obspy.Stream:
        """Read all waveforms in the output."""
        return read_ascii_stream(self.path)

    def get_source_time_function(self) -> obspy.Stream:
        """Return the source time function as a stream."""
        path = self.path / self._source_time_name
        tr = read_generic_trace(path)
        return obspy.Stream([tr])

    def validate(self):
        """Run all validators (start with _validate)"""
        method_names = [x for x in dir(self) if x.startswith("_validate")]
        for method_name in method_names:
            getattr(self, method_name)()
        return self

    @property
    def lims(self) -> dict[str, tuple[float, float]]:
        """Return a dataframe of the liquid hist"""
        out = {}
        for char in "xyz":
            with suppress(AttributeError):
                out[char] = getattr(self.stats, f"{char}_lims")
        return out

    def _check_required_files_exist(self):
        """Ensure required files exist."""
        for file_name in self._required_files:
            assert (self.path / file_name).exists()

    def __str__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name} :: {self.path}"
        return msg

    __repr__ = __str__
