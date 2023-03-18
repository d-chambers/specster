"""
Output for 2D simulations.
"""
from functools import cached_property

from specster.core.output import BaseOutput


def parse_xspec2d_stdout(path):
    """Parse the output of xspec2d"""


class OutPut2D(BaseOutput):
    """
    Output object for 2D simulations.
    """

    _required_files = ("xspecfem2D_stdout.txt",)

    @cached_property
    def _xspec_stdout(self):
        """Read in the xpsec stdout."""
        return (self.path / "xspecfem2D_stdout.txt").read_text()

    def _validate_cfl(self):
        """Ensure"""
