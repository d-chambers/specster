"""
Output for 2D simulations.
"""

from __future__ import annotations

import re
import warnings
from functools import cache, cached_property
from pathlib import Path
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field

from specster.core.misc import match_between
from specster.core.models import SpecsterModel
from specster.core.output import BaseOutput
from specster.core.parse import read_ascii_kernels
from specster.core.plotting import plot_gll_historgrams, plot_kernels

# from .viz import plot_kernels, plot_single_kernel

KERNEL_TYPES = Literal["rhop_alpha_beta", "rho_kappa_mu"]


class GLLHistRow(SpecsterModel):
    """A row from the GLLs histograms."""

    bin_start: float
    bin_end: float
    elements: int
    percentage: float

    @classmethod
    def from_args(cls, args):
        """Init from args, assuming args are the same as field order."""
        fields = list(cls.model_fields)
        assert len(args) == len(fields)
        kwargs = {i: v for i, v in zip(fields, args)}
        return cls(**kwargs)


class SPECFEM2DStats(SpecsterModel):
    """
    Stats for specfem2d outputs.

    Collects key things from output file and tries to ensure the run
    was healthy.
    """

    mpi_slices: int | None = Field(default=None, description="Number of MPI slices")
    spec_duration: float = Field(description="Duration of specfem run in seconds")
    receiver_count: int = Field(description="Number of receivers")
    max_cfl: float = Field(description="Max CLF stability condition, should be < 0.5")
    elements: int = Field(description="Total number of elements")
    regular_elements: int = Field(
        description="Total number of regular (non-PML) elements"
    )
    pml_elements: int = Field(description="Total number of regular PML elements")
    acoustic_elements: int = Field(description="Total number of acoustic elements")
    elastic_elements: int = Field(description="Total number of elastic elements")
    max_grid_size: float = Field(description="max size of element")
    min_grid_size: float = Field(description="min size of element")
    min_gll_distance: float = Field(description="min distance between GLL points")
    av_gll_distance: float = Field(description="max distance between GLL points")
    max_frequency_resolved: float = Field(description="max resolvable frequency")
    max_time_step: float = Field(description="max suggested timestep.")
    max_source_freq: float = Field(
        description="Maximum suggested (Ricker) source frequency",
    )
    dt: float = Field(description="Current timestep")
    points_per_p_min: float = Field(
        description="minimum GLL points per highest p frequency"
    )
    points_per_p_max: float = Field(
        description="max GLL points per highest p frequency"
    )
    points_per_s_min: float = Field(
        -1, description="min GLL points per highest s frequency"
    )
    points_per_s_max: float = Field(
        -1, description="max GLL points per highest s frequency"
    )
    solid_gll_hist: list[GLLHistRow] = Field(
        description="histogram of min points per S wavelength in solid regions"
    )
    fluid_gll_hist: list[GLLHistRow] = Field(
        description="histogram of min points per P wavelength in fluid regions"
    )
    x_lims: tuple[float, float] | None = Field(
        default=None, description="min and max value along x dimension."
    )
    z_lims: tuple[float, float] | None = Field(
        default=None, description="min and max values along z dimension."
    )

    @classmethod
    def parse_output_files(cls, spec_path, mesh_path=None) -> Self:
        """Parse the stdout file into instance"""
        spath = Path(spec_path)
        # replace parens because I suck at regex
        spec_text = spath.read_text().replace(")", "").replace("(", "")
        spec_dict = cls._get_spec_data(spec_text)
        if mesh_path and Path(mesh_path).exists():
            mpath = Path(mesh_path)
            mtext = mpath.read_text().replace(")", "").replace("(", "")
            spec_dict.update(cls._get_mesh_dict(mtext))
        # TODO parse mesher outputs
        return cls(**spec_dict)

    @classmethod
    def _get_mesh_dict(cls, text):
        """Get the mesh dict params."""
        out = dict(
            x_lims=cls._get_minmax(text, "X"),
            z_lims=cls._get_minmax(text, "Z"),
        )
        return out

    @classmethod
    def _get_minmax(cls, text, comp):
        """Get the min/max components."""
        start = f"Min and max value of {comp} in the grid ="
        vals = match_between(text, start).split()
        return vals

    @classmethod
    def _get_spec_data(cls, txt):
        """Return a dict of specfem data."""
        try:  # dev branch needs this
            spec_duration = match_between(
                txt, "date and time of the system\nin seconds     =", "s"
            )
        except ValueError:  # 8.1.0 needs this
            spec_duration = match_between(txt, "time of the system :", "s")

        out = dict(
            mpi_slices=match_between(txt, "total of", "slices", 1),
            receiver_count=match_between(
                txt, "found a total of", "receivers", default=-1
            ),
            spec_duration=spec_duration,
            max_cfl=match_between(txt, r"must be below about 0.50 or so"),
            elements=match_between(txt, "number of elements:", default=-1),
            regular_elements=match_between(txt, "of which", "are regular elements"),
            pml_elements=match_between(txt, "and ", "are PML elements"),
            acoustic_elements=match_between(
                txt, "of acoustic elements           =", default=-1
            ),
            elastic_elements=match_between(
                txt, "of elastic/visco/poro elements =", default=-1
            ),
            max_grid_size=match_between(txt, "Max grid size =", default=-1),
            min_grid_size=match_between(txt, "Min grid size =", default=-1),
            min_gll_distance=match_between(txt, "Minimum GLL point distance  ="),
            av_gll_distance=match_between(txt, "Average GLL point distance  ="),
            max_frequency_resolved=match_between(
                txt, "Maximum frequency resolved  =", "Hz"
            ),
            max_time_step=match_between(txt, "Maximum suggested time step"),
            max_source_freq=match_between(
                txt, "Maximum suggested Ricker source frequency"
            ),
            dt=match_between(txt, "for DT :"),
            points_per_p_min=match_between(txt, "Nb pts / lambdaP_fmax min"),
            points_per_p_max=match_between(txt, "Nb pts / lambdaP_fmax max ="),
            points_per_s_min=match_between(
                txt, "Nb pts / lambdaS_fmax min =", default=np.nan
            ),
            points_per_s_max=match_between(
                txt, "Nb pts / lambdaS_fmax max =", default=np.nan
            ),
            solid_gll_hist=cls.parse_histogram(txt, "solid"),
            fluid_gll_hist=cls.parse_histogram(txt, "fluid"),
        )
        return out

    @classmethod
    def parse_histogram(cls, txt, hist_type: Literal["solid", "fluid"]):
        """Parse the histogram text for elastic or acoustic."""
        regex = f"(?s)wavelength in {hist_type} regions:(.*?)total percentage"
        match = re.search(regex, txt, flags=re.MULTILINE)
        if match is None:
            msg = f"unable to find histogram {hist_type}"
            warnings.warn(msg)
            return []
        # pull out lines to read into histogram data
        lines = [
            x.replace("-", "").replace("%", "").split()
            for x in match.group(1).split("\n")
            if x.endswith("%")
        ]
        return [GLLHistRow.from_args(x) for x in lines]


class OutPut2D(BaseOutput):
    """
    Output object for 2D simulations.
    """

    _required_files = ("xspecfem2D_stdout.txt",)

    def __init__(self, path, control):
        super().__init__(path, control)
        self.stats = SPECFEM2DStats.parse_output_files(
            spec_path=self.path / "xspecfem2D_stdout.txt",
            mesh_path=self.path / "xmeshfem2D_stdout.txt",
        )

    @cached_property
    def _xspec_stdout(self):
        """Read in the xpsec stdout."""
        return (self.path / "xspecfem2D_stdout.txt").read_text()

    def _validate_cfl(self):
        """Ensure"""

    @property
    def solid_gll_hist_df(self) -> pd.DataFrame:
        """Return a dataframe of the solid hist"""
        data = [x.model_dump() for x in self.stats.solid_gll_hist]
        return pd.DataFrame(data)

    @property
    def fluid_gll_hist_df(self) -> pd.DataFrame:
        """Return a dataframe of the liquid hist"""
        data = [x.model_dump() for x in self.stats.fluid_gll_hist]
        return pd.DataFrame(data)

    # def load_event_kernels(
    #         self,
    #         kernel_type: KERNEL_TYPES = "rhop_alpha_beta",
    # ) -> Dict[str, pd.DataFrame]:
    #     """Load kernels into a dict."""
    #
    #     out = {}
    #     names = ["x", "z"] + list(kernel_type.split("_"))
    #     glob = f"*{kernel_type}_kernel.dat"
    #     for path in self.path.glob(glob):
    #         name = path.name.split("_")[0]
    #         out[name] = pd.read_csv(
    #             path, delim_whitespace=True, names=names, header=None
    #         )
    #     return out

    def plot_gll_per_wavelength_histogram(self):
        """Plots GLL per wavelength."""
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        fluid = self.fluid_gll_hist_df
        plot_gll_historgrams(fluid, ax=axes[0], title="Fluid GLL Histogram")

        solid = self.solid_gll_hist_df
        plot_gll_historgrams(solid, ax=axes[1], title="Solid GLL Histogram")

        plt.tight_layout()
        return fig, axes

    def plot_kernel(self, kernel=None):
        """Make a plot of the material models, stations, and sources."""
        # material_df = self.get_material_model_df()
        df = self.load_kernel(kernel=kernel)
        fig, axes = plot_kernels(self, df, columns=kernel)

        return fig, axes

    @cache
    def load_kernel(
        self,
        kernel=None,
    ) -> pd.DataFrame:
        """Load a kernel into memory."""
        coords = list(self._control._coord_columns)
        out = read_ascii_kernels(self.path, kernel)
        if set(out.columns) & set(coords):
            out = out.set_index(coords)
        return out

    #
    # plot_kernels = plot_kernels
    # plot_single_kernel = plot_single_kernel
