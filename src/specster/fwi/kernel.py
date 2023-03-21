"""
2D Kernel with SPECFEM2D
Based on a notebook by Andrea R.
Utility functions written by Ridvan Orsvuran.
Following file structure as in the Seisflows example (By Bryant Chow)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specster.core.misc import grid


class KernelKeeper:
    """A simple class for managing kernels."""

    def __init__(self, output_directory, sources=None, receivers=None):
        self.kernel_files = list(Path(output_directory).rglob("*kernel.dat"))
        # find rho alpha beta and load it
        out = [x for x in self.kernel_files if "rhop_alpha_beta" in x.name]
        assert len(out) == 1, "Exactly one kernel should exist."
        self.rho_alpha_beta = self.load_kernel_file(out[0])
        self._sources = sources
        self._receivers = receivers

    def load_kernel_file(self, file_path):
        """Load a particular kernel file into a dataframe."""
        names = ["x", "z"] + list(file_path.name.split("_")[1:-1])
        df = pd.read_csv(file_path, delim_whitespace=True, names=names, header=None)
        return df

    def plot(self, columns=("rhop", "alpha", "beta"), scale=0.15, out_file=None):
        """
        Plot several kernels.
        """
        fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
        for ax, column in zip(axes.flatten(), columns):
            self.plot_single(column, ax=ax, scale=scale)
        if out_file is not None:
            plt.tight_layout()
            fig.savefig(out_file)
        return fig, axes

    def plot_single(self, column, ax=None, scale=0.25):
        """Plot Rho, Alpha, Beta"""
        df = self.rho_alpha_beta
        data = self.rho_alpha_beta[column]
        abs_max_val = np.abs(data).max()
        min_val, max_val = -abs_max_val * scale, abs_max_val * scale

        # extract/format data
        x_vals, z_vals, data = grid(df["x"], df["z"], df[column])
        extent = [df["x"].min(), df["x"].max(), df["z"].min(), df["z"].max()]

        # Setup figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # plot, set labels etc.
        im = ax.imshow(
            data, extent=extent, cmap="seismic_r", vmin=min_val, vmax=max_val
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{column.title()} Kernel")

        # Plot source and
        kwargs = dict(color="black", edgecolor="white")
        if self._sources is not None:
            for (x, z) in self._sources:
                ax.scatter(x, z, 1000, marker="*", **kwargs)
        if self._receivers is not None:
            for (x, z) in self._receivers:
                ax.scatter(x, z, 450, marker="v", **kwargs)

        plt.colorbar(im, ax=ax)
        ax.tick_params(axis="both", which="major", labelsize=14)
        return ax
