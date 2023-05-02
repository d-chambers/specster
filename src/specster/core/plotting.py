"""
Module for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import specster
from specster.core.misc import grid

from .grid import df_to_grid


def plot_gll_data(
    df,
    coord_labels=("x", "z"),
    exclude=("proc",),
    kernel=None,
    alpha=None,
    max_dist=4,
):
    """Plot the values in the grid."""
    if not set(coord_labels) & set(df.columns):
        df = df.reset_index()
    non_coord_cols = set(df.columns) - set(coord_labels) - set(exclude)
    if kernel is not None:
        if isinstance(kernel, str):
            kernel = {kernel}
        kernel = set(kernel)
        assert kernel.issubset(non_coord_cols)
        non_coord_cols = sorted(set(kernel) & set(non_coord_cols))
    fig_size = (6 * len(non_coord_cols), 6)
    fig, axes = plt.subplots(1, len(non_coord_cols), figsize=fig_size)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for non_coord_col, ax in zip(sorted(non_coord_cols), axes):
        coords, vals = df_to_grid(
            df, non_coord_col, coords=coord_labels, max_dist=max_dist
        )
        extents = [min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])]
        im = ax.imshow(vals, origin="lower", extent=extents, alpha=alpha)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title(non_coord_col)
        ax.set_ylabel(coord_labels[1])
        ax.set_xlabel(coord_labels[0])
        fig.colorbar(im, cax=cax, fraction=0.039, pad=0.04)
        _maybe_switch_axis_to_km(ax)
    plt.tight_layout()
    return fig, axes


def plot_gll_historgrams(df, ax=None, title=""):
    """
    Plot a historgram of a gll.

    These are produced by e.g., Output2D
    """
    ax = ax or plt.subplots(1, 1)
    bin_center = (df["bin_start"] + df["bin_end"]) / 2
    width = bin_center.diff().mean()
    ax.bar(bin_center, df["elements"], width=width * 0.95)
    ax.set_title(title)
    ax.set_xlabel("GLL points per shortest wavelength")
    ax.set_ylabel("# Elements")
    return ax


def plot_kernels(
    output: "specster.OutPut2D",
    kernel_df,
    columns=None,
    scale=0.15,
    out_file=None,
    **kwargs,
):
    """
    Plot several kernels.
    """
    default_cols = [x for x in kernel_df.columns if x not in {"x", "z"}]
    cols = columns if columns is not None else default_cols
    columns = [columns] if isinstance(columns, str) else columns
    fig, axes = plt.subplots(1, len(columns))
    flat = axes if not isinstance(axes, np.ndarray) else axes.flatten()
    if isinstance(flat, plt.Axes):
        flat = [flat]
    for ax, column in zip(flat, columns):
        plot_single_kernel(output, kernel_df, column, ax=ax, scale=scale)
    if out_file is not None:
        plt.tight_layout()
        fig.savefig(out_file)
    return fig, axes


def plot_single_kernel(
    output: "specster.OutPut2D",
    df,
    column,
    ax=None,
    scale=0.25,
    max_stations=10,
):
    """Plot Rho, Alpha, Beta"""
    data = df[column]
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
        data,
        extent=extent,
        cmap="seismic_r",
        vmin=min_val,
        vmax=max_val,
        origin="lower",
        alpha=0.5,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{column.title()} Kernel")

    # Plot source and
    # kwargs = dict(color="black", edgecolor="white")
    source_df = output._control.get_source_df()
    station_df = output._control.get_station_df()
    if not station_df.empty and len(station_df) < max_stations:
        ax.plot(station_df["xs"], station_df["zs"], "^", color="k")
    if not source_df.empty:
        ax.plot(source_df["xs"], source_df["zs"], "*", color="red")
    plt.colorbar(im, cax=cax)
    ax.tick_params(axis="both", which="major", labelsize=14)
    _maybe_switch_axis_to_km(ax)
    return ax


def _maybe_switch_axis_to_km(ax: plt.Axes, max_value=10_000):
    """
    Look at both x/y axis and switch to km if they are too large.

    This just helps presentability of the figures.
    """

    xlims = ax.get_xlim()
    x_diff = xlims[1] - xlims[0]
    ylims = ax.get_ylim()
    y_diff = ylims[1] - ylims[0]
    if not (abs(x_diff) > max_value or abs(y_diff) > max_value):
        return
    ax.xaxis.set_major_formatter(lambda x, pos: str(int(x / 1_000)))
    ax.yaxis.set_major_formatter(lambda x, pos: str(int(x / 1_000)))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
