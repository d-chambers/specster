"""
2D plotting routines.
"""

import matplotlib.pyplot as plt
import numpy as np

import specster
from specster.core.misc import grid


def plot_kernels(
    output: "specster.OutPut2D",
    kernel_df,
    columns=None,
    scale=0.15,
    out_file=None,
):
    """
    Plot several kernels.
    """
    default_cols = [x for x in kernel_df.columns if x not in {"x", "z"}]
    cols = columns if columns is not None else default_cols
    fig, axes = plt.subplots(1, len(cols))
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
    im = ax.imshow(data, extent=extent, cmap="seismic_r", vmin=min_val, vmax=max_val)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{column.title()} Kernel")

    # Plot source and
    kwargs = dict(color="black", edgecolor="white")
    source_df = output._control.get_source_df()
    station_df = output._control.get_station_df()
    if not source_df.empty:
        ax.scatter(source_df["xs"], source_df["zs"], 1000, marker="*", **kwargs)
    if not station_df.empty and len(station_df) < max_stations:
        ax.scatter(station_df["xs"], station_df["zs"], 600, marker="^", **kwargs)
    plt.colorbar(im, ax=ax)
    ax.tick_params(axis="both", which="major", labelsize=14)
    return ax


def plot_geometry(control):
    """Plot geometry associated with testcase."""
    # fig, ax = plt.subplots(1, 1)
    # sta = local.station_location
    # source = local.source_location
    #
    # ax.scatter(
    #     source[0], source[1], 1000, marker="*", color="black", edgecolor="white"
    # )
    # ax.scatter(sta[0], sta[1], 450, marker="v", color="black", edgecolor="white")
    # ax.set_xlim(*local.x_lims)
    # ax.set_ylim(*local.z_lims)
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_title("Model Geometry")
    # return fig
