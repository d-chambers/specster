"""
Module for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obsplus.waveforms import get_waveforms

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
    ax=None,
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
        plot_single_gll(ax, df, non_coord_col, coord_labels, max_dist, alpha=alpha)
    plt.tight_layout()
    return fig, axes


def plot_single_gll(ax, df, column, coord_labels, max_dist, alpha=None, vlims=None):
    """Plot a single GLL column in dataframe."""
    if not vlims:
        vmin, vmax = None, None
    else:
        assert len(vlims) == 2
        vmin, vmax = vlims[0], vlims[1]
    coords, vals = df_to_grid(df, column, coords=coord_labels, max_dist=max_dist)
    extents = [min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])]
    im = ax.imshow(
        vals.T, origin="lower", extent=extents, alpha=alpha, vmin=vmin, vmax=vmax
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(column)
    ax.set_ylabel(coord_labels[1])
    ax.set_xlabel(coord_labels[0])
    fig = ax.get_figure()
    fig.colorbar(im, cax=cax, fraction=0.039, pad=0.04)
    _maybe_switch_axis_to_km(ax)
    return ax


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
    alpha=0.5,
):
    """Plot Rho, Alpha, Beta"""
    if not {"x", "z"}.issubset(set(df.columns)):
        df = df.reset_index()
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
        alpha=alpha,
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


def plot_misfit(self, st_obs=None, st_synth=None, station=None):
    """
    Create a plot of misfit results.

    Parameters
    ----------
    self
        An instance of BaseMisfit.
    st_obs
        The stream with observed data (optional if already set)
    st_synth
        The stream with synthetic data (optional if already set)
    station
        A station string, if None use first station.
    """

    def add_legends(ax):
        """Add the legends for component and synth/observed."""
        line1 = Line2D([0], [0], color="0.5", ls="--", label="predicted")
        line2 = Line2D([0], [0], color="0.5", ls="-", label="observed")

        # Create a legend for the first line.
        leg1 = ax.legend(handles=[line1, line2], loc="upper right")
        ax.add_artist(leg1)

        color_lines = [
            Line2D(
                [0],
                [0],
                color=_component_colors[x],
                ls="-",
                label=f"{x} component",
            )
            for x in _component_colors
        ]
        ax.legend(handles=color_lines, loc="upper left")

    _component_colors = {"Z": "orange", "X": "cyan", "Y": "Red"}
    fig, (wf_ax, ad_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    # calc adjoints
    adjoint = self.get_adjoint_sources(st_obs, st_synth)
    misfit = self.get_misfit()
    # get dataframe of waveforms
    df = self.waveform_df_.assign(misfit=misfit)
    unique_stations = np.unique(df["station"])
    station = unique_stations[0] if station is None else station

    sub_df = df[df["station"] == station]
    sub_adjoint = get_waveforms(adjoint, station=station)
    # first plot traces
    for _, row in sub_df.iterrows():
        color = _component_colors[row["channel"][-1]]
        tr_obs, tr_synth = row["tr_obs"], row["tr_synth"]
        wf_ax.plot(tr_obs.times(), tr_obs.data, "-", color=color, alpha=0.5)
        wf_ax.plot(tr_synth.times(), tr_synth.data, "--", color=color, alpha=0.5)
        add_legends(wf_ax)
    # next plot adjoint
    for tr in sub_adjoint:
        color = _component_colors[tr.stats.channel[-1]]
        ad_ax.plot(tr.times(), tr.data, "-", color=color, alpha=0.5)

    wf_ax.set_title("Waveforms")
    ad_ax.set_title("Adjoint Source")

    ad_ax.set_xlabel("Time (s)")
    fig.supylabel("Displacement (m)")

    return fig, (wf_ax, ad_ax)
