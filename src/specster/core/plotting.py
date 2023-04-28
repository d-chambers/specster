"""
Module for plotting.
"""

import matplotlib.pyplot as plt

from .grid import df_to_grid


def plot_kernels(df, coord_labels=("x", "z"), exclude=('proc',)):
    """Plot the values in the grid."""
    non_coord_cols = set(df.columns) - set(coord_labels) - set(exclude)
    fig, axes = plt.subplots(len(non_coord_cols), 1)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for non_coord_col, ax in zip(sorted(non_coord_cols), axes):
        coords, vals = df_to_grid(df, non_coord_col, coords=coord_labels)
        extents = [min(coords[1]), max(coords[1]), min(coords[0]), max(coords[0])]
        im = ax.imshow(vals, aspect='auto', origin='lower', extent=extents)
        ax.set_title(non_coord_col)
        ax.set_ylabel(coord_labels[1])
        ax.set_xlabel(coord_labels[0])
        fig.colorbar(im, ax=ax)
    return fig, axes

