"""
Module to interpolate values onto regular grids.

Useful for converting to/from GLL binary fromat and regular
rectangular format (hopefully with minimal loss).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator


def _get_regularly_sampled_coords(array):
    """Get extrapolate array into regularly sampled coords."""

    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)
    lens = maxs - mins
    dx = np.sqrt(len(array) / np.sum(lens))
    out = [np.arange(mi, ma + dx, dx) for mi, ma in zip(mins, maxs)]
    return out


def df_to_grid(df, column, coords=('x', 'z')):
    """Convert df to a grid of values."""
    new_coords = _get_regularly_sampled_coords(df[['x', 'z']].values)
    old_coords = df[list(coords)]
    X, Y = np.meshgrid(*new_coords)
    # interp = LinearNDInterpolator(old_coords, df[column])
    interp = NearestNDInterpolator(old_coords, df[column])
    out = interp(X, Y)
    return new_coords, out


def grid_to_df(grid_coords, values, df_coords):
    """
    Go back from a grid to a dataframe (list-like) coords.
    """
    inter = RegularGridInterpolator(grid_coords, values)
    breakpoint()




