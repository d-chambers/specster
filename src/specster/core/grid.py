"""
Module to interpolate values onto regular grids.

Useful for converting to/from GLL binary format and regular
rectangular format (hopefully with minimal loss).
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree


def get_average_dx(array):
    """Get an average sample spacing for an uneven array."""
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)
    lens = maxs - mins
    dx = np.sqrt(np.prod(lens) / len(array))
    return dx


def _get_regularly_sampled_coords(array):
    """Get extrapolate array into regularly sampled coords."""
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)
    dx = get_average_dx(array)
    out = [np.arange(mi, ma + dx, dx) for mi, ma in zip(mins, maxs)]
    return out, dx


def df_to_grid(df, column, coords=("x", "z"), max_dist=4):
    """Convert df to a grid of values."""
    if not set(df.columns) & set(coords):
        df = df.reset_index()
    xz = df[["x", "z"]].values
    new_coords, dx = _get_regularly_sampled_coords(xz)
    old_coords = df[list(coords)].values
    X, Y = np.meshgrid(*new_coords, indexing="ij")
    # interp = LinearNDInterpolator(old_coords, df[column], fill_value=0.0)
    interp = NearestNDInterpolator(old_coords, df[column])
    out = interp(X, Y)
    # Nan out points with distance gt than 2DX. This is needed because
    # specfem files don't store 0s of values above topography.
    tree = cKDTree(old_coords)
    xi = _ndim_coords_from_arrays((X, Y), ndim=old_coords.shape[1])
    dists, indexes = tree.query(xi)
    out[dists > max_dist * dx] = np.NaN
    return new_coords, out


def grid_to_df(grid_coords, values, df, coords=("x", "z")):
    """
    Go back from a grid to a dataframe (list-like) coords.
    """
    inter = RegularGridInterpolator(grid_coords, values)
    if not set(df.columns) & set(coords):
        df = df.reset_index()
    xz = df[["x", "z"]].values
    out = inter(xz)
    return out
