"""
Conditioners to apply to gradients.
"""

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.ndimage import gaussian_filter

from specster.constants import XYZ
from specster.core.grid import df_to_grid, get_average_dx, grid_to_df

x, y = np.mgrid[0:4, 0:4]
points = np.c_[x.ravel(), y.ravel()]
tree = spatial.cKDTree(points)


def median_filter(kernel_df, station_df, dx_factor=1):
    """
    Mute bright spots related to stations from kernels.
    """
    out = kernel_df.copy()
    sta_df = station_df.rename(columns={"xs": "x", "zs": "z", "ys": "y"})
    coords = sorted(set(XYZ) & set(sta_df.columns))

    coords_kernel = out.reset_index()[coords].values
    dx = get_average_dx(coords_kernel)
    coords_stations = sta_df[coords].values

    # use cKDTree to get points close to stations
    tree = spatial.cKDTree(coords_kernel)
    points = tree.query_ball_point(coords_stations, r=dx * dx_factor)
    for column in kernel_df.columns:
        if column == "proc":
            continue
        data = kernel_df[column].values
        for sub_array in points:
            data[sub_array] = np.median(data[sub_array])
        out[column] = data
    return out


def smooth(kernel_df, sigma=5):
    """
    Smooth a kernel.

    Parameters
    ----------
    kernel_df
        Dataframe containing the kernels to smooth
    sigma
        The sigma values (in terms of dx) for Gaussian kernel. Can be a
        single number or list of values equal to dimensions of kernel_df.
    """
    out = pd.DataFrame(index=kernel_df.index)
    for col in kernel_df.columns:
        coords, grid = df_to_grid(kernel_df, col)
        smoothed = gaussian_filter(grid, sigma, mode="nearest")
        values = grid_to_df(coords, smoothed, kernel_df)
        out[col] = values
    return out
