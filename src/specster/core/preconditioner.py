"""
Conditioners to apply to gradients.
"""

import numpy as np
from scipy import spatial

from specster.constants import XYZ
from specster.core.grid import get_average_dx

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
    for column in kernel_df:
        if column == "proc":
            continue
        data = kernel_df[column].values
        for sub_array in points:
            data[sub_array] = np.median(data[sub_array])
        out[column] = data
    return out
