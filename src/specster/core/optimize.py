"""
Module for simple optmiziations.
"""
import numpy as np


def golden_section_search(f, a, b, tol=1e-5, max_iter=1000):
    """Golden-section search.

    Modified from here: https://en.wikipedia.org/wiki/Golden-section_search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    """
    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return np.mean([a, b])

    # Required steps to achieve tolerance
    n_max = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    n = min(n_max, max_iter)

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return np.mean([a, d])
    else:
        return np.mean([c, b])
