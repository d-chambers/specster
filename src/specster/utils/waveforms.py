"""
Utilities for working with specfem waveforms
"""
from pathlib import Path

import numpy as np
import obspy


def read_stream(path):
    """
    Read all ascii streams.
    """
    path = Path(path)
    assert path.is_dir()
    traces = [read_trace(x) for x in path.glob('*semd')]
    return obspy.Stream(traces)


def read_trace(filename):
    """Reads an ASCII file and returns a obspy Traces"""
    data = np.loadtxt(filename)
    # first column is time, second column is the data
    times = data[:, 0]
    disp = data[:, 1]
    # get station name from the filename
    net, sta, comp, *_ = filename.split("/")[-1].split(".")
    delta = times[1] - times[0]
    headers = {"station": sta, "network": net, "channel": comp, "delta": delta, "b": times[0]}
    return obspy.Trace(disp, headers)


def save_trace(tr, filename):
    """Writes out the traces as an ASCII file. Uses b value as the beginning."""
    data = np.zeros((len(tr.data), 2))
    data[:, 0] = tr.times() + tr.stats.b
    data[:, 1] = tr.data
    np.savetxt(filename, data)
