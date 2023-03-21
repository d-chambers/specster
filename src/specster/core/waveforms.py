"""
Utilities for working with specfem waveforms
"""
from pathlib import Path

import numpy as np
import obspy


def read_ascii_stream(path):
    """
    Read all ascii streams.
    """
    path = Path(path)
    assert path.exists(), f"{path} does not exist."
    if path.is_dir():  # read all files in path
        traces = [read_ascii_trace(x) for x in path.glob("*semd")]
    else:
        traces = [read_ascii_trace(path)]
    return obspy.Stream(traces)


def read_generic_trace(path):
    """Read a trace without meaning in file name (eg source-time func)."""
    path = Path(path)
    data = np.loadtxt(str(path))
    # first column is time, second column is the data
    times = data[:, 0]
    disp = data[:, 1]
    delta = times[1] - times[0]
    headers = {"delta": delta, "b": times[0], "starttime": obspy.UTCDateTime(times[0])}
    return obspy.Trace(disp, headers)


def read_ascii_trace(path):
    """Reads an ASCII file and returns a obspy Traces"""
    path = Path(path)
    data = np.loadtxt(str(path))
    # first column is time, second column is the data
    times = data[:, 0]
    disp = data[:, 1]
    # get station name from the filename
    net, sta, comp, *_ = path.name.split("/")[-1].split(".")
    delta = times[1] - times[0]
    headers = {
        "station": sta,
        "network": net,
        "channel": comp,
        "delta": delta,
        "b": times[0],
        "starttime": obspy.UTCDateTime(times[0]),
    }
    return obspy.Trace(disp, headers)


def write_ascii_waveforms(tr, filename):
    """
    Writes out the traces as an ASCII file.
    Uses b value as the beginning.
    """
    data = np.zeros((len(tr.data), 2))
    data[:, 0] = tr.times() + tr.stats.starttime.timestamp
    data[:, 1] = tr.data
    np.savetxt(filename, data)
