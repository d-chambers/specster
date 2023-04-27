"""
Utils module to help parse specfem files.
"""
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
import obspy
import pandas as pd

from specster.constants import _SUB_VALUES, PROC_ZEROES, IGNORE_BINS


def extract_parline_key_value(line):
    """Extract key/value pairs from a single line of the par file."""
    key_value = line.split("=")
    key = key_value[0].strip().lower()
    value = key_value[1].split("#")[0].strip()
    return key, _SUB_VALUES.get(value, value)


def iter_file_lines(path, ignore="#"):
    """Read lines of a file, dont include comment lines."""
    with open(path, "r") as fi:
        for line in fi.readlines():
            stripped = line.strip()
            if stripped.startswith(ignore) or not stripped:
                continue
            yield line


def read_binaries_in_directory(path) -> pd.DataFrame:
    """
    Read all the binary data files in a path.

    Parameters
    ----------
    path

    Returns
    -------

    """
    out = defaultdict(dict)

    for bin_path in Path(path).glob('proc*.bin'):
        name = bin_path.name
        proc = int(name.split('_')[0].replace("proc", ""))
        field_name = "".join(name.split('_')[1:]).replace('.bin', '')
        if field_name in IGNORE_BINS:
            continue
        array = read_specfem_binary(bin_path)
        out[proc][field_name] = array

    dfs = [pd.DataFrame(v).assign(proc=i) for i, v in out.items()]
    if not dfs:
        return pd.DataFrame()
    total_df = pd.concat(dfs)
    return total_df[sorted(total_df.columns)]


def write_directory_binaries(df, path):
    """
    Write contents in df back to directory.

    Parameters
    ----------
    df
        The dataframe. Must have proc column.
    path
        The directory path to write to.
    """
    path = Path(path)
    assert not path.is_file(), "path must be a directory."
    path.mkdir(exist_ok=True, parents=True)
    for proc, proc_df in df.groupby("proc"):
        proc = int(proc)
        for col in proc_df.columns:
            data = proc_df[col].values
            name = f"proc{proc:06d}_{col}.bin"
            data_path = path / name
            write_specfem_binary(data, data_path)


def read_specfem_binary(path):
    """
    Reads the specfem2D fortran style code into an array.

    Notes
    -----

    Based on:
    https://github.com/adjtomo/seisflows/blob/c7ef6b4bdb96b1d8ca223cb104b2190c3c9571fb
    /seisflows/tools/specfem.py#L247
    """

    def _has_buffer(fi, byte_size):
        """Check if file has int buffer at start/end."""
        fi.seek(0)
        maybe_byte_count = np.fromfile(fi, dtype="int32", count=1)[0]
        return maybe_byte_count == byte_size - 8

    path = Path(path)
    byte_size = path.stat().st_size
    with path.open("rb") as file:
        if _has_buffer(file, byte_size):
            file.seek(4)
            data = np.fromfile(file, dtype="float32")
            return data[:-1]
        else:
            file.seek(0)
            data = np.fromfile(file, dtype="float32")
            return data


def write_specfem_binary(data, path):
    """
    Writes specfem binary file.

    Notes
    -----
    Uses single precision numbers, and writes 4byte int at start and end.

    based on:
    https://github.com/adjtomo/seisflows/blob/c7ef6b4bdb96b1d8ca223cb104b2190c3c9571fb
    /seisflows/tools/specfem.py#L278

    """
    path = Path(path)
    data_size_in_bytes = np.array([4 * len(data)], dtype="int32")
    data = np.array(data).astype(np.float32)

    with path.open("wb") as fi:
        data_size_in_bytes.tofile(fi)
        data.tofile(fi)
        data_size_in_bytes.tofile(fi)


def read_waveform_source_directory(path) -> Dict[str, obspy.Stream]:
    """Read waveforms into source dictionary."""
    path = Path(path)
    out = {}
    for source_dir in [f for f in path.iterdir() if f.is_dir()]:
        name = source_dir.name
        out[name] = read_ascii_stream(source_dir)
    return out


def read_ascii_stream(path):
    """
    Read all ascii streams.
    """
    path = Path(path)
    assert path.exists(), f"{path} does not exist."
    if path.is_dir():  # read all files in path
        traces = [read_ascii_trace(x) for x in sorted(path.rglob("*semd"))]
    else:
        traces = [read_ascii_trace(path)]
    return obspy.Stream(traces).sort()


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
