"""
Utils module to help parse specfem files.
"""
from pathlib import Path

import numpy as np

from specster.constants import _SUB_VALUES


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
