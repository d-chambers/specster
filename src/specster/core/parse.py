"""
Utils module to help parse specfem files.
"""
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
