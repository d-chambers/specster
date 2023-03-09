"""
Misc small utilities.
"""
from functools import cache
from pathlib import Path

from jinja2 import Template

from specster.constants import special_dirs


def get_directory_path(base_path: Path, directory_name: str) -> Path:
    """Get the directory path, make it if it isn't there."""
    out = base_path / directory_name
    out.mkdir(exist_ok=True, parents=True)
    return out


def find_file_startswith(path: Path, startswith="Par_file"):
    """Try to find a file that starts with file_start in a directory."""
    if path.is_file() and startswith in path.name:
        return path
    parfiles = sorted(path.glob(f"{startswith}*"))
    if len(parfiles):
        return parfiles[0]
    msg = f"Unable to find {startswith} file in {path}"
    raise FileNotFoundError(msg)


def find_data_path(path):
    """Look for the data path."""
    if path.name.startswith("DATA"):
        return path
    return path / "DATA"


def find_base_path(path):
    """find the base path"""
    if path.name in special_dirs:
        return path.parent
    return path


@cache
def load_templates_from_directory(path: Path) -> dict:
    """Load all templates in directory."""
    assert path.exists() and path.is_dir()

    out = {}
    for path in path.glob("*"):
        with path.open("r") as fi:
            name = path.name.lower().rsplit(".", 1)[0]
            out[name] = Template(fi.read())
    return out