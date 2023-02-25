"""
Misc small utilities.
"""
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


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


@cache
def get_env(template_path):
    """Get the template environment."""
    template_path = Path(template_path)
    env = Environment(loader=FileSystemLoader(template_path))
    return env


@cache
def get_template(template_path, name):
    """Get the template for rendering tables."""
    env = get_env(template_path)
    template = env.get_template(name)
    return template
