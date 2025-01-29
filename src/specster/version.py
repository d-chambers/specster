"""Module for reporting the version of specster."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("specster")
# package is not installed
except PackageNotFoundError:
    __version__ = "0.0.0"
