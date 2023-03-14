"""
Module for handling specster's settings/behavior.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseSettings, Field, root_validator

import specster

from .exceptions import MissingSpecFEMError


class Settings(BaseSettings):
    """Global settings for specster."""

    spec_bin_path: Optional[Path] = Field(
        env="SPECFEM_BIN_PATH",
        default=None,
        description="Path to folder containing specfem binaries",
    )
    spec_path: Optional[Path] = Field(
        env="SPECFEM_PATH",
        default=None,
        description="Path to folder containing specfem source",
    )

    package_path: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).parent.absolute(),
    )

    @root_validator()
    def validate_spec_bin(cls, values):
        """Validate the spectral bins"""
        spec_bin_path = values.get("spec_bin_path")
        if not spec_bin_path:
            spec_path = values.get("spec_path")
            if spec_path is None:
                raise ValueError("No spec path specified!")
            values["spec_bin_path"] = spec_path / "bin"
        return values

    mode: Literal["2D", "3d"] = "2D"

    def assert_bin_available(self):
        """Ensure the bin directory is found else raise."""
        spec_bin_path = self.spec_bin_path
        if spec_bin_path is not None and self.spec_bin_path.exists():
            return
        # if spec_path is set and binaries exist, just use those.
        if self.spec_path is not None and (self.spec_path / "bin").exists():
            self.spec_bin_path = self.spec_path / "bin"
            return
        msg = (
            "SpecFEM binaries not found! You should make sure you have "
            "downloaded and installed specfem from the specfem github"
            "repo: https://github.com/specfem"
        )
        raise MissingSpecFEMError(msg)

    def assert_specfem_available(self):
        """Ensure the bin directory is found else raise."""
        if self.spec_bin_path is not None and self.spec_bin_path.exists():
            return
        # if spec_path is set and binaries exist, just use those.
        if self.spec_path is not None and (self.spec_path / "bin").exists():
            self.spec_bin_path = self.spec_path / "bin"
            return
        msg = (
            "SpecFEM repo not found! You should make sure you have "
            "downloaded specfem from the specfem github"
            "repo: https://github.com/specfem"
        )
        raise MissingSpecFEMError(msg)


def write_settings(path, settings: Optional[Settings] = None):
    """
    Write the settings to disk as a simple json file.

    Parameters
    ----------
    path
        The path to the settings.
    settings
        A settings object, if None use the global settings.
    """
    settings = settings if settings is not None else specster.settings
    json = settings.json()
    with Path(path).open("w") as fi:
        fi.write(json)


def read_settings(path, use_settings=False):
    """
    Read settings from disk.

    Parameters
    ----------
    path
        Path to the settings json file.
    use_settings
        If True, set the loading settings to the global default.
    """
    settings = Settings.parse_file(path)
    if use_settings:
        specster.settings = settings
    return settings
