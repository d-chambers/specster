"""
Module for handling specster's settings/behavior.
"""
import os
from functools import partial
from pathlib import Path
from typing import Optional

from pydantic import Field

import specster
from specster.core.models import SpecsterModel


def maybe_load_env_path(name):
    """Maybe load an environmental variable, else return None."""
    out = os.environ.get(name, None)
    return out if not out else Path(out)


class Settings(SpecsterModel):
    """Global settings for specster."""

    specfem2d_bin_path: Optional[Path] = Field(
        default_factory=partial(
            maybe_load_env_path,
            name="SPECFEM_BIN_PATH",
        ),
        description="Path to folder containing specfem binaries",
    )
    specfem2d_path: Optional[Path] = Field(
        default_factory=partial(
            maybe_load_env_path,
            name="SPECFEM2D_PATH",
        ),
        description="Path to folder containing specfem2D source",
    )

    package_path: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).parent.absolute(),
    )

    ci: bool = Field(
        validation_alias="CI",
        default=False,
        description="If running on continuous integration.",
    )

    def get_specfem2d_binary_path(self):
        """ "Get the binary path to specfem 2D."""
        spec_bin_path = self.specfem2d_bin_path
        if not spec_bin_path or not Path(spec_bin_path).exists():
            spec_path = self.specfem2d_path
            if spec_path is None or not Path(self.specfem2d_path).exists():
                raise ValueError(
                    "You must specify either settings.specfem2d_bin_path"
                    "or settings.specfem2d_path."
                )
            spec_bin_path = spec_path / "bin"
        return spec_bin_path


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
