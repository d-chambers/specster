"""
Module for working with station data.
"""
from pathlib import Path

from pydantic import Field

from specster.core.misc import find_file_startswith
from specster.core.models import SpecFloat, SpecsterModel
from specster.core.parse import iter_file_lines


def read_stations(value, path, **kwargs):
    """Read stations from an external file."""
    # Note: value is poorly named but required for compat with
    # other special functions in parsing logic. See constructs in
    # the par file for more details.
    path = Path(path)
    if not path.is_dir():
        path = path.parent
    if not value:  # nothing to do
        return []
    station_path = find_file_startswith(path, "STATIONS")
    assert station_path.exists()
    stations = [Station2D.read_line(line) for line in iter_file_lines(station_path)]
    return stations


class Station2D(SpecsterModel):
    """A single station."""

    _spaces = {
        "station": 5,
        "network": 6,
        "xs": 21,
        "zs": 21,
        "void1_": 10,
        "void2_": 12,
    }
    _decimal_precision = {
        "xs": 7,
        "zs": 7,
        "void1_": 1,
        "void2_": 2,
    }

    station: str = Field("001", description="station name")
    network: str = Field("UU", description="network name")
    xs: SpecFloat = Field(0.0, description="X location in meters")
    zs: SpecFloat = Field(0.0, description="Z location in meters")
    # TODO: See what these columns actually are
    void1_: SpecFloat = ""
    void2_: SpecFloat = ""

    def write_model_data(self, key=None):
        """Write the model data to disk."""
        out = []
        for field in self.__fields__:
            val = getattr(self, field)
            if field in self._decimal_precision:
                prec = ".0{dec}f".format(dec=self._decimal_precision.get(field))
                str_val = ("{val:" + f"{prec}" + "}").format(val=float(val))
            else:
                str_val = f"{val}"
            out.append(str_val.rjust(self._spaces[field]))
        return "".join(out)


def _maybe_use_station_file(self, station_list):
    """Maybe use the station file, switch appropriate params."""
    # set use existing stations
    use_stations = bool(station_list)
    self.par.receivers.use_existing_stations = use_stations
    # then read station_file if needed
    if use_stations:
        stations = read_stations(True, self.base_path / "DATA" / "STATIONS")
        self.par.receivers.stations = stations
    else:
        self.par.receivers.stations = []
