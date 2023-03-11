"""
Tests for reading par files.
"""

import pytest

import specster.d2.io.parfile as pf


class TestReadMaterialLine:
    """Tests for reading the material line."""

    basic_lines = (
        "1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0",
        "2 1 2500.d0 2700.d0 0 0 0 9999 9999 0 0 0 0 0 0",
        "3 1 2200.d0 2500.d0 1443.375d0 0 0 9999 9999 0 0 0 0 0 0",
        "4 1 2200.d0 2200.d0 1343.375d0 0 0 9999 9999 0 0 0 0 0 0",
    )

    @pytest.mark.parametrize("line", basic_lines)
    def test_elastic_example(self, line):
        """Tests for reading elastic material line."""
        elas = pf.ElasticModel.read_line(line)
        assert isinstance(elas, pf.ElasticModel)


class TestReadRegionLine:
    """Tests for reading a region from par file."""

    region_lines = (
        "1 80  1 20 1",
        "1 59 21 40 2",
        "71 80 21 40 2",
        "1 80 41 60 3",
        "60 70 21 40 4",
    )

    @pytest.mark.parametrize("line", region_lines)
    def test_elastic_example(self, line):
        """Tests for reading elastic material line."""
        elas = pf.Region2D.read_line(line)
        assert isinstance(elas, pf.Region2D)


class TestReadReceivers:
    """Tests for reading receivers from a string."""

    def test_read_lines(self, receiver_2d_line):
        """Simple test for reading receiver lines"""
        comment_free_lines = [
            x for x in receiver_2d_line.splitlines() if not x.startswith("#") and x
        ]
        nlines = int(comment_free_lines[0].split("=")[1].strip())
        iterable = (x for x in comment_free_lines[1:])
        out = pf.ReceiverSets.read_receiver_sets(nlines, iterable, state={})
        assert isinstance(out, pf.ReceiverSets)


class TestParseExamplePars:
    """Tests for reading example files."""

    def test_parse(self, run_parameters_2d):
        """Ensure each parfile can be read."""
        assert isinstance(run_parameters_2d, pf.SpecParameters2D)


class TestModels:
    """Tests for models read in."""

    def test_nbmodels_matches_model_number(self, run_parameters_2d):
        """Ensure number of models matches reported value."""
        matmods = run_parameters_2d.material_models
        assert matmods.nbmodels == len(matmods.models)


class TestSources:
    """Tests for sources attached to parfiles."""

    def test_sources_attached(self, run_parameters_2d):
        """Ensure run has some sources attached."""
        sources = run_parameters_2d.sources
        assert sources.nsources == len(sources.sources)
        for source in sources.sources:
            assert isinstance(source, pf.Source)


class TestStations:
    """Ensure external stations are read in ."""

    def test_external_station(self, run_parameters_2d):
        """Ensure external station file is read if used."""
        if not run_parameters_2d.receivers.use_existing_stations:
            pytest.skip("No external stations on this file.")
        assert len(run_parameters_2d.receivers.stations)
        for station in run_parameters_2d.receivers.stations:
            assert isinstance(station, pf.Station)


class TestDisp:
    """Tests for displaying data lines."""

    @pytest.fixture(scope="class")
    def param_display(self, run_parameters_2d):
        """Return display object for testing."""
        return run_parameters_2d.disp

    def test_toplevel_display(self, param_display):
        """Ensure display works."""
        title = param_display.title
        assert isinstance(title, str)
        assert len(title)
        title_2 = param_display.TITLE
        assert title_2 == title

    def test_recursive_display(self, param_display):
        """This should work for nested display as well."""
        # sub = param_display.visualizations
        # sub.postscript_display
