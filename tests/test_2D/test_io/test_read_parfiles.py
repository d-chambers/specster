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
        out = pf.ReceiverSets.read_receiver_sets(nlines, iterable)
        assert isinstance(out, pf.ReceiverSets)


class TestParseExamplePars:
    """Tests for reading example files."""

    def test_parse(self, par_file_path):
        """Ensure each parfile can be read."""
        par = pf.read_parfile(par_file_path)
        assert isinstance(par, pf.RunParameters)
