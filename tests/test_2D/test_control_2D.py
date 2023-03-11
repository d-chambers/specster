"""
Tests for control class.
"""
import specster as sp


class TestInit:
    """Tests for creating controls"""

    def test_from_data_directory(self, control_2d):
        """Tests for getting data from data directory."""
        assert isinstance(control_2d, sp.Control2d)

    def test_write_equals(self, control_2d, tmp_path):
        """Ensure writing then reading evaluates to equal."""
        # out = control_2d.write(path=tmp_path)
        # new = sp.Control2d(tmp_path)
        # assert out == new == control_2d


class TestCopy:
    """Tests for copying runSpec."""

    def test_copy_parfile_copy(self, control_2d):
        """Ensure the control class can be copied."""
        copy1 = control_2d.copy()
        assert copy1.base_path != control_2d.base_path


class TestWriteAndRun:
    """Tests for copying, writing, and running."""

    def test_write(self, control_2d, tmp_path):
        """Test for writing control to file."""
        copy = control_2d.copy(tmp_path)
        copy.write()
