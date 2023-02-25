"""
Tests for control class.
"""
import specster as sp


class TestReadFromFile:
    """Tests for creating control from files."""

    def test_from_data_directory(self, control_2d):
        """Tests for getting data from data directory."""
        assert isinstance(control_2d, sp.Control2d)


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
