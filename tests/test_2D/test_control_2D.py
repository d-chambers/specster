"""
Tests for control class.
"""
import specster as sp
from specster.utils.misc import assert_models_equal


class TestInit:
    """Tests for creating controls"""

    def test_from_data_directory(self, control_2d):
        """Tests for getting data from data directory."""
        assert isinstance(control_2d, sp.Control2d)

    def test_write_equals(self, control_2d, tmp_path):
        """Ensure writing then reading evaluates to equal."""
        out = control_2d.write(path=tmp_path)
        new = sp.Control2d(tmp_path)
        assert_models_equal(out.par, new.par)

    def test_default_init(self):
        """Ensure the default data is populated."""
        control = sp.Control2d()
        assert isinstance(control, sp.Control2d)


class TestCopy:
    """Tests for copying runSpec."""

    def test_copy_parfile_copy(self, control_2d):
        """Ensure the control class can be copied."""
        copy1 = control_2d.copy()
        assert copy1.base_path != control_2d.base_path


class TestRun:
    """Tests for copying, writing, and running."""

    def test_write_and_run_default(self, tmp_path):
        """Test the default can be written and run."""
        # control = sp.Control2d().copy(tmp_path)
        # control.xspecfem2d()
        # control.xspecfem2d()
