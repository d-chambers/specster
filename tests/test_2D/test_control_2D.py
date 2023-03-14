"""
Tests for control class.
"""

import pytest

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


class TestModify:
    """Various tests for modifying control structure."""

    def test_modify_regions(self):
        """Ensure modifying regions also updates nbregions."""
        control = sp.Control2d()
        region_count = control.par.internal_meshing.regions.nbregions
        control.regions = control.regions[1:]
        assert region_count + 1 == control.par.internal_meshing.regions.nbregions


@pytest.mark.e2e
class TestEnd2End:
    """Various end-to-end tests."""

    @pytest.fixture(scope="class")
    def modified_control(self, tmp_path_factory):
        """Create a control class, perform several modifications."""
        tmp_path = tmp_path_factory.mktemp("end_to_end")
        control = sp.Control2d().copy(tmp_path)
        # first change model params
        mods = control.models
        mods[0].rho *= 1.01
        mods[0].Vp *= 1.01
        control.models = mods
        # Then remove all but 1 stations
        # sources = control.sources
        # breakpoint()
        return control

    @pytest.mark.slow
    def test_write_and_run_default(self, modified_control):
        """Test the default can be written and run."""
        modified_control.xmeshfem2d()
        modified_control.xspecfem2d()

    # @pytest.mark.slow
    # def
