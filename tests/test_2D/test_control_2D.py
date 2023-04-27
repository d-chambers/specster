"""
Tests for control class.
"""
from pathlib import Path
import copy

import pytest

import specster as sp
from specster.core.misc import assert_models_equal
from specster.d2.control2d import load_2d_example


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

    def test_stations_read(self, control_2d_default):
        """Ensure stations were read in."""
        stations = control_2d_default.stations
        assert len(stations) == 22, "should be 22 stations"


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
        control.regions = control.regions[1:]
        regions = control.par.internal_meshing.regions
        assert len(regions.regions) == regions.nbregions


class TestLoad2DExamples:
    """Ensure a 2D example can be loaded."""

    def test_load_tromp(self):
        """Laod one of the simple examples."""
        spec2d_path = sp.settings.specfem2d_path
        if spec2d_path is None or not spec2d_path.exists():
            pytest.skip("Specfem2d is not on source computer.")
        control = load_2d_example("Tromp2005")
        assert isinstance(control, sp.Control2d)


@pytest.mark.e2e
@pytest.mark.slow
class TestEnd2End:
    """Various end-to-end tests."""

    @pytest.mark.slow
    def test_write_and_run_default(self, modified_control_ran):
        """Test the default can be written and run."""
        output = modified_control_ran.output
        assert output.path.exists()


@pytest.mark.e2e
@pytest.mark.slow
class TestParallelEnd2End:
    """Various end-to-end tests."""

    def test_control_3_sources(self, control_2d_default_3_sources):
        """Ensure the expected directories were created and have waveforms."""
        control = control_2d_default_3_sources
        expected = Path(control.base_path / control._each_source_path)
        assert expected.exists() and expected.is_dir()
        sub_dirs = [x for x in expected.iterdir()]
        assert len(sub_dirs) == 3
        assert [int(x.name) for x in sorted(sub_dirs)] == [0, 1, 2]

    def test_each_source_output(self, control_2d_default_3_sources):
        """Ensure a list of output objects can be returned."""
        control = control_2d_default_3_sources
        out = control_2d_default_3_sources.each_source_output
        assert all([isinstance(x, sp.OutPut2D) for x in out])
