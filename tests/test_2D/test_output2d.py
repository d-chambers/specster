"""
Tests for 2d output object.
"""
import matplotlib.pyplot as plt
import obspy
import pandas as pd
import pytest


@pytest.fixture(scope="class")
def plotting_control(modified_control, tmp_path_factory):
    """Ensure basic plotting is completed."""
    new_path = tmp_path_factory.mktemp("tests")
    return modified_control.copy(new_path)


@pytest.fixture(scope="class")
def basic_output(modified_control_ran):
    """return output object from modified control run."""
    return modified_control_ran.output


class TestBasic:
    """Ensure waveforms can be read in."""

    def test_lims(self, basic_output):
        """Ensure the limits of the simulation were run and are parsable."""
        assert len(basic_output.stats.x_lims)
        assert len(basic_output.stats.z_lims)

    def test_get_waveforms(self, basic_output):
        """Simply test waveforms were returned."""
        st = basic_output.get_waveforms()
        assert isinstance(st, obspy.Stream)

    def test_get_source_time_function(self, basic_output):
        """Ensure source time function can be fetched as stream."""
        st = basic_output.get_source_time_function()
        assert isinstance(st, obspy.Stream)
        assert len(st) == 1

    def test_dataframes(self, basic_output):
        """Ensure the histograms can be converted to dataframes."""
        df_solid = basic_output.solid_gll_hist_df
        df_liquid = basic_output.fluid_gll_hist_df
        assert len(df_liquid) and len(df_solid)
        assert isinstance(df_liquid, pd.DataFrame)
        assert isinstance(df_solid, pd.DataFrame)


class TestPlotGLLHistograms:
    """Plot the histograms of GLL points per wavelength."""

    def test_basic_plot(self, modified_control_ran):
        """Ensure basic plotting works of histogram."""
        output = modified_control_ran.output
        fig, axes = output.plot_gll_per_wavelength_histogram()
        assert isinstance(fig, plt.Figure)
        assert len(axes)


@pytest.mark.slow
class TestPlotGeometry:
    """Tests to ensure geometry can be plotted."""

    def test_basic(self, plotting_control):
        """Ensure basic plotting is completed."""
        fig, axes = plotting_control.plot_geometry()
        assert isinstance(fig, plt.Figure)
        assert len(axes)
