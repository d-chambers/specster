"""
Tests for 2d output object.
"""
import obspy
import pandas as pd
import pytest


@pytest.fixture(scope="class")
def basic_output(modified_control_ran):
    """return output object from modified control run."""
    return modified_control_ran.output


class TestBasic:
    """Ensure waveforms can be read in."""

    def test_stream_returned(self, basic_output):
        """Simply test waveforms were returned."""
        st = basic_output.get_waveforms()
        assert isinstance(st, obspy.Stream)

    def test_dataframes(self, basic_output):
        """Ensure the histograms can be converted to dataframes."""
        df_solid = basic_output.solid_gll_hist_df
        df_liquid = basic_output.fluid_gll_hist_df
        assert len(df_liquid) and len(df_solid)
        assert isinstance(df_liquid, pd.DataFrame)
        assert isinstance(df_solid, pd.DataFrame)
