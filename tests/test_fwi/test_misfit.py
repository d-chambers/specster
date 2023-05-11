"""
Tests for misfit functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import pytest

from specster.core.misc import get_stream_summary_df
from specster.exceptions import UnsetStreamsError
from specster.fwi import AmplitudeMisfit, TravelTimeMisfit, WaveformMisfit


@pytest.fixture(scope="class")
def true_st(test_data_path):
    """Load the true stream."""
    path = test_data_path / "misfit_waveforms" / "true.mseed"
    return obspy.read(path)


@pytest.fixture(scope="class")
def initial_st(test_data_path):
    """Load the initial stream."""
    path = test_data_path / "misfit_waveforms" / "initial.mseed"
    return obspy.read(path)


@pytest.fixture(scope="class")
def waveform_misfit():
    """Return the default waveform misfit."""
    return WaveformMisfit()


@pytest.fixture(scope="class")
def waveform_misfit_with_window():
    """Return the default waveform misfit with a windowing df."""
    return WaveformMisfit()


class TestMisfitInterface:
    """Basic tests for misfits."""

    def test_misfit_basic(self, waveform_misfit, initial_st, true_st):
        """Tests that the misfit is calculated for basic waveform misfit."""
        misfit = waveform_misfit.get_misfit(true_st, initial_st)
        assert len(misfit) == len(initial_st) == len(true_st)
        assert np.all(abs(misfit) >= 0)
        misfit2 = waveform_misfit.get_misfit()
        assert (misfit == misfit2).all()

    def test_misfit_no_tr_raises(self):
        """Ensure waveforms with no traces raises."""
        waveform_misfit = WaveformMisfit()

        with pytest.raises(UnsetStreamsError):
            waveform_misfit.get_misfit()

    def test_adjoint_source(self, waveform_misfit, initial_st, true_st):
        """Test for creating adjoint sources."""
        adjoint = waveform_misfit.get_adjoint_sources(true_st, initial_st)
        assert len(initial_st) == len(adjoint) == len(true_st)
        for tr1, tr2, tr3 in zip(initial_st, true_st, adjoint):
            expected_data = tr1.data - tr2.data
            assert np.allclose(expected_data, tr3.data)


class TestMisfitWindowing:
    """Tests for windowing"""

    @pytest.fixture(scope="class")
    def slim_window(self, initial_st):
        """
        Create a 'slim window' which just trims intial_st by a few secs
        and dropping the last trace.
        """
        df = get_stream_summary_df(initial_st)
        df["starttime"] += np.timedelta64(1, "s")
        df["endtime"] -= np.timedelta64(1, "s")
        return df.iloc[:-1]

    def test_basic_windowing(self, initial_st, true_st, slim_window):
        """Ensure a window"""
        misfit = WaveformMisfit(window_df=slim_window)
        mis = misfit.get_misfit(true_st, initial_st)
        assert len(mis) == len(slim_window)
        adjoint = misfit.get_adjoint_sources()
        assert len(adjoint) == len(initial_st)


class TestTravetimeMisfit:
    """Ensure the travel time misfit can be run."""

    def test_misfit(self, initial_st, true_st):
        """Ensure a window"""
        misfit = TravelTimeMisfit()
        mis = misfit.get_misfit(true_st, initial_st)
        assert not pd.isnull(mis).any()

    def test_adjoint(self, initial_st, true_st):
        """Ensure a stream is returned."""
        misfit = TravelTimeMisfit()
        adjoints = misfit.get_adjoint_sources(true_st, initial_st)
        assert len(adjoints) == len(initial_st)


class TestAmplitudeMisfit:
    """Ensure the amplitude misfit can be run."""

    def test_misfit(self, initial_st, true_st):
        """Ensure a window"""
        misfit = AmplitudeMisfit()
        mis = misfit.get_misfit(true_st, initial_st)
        assert not pd.isnull(mis).any()

    def test_adjoint(self, initial_st, true_st):
        """Ensure an adjoint stream is returned."""
        misfit = AmplitudeMisfit()
        adjoints = misfit.get_adjoint_sources(true_st, initial_st)
        assert len(adjoints) == len(initial_st)


class TestMisfitPlotting:
    """Test that the misfit can be plotted."""

    def test_simple_plot(self, initial_st, true_st):
        """A simple test of plotting."""
        misfit = WaveformMisfit()
        fig, *_ = misfit.plot(true_st, initial_st)
        assert isinstance(fig, plt.Figure)
