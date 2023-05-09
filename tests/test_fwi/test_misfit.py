"""
Tests for misfit functions.
"""

import numpy as np
import obspy
import pytest

from specster.exceptions import UnsetStreamsError
from specster.fwi import WaveformMisfit


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
