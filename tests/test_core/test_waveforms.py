"""
Tests for reading/writing waveforms.
"""
from pathlib import Path

import numpy as np

import specster.core.parse

BASE_TEST_PATH = Path(__file__).parent.parent


class TestReadWaveforms:
    """Tests for reading waveform files into obspy streams."""

    def test_read_ascii_waveforms(self, test_data_path):
        """Ensure all the ascii waveforms can be read."""
        path = test_data_path / "ascii_waveform_files"
        st = specster.core.parse.read_ascii_stream(path)
        waveform_files = list(path.glob("*.semd"))
        assert len(st) == len(waveform_files)
        for tr in st:
            assert np.all(~np.isnan(tr.data))
