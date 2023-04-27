"""
Tests for parsing functions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from specster.core.parse import read_binaries_in_directory, write_directory_binaries

test_data_path = Path(__file__).parent / "test_data" / "bin_directory"


@pytest.fixture(scope='class')
def bin_df():
    """Read test binaries."""
    return read_binaries_in_directory(test_data_path)


class TestParseBins:
    """Tests for reading binary data in a directory."""

    def test_read(self, bin_df):
        """Ensure bins are read in"""
        assert isinstance(bin_df, pd.DataFrame)
        assert {'x', 'z', 'vs', 'vp'}.issubset(set(bin_df.columns))

    def test_write_bins(self, bin_df, tmp_path_factory):
        """Ensure bins can be round tripped."""
        path = tmp_path_factory.mktemp("roundtrip_bins")
        # convert a few cols to float64 just to check astype works
        new = bin_df.copy()
        new[new.columns[:2]] = new[new.columns[:2]].astype(np.float64)
        write_directory_binaries(new, path)
        bin_df2 = read_binaries_in_directory(path)
        assert bin_df.equals(bin_df2)
