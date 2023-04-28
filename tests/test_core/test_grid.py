"""
Tests for converting to/from GLL and evenly sampled grids.
"""
import pytest

from specster.core.grid import grid_to_df, df_to_grid
from specster.core.parse import read_ascii_kernels


class TestConvertToRegularGrid:
    """Convert a kernel to a regular grid."""

    def test_convert_to_regular_sampled(self, weights_kernel):
        """Ensure weights can be converted to grid and back"""
        coords, array = df_to_grid(weights_kernel, 'weights')
        assert len(coords) == 2


class TestConvertToDf:
    """Convert a dataframe back to kernel."""

