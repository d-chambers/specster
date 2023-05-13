"""
Tests for converting to/from GLL and evenly sampled grids.
"""

from specster.core.grid import df_to_grid


class TestConvertToRegularGrid:
    """Convert a kernel to a regular grid."""

    def test_convert_to_regular_sampled(self, weights_kernel):
        """Ensure weights can be converted to grid and back"""
        coords, array = df_to_grid(weights_kernel, "weights")
        assert len(coords) == 2
        assert tuple(len(x) for x in coords) == array.shape


class TestConvertToDf:
    """Convert a dataframe back to kernel."""
