"""Tests for pre-conditioning."""

from pathlib import Path

import pandas as pd
import pytest
from specster.core.preconditioner import median_filter, smooth

here = Path(__file__).parent

data_dir = here / "test_data" / "precon"


@pytest.fixture(scope="class")
def example_stations():
    """Return a dataframe of example stations."""
    return pd.read_parquet(data_dir / "example_stations.parquet")


@pytest.fixture(scope="class")
def example_kernel():
    """Return a dataframe of example stations."""
    out = pd.read_parquet(data_dir / "iteration_kernel.parquet")
    assert len(out.columns)
    return out


@pytest.fixture(scope="class")
def filtered_kernel(example_stations, example_kernel):
    """Return example stations and kernel with median filter applied."""
    return median_filter(example_kernel, example_stations)


class TestMedianFilter:
    """Test that the median filter mutes stations."""

    def test_shape(self, example_kernel, filtered_kernel):
        """Ensure the shape of the kernel remains unchanged."""
        assert len(example_kernel) == len(filtered_kernel)


class TestSmoothKernel:
    """Tests for smoothing the kernel."""

    def test_simple_smooth(self, example_kernel):
        """Test that smoothing doesn't raise an error."""
        _ = smooth(example_kernel, 12)
