"""
Tests for reading Kernels
"""
from pathlib import Path

import pandas as pd

from specster.core.parse import read_ascii_kernels


class TestReadKernel:
    """Tests for reading kernels."""

    def test_read_all_kernels(self, kernel_2d_dir_path):
        """tests for reading the Hessian."""
        kernel = read_ascii_kernels(kernel_2d_dir_path)
        assert isinstance(kernel, pd.DataFrame)
        assert len(kernel)

    def test_read_hessian(self, kernel_2d_dir_path):
        """tests for reading the Hessian."""
        kernel = read_ascii_kernels(kernel_2d_dir_path, kernel="Hessian2")
        assert isinstance(kernel, pd.DataFrame)
        assert len(kernel)
