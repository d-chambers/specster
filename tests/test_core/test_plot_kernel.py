"""
Tests for plotting kernels.
"""
import matplotlib.pyplot as plt

from specster.core.plotting import plot_kernels


class TestPlotKernels:
    """Test plotting kernels"""

    def test_basic_plot(self, weights_kernel):
        """Test basic plots."""
        fig, axes = plot_kernels(weights_kernel)
        assert isinstance(fig, plt.Figure)
        assert len(axes)
