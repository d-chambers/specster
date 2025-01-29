"""
Tests for specster utilities.
"""

import pytest
from specster.core.render import number_to_spec_str


class TestNumberToSpecStr:
    """Tests for converting a normal number to a specfem string."""

    inputs = (10.00, 1000, 1.2, 0.001, 0, 9999)
    expected = ("1.0d1", "1.0d3", "1.2d0", "1.0d-3", "0", "9999")

    @pytest.mark.parametrize("test_input,expected", zip(inputs, expected))
    def test_basic(self, test_input, expected):
        """Tests basic input/outputs."""
        out = number_to_spec_str(test_input)
        assert out == expected
