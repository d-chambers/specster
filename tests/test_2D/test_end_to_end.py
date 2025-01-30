"""
End to end tests.
"""

import pytest
import specster as sp


@pytest.fixture(scope="class")
def acoustic_control(tmp_path_factory):
    """An acoustic model."""
    path = tmp_path_factory.mktemp("acoustic_control")
    cont = sp.load_2d_example("acoustic_reflector").copy(path)
    breakpoint()
    cont.prepare_fwi_forward().run()
    return cont


@pytest.mark.slow
class TestAcoustic2D:
    """Test case for acoustic 2D."""

    def test_acoustic(self, acoustic_control):
        """Simply run the acoustic example"""
        out = acoustic_control.output
        assert isinstance(out, sp.OutPut2D)
