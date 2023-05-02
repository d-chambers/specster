"""
Test end to end for FWI workflow.
"""

import pytest


@pytest.fixture(scope="class")
def run_inverter(control_2d_inclusion_inversion):
    """run the inverter"""
    return control_2d_inclusion_inversion.run_iteration()


@pytest.mark.e2e
# @pytest.mark.slow
class TestBasicWorkflow:
    """Ensure the basic workflow functions"""

    def test_inverter(self, run_inverter):
        """Ensure the inverter was run."""
        breakpoint()
