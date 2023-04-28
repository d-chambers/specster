"""
Test end to end for FWI workflow.
"""
import copy

import specster
from specster import fwi
from specster.core.misc import load_cache, cache_file_or_dir
import pytest


@pytest.fixture(scope="class")
def true_control(control_2d_default_3_sources):
    """Get the true control."""
    return control_2d_default_3_sources





@pytest.fixture(scope='class')
def inverter(true_control, initial_control, tmp_path_factory):
    """Init  the inverter."""
    path = tmp_path_factory.mktemp("inverter_test")
    inverter = fwi.Inverter(
        observed_data_path=true_control,
        control=initial_control,
        misfit=fwi.WaveformMisFit,
        true_control=true_control,
        working_path=path,
    )
    return inverter


@pytest.fixture(scope='class')
def run_inverter(inverter):
    """run the inverter"""
    return inverter.run_inversion_iteration()


@pytest.mark.e2e
@pytest.mark.slow
class TestBasicWorkflow:
    """Ensure the basic workflow functions"""
    def test_inverter(self, run_inverter):
        """Ensure the inverter was run."""
        breakpoint()
