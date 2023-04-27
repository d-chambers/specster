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


@pytest.fixture(scope="class")
def initial_control(control_2d_default_3_sources):
    """Get the initial control and generate waveforms."""
    cache_name = "initial_control_fwi"
    control_true = control_2d_default_3_sources
    if not load_cache(cache_name):
        control = control_true.copy()
        # make salt a bit slower and run.
        material = control.par.material_models.models[3]
        material.Vp *= 0.97
        material.Vs *= 0.97
        control.write()
        control.run_each_source()
        cache_file_or_dir(control.base_path, cache_name)
    else:
        control = specster.Control2d(load_cache(cache_name))
    return control


@pytest.fixture(scope='class')
def inverter(true_control, initial_control, tmp_path_factory):
    """Init  the inverter."""
    path = tmp_path_factory.mktemp("inverter_test")
    inverter = fwi.Inverter(
        observed_data_path=true_control,
        initial_control=initial_control,
        misfit=fwi.WaveformMisFit,
        true_control=true_control,
        working_path=path,
    )
    return inverter


@pytest.fixture(scope='class')
def run_inverter(inverter):
    """run the inverter"""
    return inverter.invert()


@pytest.mark.e2e
@pytest.mark.slow
class TestBasicWorkflow:
    """Ensure the basic workflow functions"""
    def test_inverter(self, run_inverter):
        """Ensure the inverter was run."""
        breakpoint()
