"""
Class for inverting material properties.
"""
import shutil
from typing import Optional, Literal, Type, Union
from pathlib import Path
from functools import cache

import numpy as np

import specster as sp
from ..core.control import BaseControl
from .misfit import _BaseMisFit
from specster.core.parse import read_ascii_stream


def _run_each_source(control):
    """Run each source, or if there is only one create correct output dir."""
    path = control.each_source_path
    if not path.exists():
        # just create output path
        if len(control.sources) == 1 and control.output_path.exists():
            shutil.copytree(control.output_path, control.each_source_path)
        else:
            control.run_each_source()
    return control


def _get_streams_from_each_source_dir(path):
    """Given a directory of event waveforms, get list of streams."""
    sorted_dirs = sorted(x for x in Path(path).iterdir())
    return [read_ascii_stream(x) for x in sorted_dirs]


class Inverter:
    """
    Class for running inversions.

    Parameters
    ----------
    observed_data_path
        The directory where the observed waveforms are stored.
        Should contain directories named after sources (sequential)
        then each of these waveforms in the convention semp format.
    """

    def __init__(
            self,
            observed_data_path: Union[Path, BaseControl],
            initial_control: BaseControl,
            misfit: Type[_BaseMisFit],
            true_control: Optional[BaseControl] = None,
            optimization: Literal['steepest descent'] = "steepest descent",
            stream_pre_processing=None,
            working_path="specster_scratch",
            pre_conditioner: Literal["default"] = "default",
            misfit_aggregator=np.linalg.norm,
    ):
        # set input params
        if isinstance(observed_data_path, BaseControl):
            observed_data_path = observed_data_path.each_source_path
        self._initial_control = _run_each_source(initial_control)
        self._misfit = misfit
        self._true_control = true_control
        self._optimization = optimization
        self._stream_pre_process = stream_pre_processing
        self.working_path = Path(working_path)
        self._create_working_directory(observed_data_path)

    def _create_working_directory(self, observed_data_path):
        """Create a working directory with observed/synthetic data."""
        # copy control
        control = self._initial_control.copy(self.working_path)
        # copy observed data
        obs_data_path = self.working_path / "observed_streams"
        if not obs_data_path.is_dir():
            shutil.copytree(observed_data_path, obs_data_path)

    def invert(self, max_iterations=10):

        """
        Run the inversion iteratively.
        """
        initial_streams = [
            self._preprocess_stream(x.get_waveforms())
            for x in self._initial_control.each_source_output
        ]
        misfit, adjoints = self._calc_misfit_adjoints(initial_streams)
        #
        # if not len(initial_st):  # need to run initial control
        #     self.initial_control.run()
        #     initial_st = self.initial_control.output.get_waveforms()
        #     assert len(initial_st)
        #
        # initial_misfit = self._calc_misfit_adjoints(current_st_list=initial_st)
        return self

    def _calc_misfit_adjoints(self, current_st_list):
        """Calculate the misfit and adjoints."""
        assert len(current_st_list) == len(self._st_obs_list)
        misfits = []
        adjoints = []
        for st_obs, st_syn in zip(self._st_obs_list, current_st_list):
            misfit = self._misfit(st_obs, st_syn)
            misfits.append(np.array(list(misfit.calc_misfit().values())))
            adjoints.append(misfit.get_adjoint_sources())
        return misfits, adjoints

    def _preprocess_stream(self, st):
        """Pre-process the stream"""
        if not callable(self._stream_pre_process):
            return st.detrend('linear').taper(0.5)
        else:
            return self._stream_pre_process(st)

    @property
    @cache
    def _st_obs_list(self):
        st_obs_list = [
            self._preprocess_stream(x) for x in
            _get_streams_from_each_source_dir(
                self.working_path / "observed_streams"
            )
        ]
        return st_obs_list