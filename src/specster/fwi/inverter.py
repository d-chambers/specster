"""
Class for inverting material properties.
"""
import shutil
from typing import Optional, Literal, Type, Union, List
from pathlib import Path
from functools import cache

import numpy as np
import pandas as pd

import specster as sp
from ..core.control import BaseControl
from .misfit import _BaseMisFit
from specster.core.misc import parallel_call
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


def _run_controls(control: sp.Control2d):
    """helper function for mapping control run over many processes."""
    control.write()
    control.run()
    return control.base_path



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
    _observed_path = "OBSERVED_STREAMS"
    _iteration_dir = "ITERATIONS"
    _stats_columns = ("data_misfit, model_misfit, kernel_step")
    _scratch_path = "SCRATCH"
    _step_range = (0.001, 0.02)

    def __init__(
            self,
            observed_data_path: Union[Path, BaseControl],
            control: BaseControl,
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

        self._misfit = misfit
        self._true_control = true_control
        self._optimization = optimization
        self._stream_pre_process = stream_pre_processing
        self.working_path = Path(working_path)
        new_control = control.copy(self.working_path)
        self._control = _run_each_source(new_control)
        self._create_working_directory(observed_data_path)
        self._df = pd.DataFrame(columns=list(self._stats_columns))
        self._misfit_aggregator = misfit_aggregator
        self._true_model = self._maybe_load_true_model()

    def _create_working_directory(self, observed_data_path):
        """Create a working directory with observed/synthetic data."""
        # copy observed data
        obs_data_path = self.working_path / self._observed_path
        if not obs_data_path.is_dir():
            shutil.copytree(observed_data_path, obs_data_path)



    def run_inversion_iteration(self):

        """
        Run the inversion iteratively.
        """
        # add new row to tracking df.
        self._df = pd.concat([self._df, pd.DataFrame(columns=self._stats_columns)])

        # get adjoints for current data.
        current_streams = [
            self._preprocess_stream(x.get_waveforms())
            for x in self._control.each_source_output
        ]
        misfit, adjoints = self._calc_misfit_adjoints(current_streams)
        self._df.loc[len(self._df)-1, 'misfit'] = misfit

        sub_controls = self.get_controls()
        for control, adjoint in zip(sub_controls, adjoints):
            control.prepare_fwi_adjoint().write_adjoint_sources(adjoint)
        # run each
        breakpoint()
        parallel_call([x.run for x in sub_controls])
        control.run()
        output = control.output
        output.load_kernel()
        pp = get_executor()
        out = pp.map(_run_controls, sub_controls)
        #
        # if not len(initial_st):  # need to run initial control
        #     self.initial_control.run()
        #     initial_st = self.initial_control.output.get_waveforms()
        #     assert len(initial_st)
        #
        # initial_misfit = self._calc_misfit_adjoints(current_st_list=initial_st)
        return self


    def _save_adjoints(self, adjoints):
        """Save adjoints back to disk."""

    def get_controls(self) -> List[sp.Control2d]:
        """Get a control2d for each event."""
        out = []
        for path in sorted(self._control.each_source_path.iterdir()):
            out.append(sp.Control2d(path))
        return out

    def _calc_misfit_adjoints(self, current_st_list):
        """Calculate the misfit and adjoints."""
        assert len(current_st_list) == len(self._st_obs_list)
        misfits = []
        adjoints = []
        for st_obs, st_syn in zip(self._st_obs_list, current_st_list):
            misfit = self._misfit(st_obs, st_syn)
            misfits.append(np.array(list(misfit.calc_misfit().values())))
            adjoints.append(misfit.get_adjoint_sources())
        misfit_total_array = np.hstack(misfits)
        return self._misfit_aggregator(misfit_total_array), adjoints

    def _preprocess_stream(self, st):
        """Pre-process the stream"""
        if not callable(self._stream_pre_process):
            return st.detrend('linear').taper(0.5)
        else:
            return self._stream_pre_process(st)

    @property
    def iteration_path(self):
        path = self.working_path / self._iteration_dir
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def scratc_path(self):
        path = self.working_path / self._scratch_path
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    @cache
    def _st_obs_list(self):
        st_obs_list = [
            self._preprocess_stream(x) for x in
            _get_streams_from_each_source_dir(
                self.working_path / self._observed_path
            )
        ]
        return st_obs_list

    def _maybe_load_true_model(self):
        """If a true control is used, load model."""
        if self._true_control:
            self._true_control.load
        pass