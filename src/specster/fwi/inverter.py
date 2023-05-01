"""
Class for inverting material properties.
"""
import pickle
import shutil
from functools import cache, partial, reduce
from operator import add
from pathlib import Path
from typing import List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd

import specster as sp
from specster.core.misc import parallel_call
from specster.core.parse import read_ascii_stream
from specster.core.printer import console, program_render

from ..core.control import BaseControl
from .misfit import _BaseMisFit


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

    _version = "0.0.0"
    _observed_path = "OBSERVED_STREAMS"
    _iteration_dir = "ITERATIONS"
    _control_path = "CONTROL"
    _control_true_path = "CONTROL_TRUE"
    _inverter_save_path = "inverter.pkl"
    _stats_columns = ("data_misfit", "model_misfit", "kernel_step")
    _scratch_path = "SCRATCH"
    _max_iteration_change = 0.02
    _kernel_to_material_map = {"alpha": "vp", "beta": "vs", "rho": "rho"}

    def __init__(
        self,
        observed_data_path: Union[Path, BaseControl],
        control: BaseControl,
        misfit: Type[_BaseMisFit],
        true_control: Optional[BaseControl] = None,
        optimization: Literal["steepest descent"] = "steepest descent",
        stream_pre_processing=None,
        working_path="specster_scratch",
        pre_conditioner: Literal["default"] = "default",
        misfit_aggregator=np.linalg.norm,
        kernels=("alpha", "beta"),
    ):
        # set input params
        if isinstance(observed_data_path, BaseControl):
            observed_data_path = observed_data_path.each_source_path

        self._misfit = misfit
        self._optimization = optimization
        self._stream_pre_process = stream_pre_processing
        self.working_path = Path(working_path)
        self._control, self._true_control = self._create_working_directory(
            observed_data_path,
            control,
            true_control,
        )
        self._df = pd.DataFrame(columns=list(self._stats_columns))
        self._misfit_aggregator = misfit_aggregator
        self._kernel_names = kernels

    def _create_working_directory(
        self,
        observed_data_path,
        control_initial,
        control_true,
    ):
        """
        Create a working directory with observed/synthetic data and
        copy true and initial control data.
        """
        # copy observed data
        obs_data_path = self.working_path / self._observed_path
        if not obs_data_path.exists():
            waveform_paths = sorted(observed_data_path.rglob("*semd"))
            assert len(waveform_paths), f"No waveforms in {observed_data_path}"
            for waveform_path in waveform_paths:
                new_path = obs_data_path / waveform_path.relative_to(observed_data_path)
                new_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(waveform_path, new_path)
        control = control_initial.copy(self.working_path / self._control_path)
        _run_each_source(control)
        if control_true is not None:
            control_true = control_true.copy(
                self.working_path / self._control_true_path
            )
        return control, control_true

    def _run_controls(self, control_list=None):
        """Run one forward iteration."""
        controls = control_list or self.get_controls()
        run_list = [partial(x.run, supress_output=True) for x in controls]
        parallel_call(run_list)

    def save_checkpoint(self):
        """Pickle the inverter into the working directory"""
        # This is just a sloppy hack until I have time to work on
        # a better serialization format
        path = self.working_path / self._inverter_save_path
        with open(path, "wb") as fi:
            pickle.dump(self, fi)

    @property
    @cache
    def _true_model(self):
        if self._true_control:
            return self._true_control.get_material_model_df()
        return None

    @classmethod
    def load_inverter(cls, path):
        """Load the pickled inverter"""
        path = Path(path)
        name = cls._inverter_save_path
        path = path if name in path.name else path / name
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist!")
        out: Inverter = pd.read_pickle(path)
        base_path = path.parent
        # need to fix paths in case this was copied from elsewhere.
        out._control = sp.Control2d(base_path / out._control_path)
        if (base_path / out._control_true_path).exists():
            out._true_control = sp.Control2d(base_path / out._control_true_path)
        return out

    def run_iteration(self):
        """
        Run the inversion iteratively.
        """
        iteration = len(self._df) + 1
        iteration_str = f"Iteration {iteration:d}"
        with program_render(console, title=f"FWI {iteration_str}", supress_output=True):
            console.rule(f"[bold red]Running FWI ({self.working_path}) {iteration_str}")
            # add new row to tracking df.
            self._df = pd.concat(
                [self._df, pd.DataFrame(columns=list(self._stats_columns))]
            )
            controls = self.get_controls()  # controls for each event
            # get adjoints for current data, update misfit and write adjoints.
            current_streams = self._get_current_event_streams()
            misfit, adjoints = self._calc_misfit_adjoints(current_streams)
            console.print(f"Data misfit is {misfit}")
            self._df.loc[len(self._df) - 1, "data_misfit"] = misfit
            self._write_adjoints_to_each_event(adjoints, control_list=controls)
            # run FWI, get aggregated kernels
            console.print(f"Running adjoints for {len(controls)} events")
            self._run_controls(control_list=controls)
            console.print(f"Summing kernels for {len(controls)} events")
            kernels = self._aggregate_kernels(controls)
            self._save_iteration_kernels(kernels, iteration)
            console.print(f"Conducting line search to find gradient scaling")
            current_material_df = self._control.get_material_model_df()
            lam = self._linesearch_alpha(kernels, current_material_df)

    def _linesearch_alpha(self, kernels, material_df):
        """Find the optimal update for line search."""
        k_df = kernels.rename(columns=self._kernel_to_material_map)
        m_df = material_df.loc[:, k_df.columns]
        breakpoint()

    def _save_iteration_kernels(self, kernel: pd.DataFrame, iteration):
        """Save the kernels to disk in the appropriate iteration."""
        it_dir = self._get_iteration_directory(iteration)
        path = it_dir / "iteration_kernel.parquet"
        kernel.to_parquet(path)

    def _load_iteration_kernels(self, iteration):
        """Save the kernels to disk in the appropriate iteration."""
        it_dir = self._get_iteration_directory(iteration)
        path = it_dir / "iteration_kernel.parquet"
        return pd.read_parquet(path)

    def _aggregate_kernels(self, control_list=None):
        """
        Load each of the kernels, apply preconditioning, sum, and
        return.
        """
        controls = control_list or self.get_controls()
        out = {}
        for control in controls:
            kernels = control.output.load_kernel().pipe(self.apply_kernel_conditioning)
            out[control.base_path.name] = kernels
        return reduce(add, out.values())

    def apply_kernel_conditioning(self, kernel_df):
        """
        Apply preprocessing to the kernels.

        By default this is just multiplying the hessian.
        """
        new = pd.DataFrame(index=kernel_df.index)
        hess_1 = kernel_df["Hessian1"]
        for kname in self._kernel_names:
            new[kname] = kernel_df[kname] / hess_1
        return new

    def _write_adjoints_to_each_event(self, adjoints, control_list=None):
        """Write the adjoints to disk."""
        controls = control_list or self.get_controls()
        for control, adjoint in zip(controls, adjoints):
            control.prepare_fwi_adjoint().write_adjoint_sources(adjoint)

    def _get_current_event_streams(self):
        current_streams = [
            self._preprocess_stream(x.get_waveforms())
            for x in self._control.each_source_output
        ]
        return current_streams

    def get_controls(self) -> List[sp.Control2d]:
        """Get a control2d for each event."""
        out = []
        for path in sorted(self._control.each_source_path.iterdir()):
            out.append(sp.Control2d(path))
        return out

    def _get_iteration_directory(self, iteration):
        """Get the directory for saving info on each iteration."""
        it_dir = Path(self.working_path) / self._iteration_dir
        path = it_dir / f"{iteration:06d}"
        path.mkdir(exist_ok=True, parents=True)
        return path

    def _calc_misfit_adjoints(self, current_st_list, include_adjoint=True):
        """Calculate the misfit and adjoints."""
        assert len(current_st_list) == len(self._st_obs_list)
        misfits = []
        adjoints = []
        for st_obs, st_syn in zip(self._st_obs_list, current_st_list):
            misfit = self._misfit(st_obs, st_syn)
            misfits.append(np.array(list(misfit.calc_misfit().values())))
            if include_adjoint:
                adjoints.append(misfit.get_adjoint_sources())
        misfit_total_array = np.hstack(misfits)
        return self._misfit_aggregator(misfit_total_array), adjoints

    def _preprocess_stream(self, st):
        """Pre-process the stream"""
        if not callable(self._stream_pre_process):
            return st.detrend("linear").taper(0.5)
        else:
            return self._stream_pre_process(st)

    @property
    def iteration_data_path(self):
        """Get the path to iteration data"""
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
            self._preprocess_stream(x)
            for x in _get_streams_from_each_source_dir(
                self.working_path / self._observed_path
            )
        ]
        return st_obs_list
