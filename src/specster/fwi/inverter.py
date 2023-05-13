"""
Class for inverting material properties.
"""
import pickle
import shutil
import time
from functools import cache, partial, reduce
from operator import add
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import specster as sp
from specster.core.misc import parallel_call
from specster.core.models import SpecsterModel
from specster.core.optimize import golden_section_search
from specster.core.parse import read_ascii_stream
from specster.core.plotting import plot_gll_data
from specster.core.preconditioner import median_filter, smooth
from specster.core.printer import console, program_render
from specster.exceptions import FailedLineSearch

from ..core.control import BaseControl
from .misfit import BaseMisfit


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


def _parabola(x, a, b):
    # Curve fitting function
    return a * x**2 + b


class IterationResults(SpecsterModel):
    """A simple model for storing results from each iteration."""

    iteration: int
    data_misfit: float
    model_misfit: Dict[str, float]
    gradient_scalar: float
    ls_lambda: float


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
    _scratch_path = "SCRATCH"
    _iteration_kernel_file_name = "iteration_kernel.parquet"
    _max_iteration_change = 0.04
    _linesearch_points = 6
    _material_model_file_name = "materials.parquet"
    _kernel_to_material_map = {"alpha": "vp", "beta": "vs", "rho": "rho", "c": "vp"}

    def __init__(
        self,
        observed_data_path: Union[Path, BaseControl],
        control: BaseControl,
        misfit: BaseMisfit,
        true_control: Optional[BaseControl] = None,
        optimization: Literal["steepest descent"] = "steepest descent",
        working_path="specster_scratch",
        kernels=("alpha", "beta"),
        hessian_preconditioning=True,
        smoothing_sigma=None,
    ):
        # set input params
        if isinstance(observed_data_path, BaseControl):
            observed_data_path = observed_data_path.each_source_path

        self._misfit = misfit
        self._optimization = optimization
        self.working_path = Path(working_path)
        self._control, self._true_control = self._create_working_directory(
            observed_data_path,
            control,
            true_control,
        )
        self._kernel_names = kernels
        self.iteration_results = []
        self.smoothing_sigma = smoothing_sigma
        self.save_checkpoint()
        self._hessian_preconditioning = hessian_preconditioning

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
        base_path = path.parent
        out: Inverter = pd.read_pickle(path)
        out.working_path = path.parent
        # need to fix paths in case this was copied from elsewhere.
        out._control = sp.Control2d(base_path / out._control_path)
        if (base_path / out._control_true_path).exists():
            out._true_control = sp.Control2d(base_path / out._control_true_path)
        return out

    def run_iteration(self, no_update=False):
        """
        Run the inversion iteratively.
        """
        iteration = len(self.iteration_results) + 1
        iteration_str = f"Iteration {iteration:d}"
        with program_render(console, title=f"FWI {iteration_str}", supress_output=True):
            # get the step size the model is allowed to change. This can be
            # the max value or twice the previous update value.
            if len(self.iteration_results):
                best_lambda = self.iteration_results[-1].ls_lambda * 2
            else:
                best_lambda = self._max_iteration_change
            start_time = time.time()
            console.rule(f"[bold red]Running FWI ({self.working_path}) {iteration_str}")
            controls = self.get_controls()  # controls for each event
            console.print("Running forward FWI mode")
            self._prep_controls_forward_use_model(controls)
            self._run_controls(controls)
            # get adjoints for current data, update misfit and write adjoints.
            c_streams = self._get_current_event_streams()
            misfit, adjoints = self._calc_misfit_adjoints(c_streams)
            console.print(f"Data misfit is {misfit}")
            self._write_adjoints_to_each_event(adjoints, control_list=controls)
            # run FWI, get aggregated kernels
            console.print(f"Running adjoint simulation for {len(controls)} events")
            self._run_controls(control_list=controls)
            console.print(f"Summing kernels for {len(controls)} events")
            kernels = self._aggregate_kernels(controls)
            self._save_iteration_kernels(kernels, iteration)
            console.print("Conducting line search to find gradient scaling")
            current_material_df = self._control.get_material_model_df()
            self._write_material_df(current_material_df, iteration)
            if not no_update:
                grad_scale, new_model, best_lambda = self._line_search(
                    kernels,
                    current_material_df,
                    misfit,
                    controls=controls,
                    console=console,
                    upper_bounds=best_lambda,
                )
            else:
                grad_scale, new_model, best_lambda = np.NaN, current_material_df, np.NaN
            maybe_model_misfit = self._calc_model_misfit(new_model)
            self._broadcast_model_updates(new_model)

            result = IterationResults(
                iteration=iteration,
                data_misfit=misfit,
                model_misfit=maybe_model_misfit,
                gradient_scalar=grad_scale,
                ls_lambda=best_lambda,
            )
            self.iteration_results.append(result)
            self.save_checkpoint()
            duration = time.time() - start_time
            console.rule(f"Finished iteration in {duration:.02f} seconds")
        return self

    def _line_search(
        self,
        kernels,
        material_df,
        current_misfit,
        controls=None,
        console=None,
        upper_bounds=None,
    ):
        """Find the optimal update for line search."""

        def _update(trial_lambda):
            """Calcualte an update to the model."""
            delta = -normed_kernels * model_norm * trial_lambda
            new_model = m_df + delta.values
            new_model["proc"] = procs.values
            self._broadcast_model_updates(new_model, controls, include_base=False)
            self._prep_controls_forward_use_model(controls)
            self._run_controls(control_list=controls)
            streams = self._get_current_event_streams()
            misfit, _ = self._calc_misfit_adjoints(streams, include_adjoint=False)
            misfits[trial_lambda] = misfit
            if console:
                console.print(
                    f"--- trial with lambda {trial_lambda} return misfit {misfit}"
                )
            return misfit

        controls = controls or self.get_controls()
        procs = material_df["proc"]
        k_df = kernels.rename(columns=self._kernel_to_material_map)
        m_df = material_df.loc[:, list(k_df.columns)]
        model_norm = m_df.max()
        kernel_norm = k_df.max()
        normed_kernels = k_df / kernel_norm
        misfits = {}
        best_lambda = golden_section_search(
            _update,
            0,
            upper_bounds or self._max_iteration_change,
            max_iter=self._linesearch_points,
        )
        misfits = pd.Series(misfits)
        if best_lambda <= 0 or misfits.min() > current_misfit:
            msg = "failed line search!"
            raise FailedLineSearch(msg)

        gradient_scalar = -best_lambda * model_norm
        final_delta = -normed_kernels * model_norm * best_lambda
        new_model = m_df + final_delta.values
        new_model["proc"] = procs.values
        return gradient_scalar, new_model, best_lambda

    def _prep_controls_forward_use_model(self, controls=None):
        """Prepare all the controls for a forward run using model databases."""
        controls = controls or self.get_controls()
        for control in controls:
            control.clear_output_traces()
            control.prepare_fwi_forward(use_binary_model=True)
            control.write(overwrite=True)

    def _broadcast_model_updates(self, model, controls=None, include_base=True):
        """Broadcast model updates to all event directories."""
        controls = controls or self.get_controls()
        for control in controls:
            control.set_material_model_df(model)
        if include_base:
            self._control.set_material_model_df(model)

    def _save_iteration_kernels(self, kernel: pd.DataFrame, iteration):
        """Save the kernels to disk in the appropriate iteration."""
        it_dir = self._get_iteration_directory(iteration)
        path = it_dir / self._iteration_kernel_file_name
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
        out = reduce(add, out.values())
        return out.fillna(0.0)

    def apply_kernel_conditioning(self, kernel_df):
        """
        Apply preprocessing to the kernels.

        By default this is just multiplying the hessian, then smoothing
        around the stations.
        """
        station_df = self._control.get_station_df()
        new = pd.DataFrame(index=kernel_df.index)
        if self._hessian_preconditioning:
            precon = kernel_df["Hessian1"].values + kernel_df["Hessian2"].values
        else:
            precon = np.ones(len(kernel_df))
        for kname in self._kernel_names:
            precon_kernel = kernel_df[kname] / precon
            new[kname] = precon_kernel
        out = new.pipe(median_filter, station_df=station_df)
        if self.smoothing_sigma:
            out = smooth(out)
        return out

    def _write_adjoints_to_each_event(self, adjoints, control_list=None):
        """Write the adjoints to disk."""
        controls = control_list or self.get_controls()
        for control, adjoint in zip(controls, adjoints):
            control.prepare_fwi_adjoint().write_adjoint_sources(adjoint)

    def _get_current_event_streams(self):
        current_streams = [x.get_waveforms() for x in self._control.each_source_output]
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
        misfiter = self._misfit.copy()
        for st_obs, st_syn in zip(self._st_obs_list, current_st_list):
            misfit = misfiter.get_misfit(st_obs, st_syn)
            misfits.append(misfit.values)
            if include_adjoint:
                adjoints.append(misfiter.get_adjoint_sources())
        misfit_total_array = np.hstack(misfits)
        return np.linalg.norm(misfit_total_array), adjoints

    @property
    def iteration_data_path(self):
        """Get the path to iteration data"""
        path = self.working_path / self._iteration_dir
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    @cache
    def _st_obs_list(self):
        path = self.working_path / self._observed_path
        st_obs_list = [x for x in _get_streams_from_each_source_dir(path)]
        return st_obs_list

    def _write_material_df(self, current_material_df, iteration):
        base_path = self._get_iteration_directory(iteration)
        save_path = base_path / self._material_model_file_name
        current_material_df.to_parquet(save_path)

    def _calc_model_misfit(self, new_model):
        """Calc L2 norm for current and true model."""
        true_mod = self._true_model
        if true_mod is None:
            return {}
        sub = true_mod[new_model.columns]
        l2_norm = np.linalg.norm(sub.values - new_model.values, axis=0)
        out = {col: norm for col, norm in zip(new_model.columns, l2_norm)}
        out.pop("proc", None)
        return out

    def plot_model_update(self, iteration):
        """Plot the model updates for a specific iteration."""
        base_path = self._get_iteration_directory(iteration)
        path = base_path / self._iteration_kernel_file_name
        assert path.exists()
        df = pd.read_parquet(path)
        return plot_gll_data(df)

    def plot_model(self, iteration):
        """Plot the model before a certain iteration."""
        base_path = self._get_iteration_directory(iteration)
        path = base_path / self._material_model_file_name
        assert path.exists()
        df = pd.read_parquet(path)
        return plot_gll_data(df)

    def plot_data_misfit(self):
        """Plot data misfit through the inversion."""
        fig, ax = plt.subplots(1, 1)
        iterations = range(len(self.iteration_results))
        misfit = [x.data_misfit for x in iterations]
        ax.plot(iterations, misfit)
        ax.set_xlabel("iterations")
        ax.set_ylabel("data misfit")

    def plot_model_misfit(self):
        """Make a plot of the misfit of the model through iterations."""
        # fig, ax = plt.subplots(1, 1)
        # iterations = range(len(self.iteration_results))
        # df = pd.DataFrame([dict(x.data_misfit) for x in self.iteration_results])
