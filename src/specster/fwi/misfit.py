"""
Modules for storing various misfit functions.
"""
import abc
from functools import cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from matplotlib.lines import Line2D
from obsplus.utils.time import to_utc
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy.integrate import simps

from specster.core.misc import get_stream_summary_df
from specster.exceptions import UnsetStreamsError


class _BaseMisfit(abc.ABC):
    _component_colors = {"Z": "orange", "X": "cyan", "Y": "Red"}
    window_df: Optional[pd.DataFrame] = None
    waveform_df_: Optional[pd.DataFrame] = None
    _default_taper = 0.95

    def validate_streams(self, st_obs, st_synthetic):
        """Custom validation for streams."""
        assert len(st_obs) == len(st_synthetic)
        for tr1, tr2 in zip(st_obs, st_synthetic):
            assert tr1.id == tr2.id
            assert tr1.stats.sampling_rate == tr2.stats.sampling_rate

    def _validate_stream_dfs(self, df_obs, df_synth):
        """Ensure the data in the streams are compatible."""
        assert (df_obs["seed_id"] == df_synth["seed_id"]).all()
        assert (df_obs["sampling_rate"] == df_synth["sampling_rate"]).all()

    def _get_overlap_df(self, df_obs, df_synth):
        """
        Get overlap dataframe.

        This is a dataframe with data for both streams.
        """
        sc, ec = "starttime", "endtime"
        starts = np.stack([df_obs[sc].values, df_synth[sc].values]).T
        ends = np.stack([df_obs[ec].values, df_synth[ec].values]).T
        out = df_obs.copy()
        out["starttime"] = np.max(starts, axis=1)
        out["endtime"] = np.max(ends, axis=1)
        return out

    def _set_traces_in_overlap_df(self, overlap_df, st_obs, st_synth):
        """Set the traces as columns in the overlap dataframe."""
        start = to_utc(overlap_df["starttime"].values)
        end = to_utc(overlap_df["endtime"].values)
        out_tr_obs = []
        out_tr_synth = []
        for ind in range(len(start)):
            t1, t2 = start[ind], end[ind]
            tr_obs = st_obs[ind].trim(starttime=t1, endtime=t2)
            tr_synth = st_synth[ind].trim(starttime=t1, endtime=t2)
            out_tr_obs.append(self.preprocess_trace(tr_obs))
            out_tr_synth.append(self.preprocess_trace(tr_synth))
        overlap_df["tr_obs"] = out_tr_obs
        overlap_df["tr_synth"] = out_tr_synth
        return overlap_df

    def _maybe_set_waveform_df(self, st_obs, st_synth):
        """
        Sets the dataframe containing data.
        """
        if st_obs is None or st_synth is None:
            if self.waveform_df_ is not None:
                return self.waveform_df_
            msg = "streams must be provided if waveform_df not yet set!"
            raise UnsetStreamsError(msg)
        # validate data contents.
        df_obs = get_stream_summary_df(st_obs)
        df_synth = get_stream_summary_df(st_synth)
        self._validate_stream_dfs(df_obs, df_synth)
        out = self._get_overlap_df(df_obs, df_synth)
        if self.window_df is not None:
            pass
            # out = get_window(self.window_df, out)
        else:
            out = self._set_traces_in_overlap_df(out, st_obs, st_synth)
        self.waveform_df_ = out

    def preprocess_trace(self, tr):
        """Function for pre-processing traces."""
        return tr.detrend("linear").taper(self._default_taper)

    def iterate_streams(self):
        """
        Yield streams for observed and synthetic data.

        First validates the streams then sets the waveform_df which will
        specify which parts of the stream are used.

        """
        obs_list = self.waveform_df_["tr_obs"].values
        synth_list = self.waveform_df_["tr_synth"].values
        for tr_obs, tr_synth in zip(obs_list, synth_list):
            yield tr_obs, tr_synth

    @abc.abstractmethod
    def calc_misfit(self, tr_obs, tr_synth) -> dict[str, float]:
        """Calculate the misfit between observed and synthetic traces."""

    @abc.abstractmethod
    def calc_adjoint(self, tr_obs, tr_synth) -> dict[str, float]:
        """Calculate the adjoint source between observed and synthetic traces."""

    def plot(self, station=None, out_file=None):
        """Create a plot of observed/synthetic."""

        def add_legends(ax):
            """Add the legends for component and synth/observed."""
            line1 = Line2D([0], [0], color="0.5", ls="--", label="predicted")
            line2 = Line2D([0], [0], color="0.5", ls="-", label="observed")

            # Create a legend for the first line.
            leg1 = ax.legend(handles=[line1, line2], loc="upper right")
            ax.add_artist(leg1)

            color_lines = [
                Line2D(
                    [0],
                    [0],
                    color=self._component_colors[x],
                    ls="-",
                    label=f"{x} component",
                )
                for x in self._component_colors
            ]
            ax.legend(handles=color_lines, loc="upper left")

        def maybe_save(fig, out_file):
            """Maybe save the figure."""
            if out_file is None:
                return
            plt.tight_layout()
            fig.savefig(out_file)
            plt.close("all")

        fig, (wf_ax, ad_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

        unique_stations = {tr.stats.station for tr in self.st_obs}
        station = list(unique_stations)[0] if station is None else station

        adjoint = self.get_adjoint_sources()

        for tr_obs, tr_synth in self.iterate_streams():
            if tr_obs.stats.station != station:
                continue
            ad_tr = adjoint[tr_obs.id]
            # make plots of observed/synthetics
            color = self._component_colors[tr_obs.stats.component]
            wf_ax.plot(tr_obs.times(), tr_obs.data, "-", color=color)
            wf_ax.plot(tr_synth.times(), tr_synth.data, "--", color=color)
            add_legends(wf_ax)
            ad_ax.plot(ad_tr.times(), ad_tr.data, "-", color=color)

        wf_ax.set_title("Waveforms")
        ad_ax.set_title("Adjoint Source")

        ad_ax.set_xlabel("Time (s)")
        fig.supylabel("Displacement (m)")

        maybe_save(fig, out_file)

        return fig, (wf_ax, ad_ax)

    def get_misfit(self, st_obs=None, st_synth=None):
        """Calculate the misfit between streams."""
        self._maybe_set_waveform_df(st_obs=st_obs, st_synth=st_synth)
        out = []
        for tr_obs, tr_synth in self.iterate_streams():
            out.append(self.calc_misfit(tr_obs, tr_synth))
        out = pd.Series(out, index=self.waveform_df_.index)
        return out

    def get_adjoint_sources(self, st_obs=None, st_synth=None) -> obspy.Stream:
        """Return the adjoint source trace."""
        self._maybe_set_waveform_df(st_obs=st_obs, st_synth=st_synth)
        out = []
        for tr_obs, tr_synth in self.iterate_streams():
            out.append(self.calc_adjoint(tr_obs, tr_synth))
        return self._assemble_output_stream(out)

    def _assemble_output_stream(self, adjoint_list):
        """Return a stream with all the traces put back together."""


class WaveformMisfit(_BaseMisfit):
    """
    Manager to calculate misfit and ajoints for waveform misfit.

    Parameters
    ----------
    observed_st
        The observed stream
    synthetic_st
        The calculated stream
    """

    def calc_misfit(self, tr_obs, tr_synth):
        """Calculate the misfit between streams."""
        dx = tr_obs.stats.delta
        misfit = simps((tr_synth.data - tr_obs.data) ** 2, dx=dx)
        return misfit

    def calc_adjoint(self, tr_obs, tr_synth):
        """Return the adjoint source trace."""
        new = tr_obs.copy()
        new.data = tr_synth.data - tr_obs.data
        return new


class TravelTimeMisfit(_BaseMisfit):
    """
    Manager to calculate misfit and ajoints for traveltime misfit.

    Parameters
    ----------
    observed_st
        The directory containing the observed data.
    synthetic_st
        The directory containing the synthetic data.
    trace_min
        A ratio to zero-out very low amplitude traces
        to avoid numerical noise. If the max value of a trace
        divided by the max value of a maximum trace is less than
        this ratio, zero its adjoint.
    """

    def __init__(self, observed_st, synthetic_st, trace_min=0.01):
        """Read in the streams for each directory."""
        self._trace_min = trace_min
        super().__init__(observed_st, synthetic_st)

    @cache
    def calc_trace_absmax(self):
        """Get a dict of the absolute max of a trace."""
        out = {}
        for tr_obs, tr_synth in self.iterate_streams():
            out[tr_obs.id] = {
                "observed": np.abs(tr_obs.data).max(),
                "synthetic": np.abs(tr_synth.data).max(),
            }
        return pd.DataFrame(out).T

    @cache
    def calc_normalization(self):
        """Calculate the normalization waveforms."""
        out = {}
        for _, tr_synth in self.iterate_streams():
            dt = 1 / tr_synth.stats.sampling_rate
            double_t_diff = tr_synth.copy().differentiate().differentiate()
            norm = double_t_diff.data * tr_synth.data
            norm_sum = simps(norm, dx=dt)
            print(norm_sum)
            out[tr_synth.id] = norm_sum

        return out

    @cache
    def calc_tt_diff(self) -> dict:
        """Calculate the travel time differences"""
        out = {}
        for tr_obs, tr_synth in self.iterate_streams():
            cor = correlate(tr_obs, tr_synth, 100)
            shift, value = xcorr_max(cor)
            values = shift / tr_obs.stats.sampling_rate
            out[tr_obs.id] = values
        return out

    @cache
    def calc_misfit(self):
        """Calculate the misfit between streams."""
        return {i: v**2 for i, v in self.calc_tt_diff().items()}

    @cache
    def get_adjoint_sources(self):
        """Return the adjoint source trace."""
        out = []
        tt_diffs = self.calc_tt_diff()
        norms = self.calc_normalization()
        for tr_obs, tr_synth in self.iterate_streams():
            out_tr = tr_synth.copy()
            dt = 1 / tr_synth.stats.sampling_rate
            tid = tr_synth.id
            diff = tt_diffs[tr_synth.id]
            norm = 1 / norms[tr_synth.id]
            tdff = np.gradient(tr_synth.data, dt)
            data = -diff * norm * tdff
            if self._should_zero_trace(tid):
                data = np.zeros_like(data)
            out_tr.data = data
            out.append(out_tr)
        return obspy.Stream(out)

    def _should_zero_trace(self, tid):
        """Return true if data should be zeroes"""
        df = self.calc_trace_absmax()
        max_vals = df.max()
        current = df.loc[tid]
        return ((current / max_vals) < self._trace_min).any()


class AmplitudeMisfit(_BaseMisfit):
    """
    Manager to calculate misfit and ajoints for amplitude misfit.

    Parameters
    ----------
    observed_st
        The directory containing the observed data.
    synthetic_st
        The directory containing the synthetic data.
    """

    def __init__(self, observed_st, synthetic_st, trace_min=0.01):
        """Read in the streams for each directory."""
        self._trace_min = trace_min
        super().__init__(observed_st, synthetic_st)

    @cache
    def calc_trace_absmax(self):
        """Get a dict of the absolute max of a trace."""
        out = {}
        for tr_obs, tr_synth in self.iterate_streams():
            out[tr_obs.id] = {
                "observed": np.abs(tr_obs.data).max(),
                "synthetic": np.abs(tr_synth.data).max(),
            }
        return pd.DataFrame(out).T

    @cache
    def calc_normalization(self):
        """Calculate the normalization waveforms."""
        out = {}
        for _, tr_synth in self.iterate_streams():
            data = tr_synth.data**2
            tid = tr_synth.id
            out[tid] = simps(data, dx=1 / tr_synth.stats.sampling_rate)
        return out

    @cache
    def calc_amp_ratio(self) -> dict:
        """Calculate the travel time differences"""
        out = {}
        for tr_obs, tr_synth in self.iterate_streams():
            rms_obs = np.sqrt(np.mean(tr_obs.data**2))
            rms_synth = np.sqrt(np.mean(tr_synth.data**2))
            out[tr_obs.id] = np.log(rms_obs / rms_synth)
        return out

    @cache
    def calc_misfit(self):
        """Calculate the misfit between streams."""
        return {i: v**2 for i, v in self.calc_amp_ratio().items()}

    @cache
    def get_adjoint_sources(self):
        """Return the adjoint source trace."""
        out = {}
        amp_ratios = self.calc_amp_ratio()
        norms = self.calc_normalization()
        for tr_obs, tr_synth in self.iterate_streams():
            out_tr = tr_synth.copy()
            norm = 1 / norms[tr_obs.id]
            amp = amp_ratios[tr_obs.id]
            synth_data = tr_synth.data
            data = -norm * amp * synth_data
            if self._should_zero_trace(tr_obs.id):
                data = np.zeros_like(data)
            out_tr.data = data
            out[out_tr.id] = out_tr
        return out

    def _should_zero_trace(self, tid):
        """Return true if data should be zeroes"""
        df = self.calc_trace_absmax()
        max_vals = df.max()
        current = df.loc[tid]
        return ((current / max_vals) < self._trace_min).any()
