"""
Modules for storing various misfit functions.
"""
import abc
import copy
from typing import Optional

import numpy as np
import obspy
import pandas as pd
from obsplus.utils.time import to_utc
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy.integrate import simps

from specster.core.misc import get_stream_summary_df
from specster.core.plotting import plot_misfit
from specster.exceptions import UnsetStreamsError


class BaseMisfit(abc.ABC):
    """An abstract base class for misfit functions."""

    window_df: Optional[pd.DataFrame] = None
    waveform_df_: Optional[pd.DataFrame] = None
    synth_df_: Optional[pd.DataFrame] = None
    taper_percentage = 0.05
    normalize_traces = False

    def __init__(self, window_df=None, normalize=False):
        self.window_df = window_df
        self.normalize_traces = normalize

    # --- methods which must be implemented in subclasses

    @abc.abstractmethod
    def calc_misfit(self, tr_obs, tr_synth) -> dict[str, float]:
        """Calculate the misfit between observed and synthetic traces."""

    @abc.abstractmethod
    def calc_adjoint(self, tr_obs, tr_synth) -> dict[str, float]:
        """Calculate the adjoint source between observed and synthetic traces."""

    # --- methods which may be implemented in subclasses

    def preprocess_trace(self, tr):
        """Function for pre-processing traces."""
        out = tr.detrend("linear").taper(self.taper_percentage)
        if self.normalize_traces:
            out = out.normalize_traces()
        return out

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
            tr = self.calc_adjoint(tr_obs, tr_synth)
            if tr is not None:
                out.append(tr)
        return self._assemble_output_stream(out)

    def new_waveforms_set(self):
        """Called when new waveforms are set. Can be implemented in subclass."""

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
        self.synth_df_ = df_synth
        self._validate_stream_dfs(df_obs, df_synth)
        out = self._get_overlap_df(df_obs, df_synth)
        if self.window_df is not None:
            out = self._get_window_df(self.window_df, out, st_obs, st_synth)
        else:
            out = self._set_traces_in_overlap_df(out, st_obs, st_synth)
        self.waveform_df_ = out
        self.new_waveforms_set()

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

    def _assemble_output_stream(self, adjoint_list):
        """Return a stream with all the traces put back together."""
        # we need traces with the same stats as synthetic ones.
        base_trace_dict = self._empty_trace_dict_from_df(self.synth_df_)
        for tr in adjoint_list:
            self._add_traces(base_trace_dict[tr.id], tr)
        return obspy.Stream(list(base_trace_dict.values()))

    def _add_traces(self, base, tr):
        """Add trace to a"""
        assert len(tr) <= len(base)
        assert tr.stats.starttime >= base.stats.starttime
        assert base.stats.sampling_rate == tr.stats.sampling_rate
        sr = base.stats.sampling_rate
        start_ind = int((tr.stats.starttime - base.stats.starttime) * sr)
        base.data[start_ind : start_ind + len(tr)] += tr.data

    def _empty_trace_dict_from_df(self, df):
        """Create a dict of unique seed ids: empty trace from a dataframe."""
        assert df["seed_id"].unique().all()
        df = df.assign(starttime=to_utc(df["starttime"]), endtime=to_utc(df["endtime"]))
        out = {}
        for inf in df.to_dict("records"):
            shape = (inf["endtime"] - inf["starttime"]) * inf["sampling_rate"]
            zeros = np.zeros(int(np.ceil(shape)) + 1)
            trace = obspy.Trace(data=zeros, header=inf)
            out[trace.id] = trace
        return out

    def _get_window_df(self, window_df, avail_df, st_obs, st_synth):
        """Get a dataframe of traces for each window selected time."""
        adf = avail_df.set_index("seed_id")
        st_obs_dict = {tr.id: tr for tr in st_obs}
        st_syn_dict = {tr.id: tr for tr in st_synth}
        out = []
        for info in window_df.to_dict(orient="records"):
            seed = info["seed_id"]
            start = to_utc(max([info["starttime"], adf.loc[seed, "starttime"]]))
            end = to_utc(min([info["endtime"], adf.loc[seed, "endtime"]]))
            tr_obs = st_obs_dict[seed].slice(starttime=start, endtime=end)
            tr_synth = st_syn_dict[seed].slice(starttime=start, endtime=end)
            info.update(
                dict(
                    starttime=start,
                    endtime=end,
                    tr_obs=self.preprocess_trace(tr_obs),
                    tr_synth=self.preprocess_trace(tr_synth),
                )
            )
            out.append(info)
        return pd.DataFrame(out)

    plot = plot_misfit

    def copy(self):
        """Copy the misfit and return it."""
        return copy.deepcopy(self)


class WaveformMisfit(BaseMisfit):
    """
    Manager to calculate misfit and ajoints for waveform misfit.
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


class TravelTimeMisfit(BaseMisfit):
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

    _abs_max = np.NaN

    def __init__(self, trace_min=0.01, **kwargs):
        """Read in the streams for each directory."""
        self._trace_min = trace_min
        super().__init__(**kwargs)

    def new_waveforms_set(self):
        """
        Update the maximum amplitude to which the traces are compared for
        potential zeroing.
        """
        wdf = self.waveform_df_
        max_tr_obs = [np.max(np.abs(tr)) for tr in wdf["tr_obs"]]
        max_tr_syn = [np.max(np.abs(tr)) for tr in wdf["tr_synth"]]
        self._abs_max = np.max(max_tr_obs + max_tr_syn)

    def calc_normalization(self, tr_synth):
        """Calculate the normalization waveforms."""
        dt = 1 / tr_synth.stats.sampling_rate
        double_t_diff = tr_synth.copy().differentiate().differentiate()
        norm = double_t_diff.data * tr_synth.data
        norm_sum = simps(norm, dx=dt)
        return norm_sum

    def calc_tt_diff(self, tr_obs, tr_synth):
        """Calculate the travel time differences"""
        cor = correlate(tr_obs, tr_synth, 100)
        shift, value = xcorr_max(cor)
        values = shift / tr_obs.stats.sampling_rate
        return values

    def calc_misfit(self, tr_obs, tr_synth):
        """Calculate the misfit between streams."""
        diff = self.calc_tt_diff(tr_obs, tr_synth)
        return diff**2

    def calc_trace_amps(self):
        """Caclualte trace amplitudes."""

    def calc_adjoint(self, tr_obs, tr_synth):
        """Return the adjoint source trace."""
        out_tr = tr_synth.copy()
        dt = 1 / tr_synth.stats.sampling_rate
        diff = self.calc_tt_diff(tr_obs, tr_synth)
        norm = 1 / self.calc_normalization(tr_synth)
        tdff = np.gradient(tr_synth.data, dt)
        data = -diff * norm * tdff
        if self._should_zero_trace(data):
            data = np.zeros_like(data)
        out_tr.data = data
        return out_tr

    def _should_zero_trace(self, data):
        """Return true if data should be zeroes"""
        if np.max(np.abs(data)) < self._abs_max:
            return True
        return False


class AmplitudeMisfit(TravelTimeMisfit):
    """
    Manager to calculate misfit and ajoints for amplitude misfit.
    """

    def calc_normalization(self, tr_synth):
        """Calculate the normalization waveforms."""
        data = tr_synth.data**2
        return simps(data, dx=1 / tr_synth.stats.sampling_rate)

    def calc_amp_ratio(self, tr_obs, tr_synth) -> float:
        """Calculate the travel time differences"""
        rms_obs = np.sqrt(np.mean(tr_obs.data**2))
        rms_synth = np.sqrt(np.mean(tr_synth.data**2))
        return np.log(rms_obs / rms_synth)

    def calc_misfit(self, tr_obs, tr_synth):
        """Calculate the misfit between streams."""
        out = self.calc_amp_ratio(tr_obs, tr_synth)
        return out**2

    def calc_adjoint(self, tr_obs, tr_synth):
        """Return the adjoint source trace."""
        out_tr = tr_synth.copy()
        norm = 1 / self.calc_normalization(tr_synth)
        amp = self.calc_amp_ratio(tr_obs, tr_synth)
        synth_data = tr_synth.data
        data = -norm * amp * synth_data
        if self._should_zero_trace(data):
            data = np.zeros_like(data)
        out_tr.data = data
        return out_tr
