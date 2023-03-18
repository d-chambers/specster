"""
Modules for storing various misfit functions.
"""
# import abc
# from functools import cache
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# import obspy
# import pandas as pd
# from matplotlib.lines import Line2D
# from obspy.signal.cross_correlation import correlate, xcorr_max
# from scipy.integrate import simps
#
#
# class _BaseMisFit(abc.ABC):
#     _component_colors = {"Z": "orange", "X": "cyan", "Y": "Red"}
#
#     def __init__(
#         self,
#         observed_path,
#         synthetic_path,
#     ):
#         """Read in the streams for each directory."""
#         self.st_obs = self.load_streams(observed_path)
#         self.st_synth = self.load_streams(synthetic_path)
#         self.validate_streams()
#
#     def preprocess(self, st):
#         """Preprocess the streams."""
#         # freq_min = local.bandwidth[0]
#         # freq_max = local.bandwidth[1]
#
#         out = (
#             st.detrend("linear")
#             .taper(0.05)
#             .filter("bandpass", freqmin=freq_min, freqmax=freq_max)
#         )
#         return out
#
#     def validate_streams(self):
#         """Custom validation for streams."""
#         st1 = self.st_obs
#         st2 = self.st_synth
#         assert len(st1) == len(st2)
#         assert {tr.id for tr in st1} == {tr.id for tr in st2}
#
#     def load_streams(self, directory_path):
#         """Load all streams in a directory."""
#         traces = []
#         for path in Path(directory_path).rglob("*semd*"):
#             traces.append(read_trace(str(path)))
#         st = self.preprocess(obspy.Stream(traces).sort())
#         return st
#
#     def iterate_streams(self):
#         """Iterate streams, yield corresponding traces for obs and synth"""
#         st_obs, st_synth = self.st_obs, self.st_synth
#         for tr_obs, tr_synth in zip(st_obs, st_synth):
#             assert tr_obs.id == tr_synth.id
#             assert tr_obs.stats.sampling_rate == tr_synth.stats.sampling_rate
#             yield tr_obs, tr_synth
#
#     def save_adjoint_sources(self, path="SEM"):
#         """Save adjoint sources to disk."""
#         path = Path(path)
#         path.mkdir(exist_ok=True, parents=True)
#         for tr in self.get_adjoint_sources().values():
#             stats = tr.stats
#             name = f"{stats.network}.{stats.station}.{stats.channel}.adj"
#             new_path = path / name
#             save_trace(tr, new_path)
#
#     @abc.abstractmethod
#     def calc_misfit(self) -> dict[str, float]:
#         """Calculate the misfit between streams."""
#
#     @abc.abstractmethod
#     def get_adjoint_sources(self) -> obspy.Stream:
#         """Return the adjoint source trace."""
#
#     def plot(self, station=None, out_file=None):
#         """Create a plot of observed/synthetic."""
#
#         def add_legends(ax):
#             """Add the legends for component and synth/observed."""
#             line1 = Line2D([0], [0], color="0.5", ls="--", label="predicted")
#             line2 = Line2D([0], [0], color="0.5", ls="-", label="observed")
#
#             # Create a legend for the first line.
#             leg1 = ax.legend(handles=[line1, line2], loc="upper right")
#             ax.add_artist(leg1)
#
#             color_lines = [
#                 Line2D(
#                     [0],
#                     [0],
#                     color=self._component_colors[x],
#                     ls="-",
#                     label=f"{x} component",
#                 )
#                 for x in self._component_colors
#             ]
#             ax.legend(handles=color_lines, loc="upper left")
#
#         def maybe_save(fig, out_file):
#             """Maybe save the figure."""
#             if out_file is None:
#                 return
#             plt.tight_layout()
#             fig.savefig(out_file)
#             plt.close("all")
#
#         fig, (wf_ax, ad_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
#
#         unique_stations = {tr.stats.station for tr in self.st_obs}
#         station = list(unique_stations)[0] if station is None else station
#
#         adjoint = self.get_adjoint_sources()
#
#         for tr_obs, tr_synth in self.iterate_streams():
#             if tr_obs.stats.station != station:
#                 continue
#             ad_tr = adjoint[tr_obs.id]
#             # make plots of observed/synthetics
#             color = self._component_colors[tr_obs.stats.component]
#             wf_ax.plot(tr_obs.times(), tr_obs.data, "-", color=color)
#             wf_ax.plot(tr_synth.times(), tr_synth.data, "--", color=color)
#             add_legends(wf_ax)
#             ad_ax.plot(ad_tr.times(), ad_tr.data, "-", color=color)
#
#         wf_ax.set_title("Waveforms")
#         ad_ax.set_title("Adjoint Source")
#
#         ad_ax.set_xlabel("Time (s)")
#         fig.supylabel("Displacement (m)")
#
#         maybe_save(fig, out_file)
#
#         return fig, (wf_ax, ad_ax)
#
#
# class WaveformMisFit(_BaseMisFit):
#     """
#     Manager to calculate misfit and ajoints for waveform misfit.
#
#     Parameters
#     ----------
#     observed_path
#         The directory containing the observed data.
#     synthetic_path
#         The directory containing the synthetic data.
#     """
#
#     @cache
#     def calc_misfit(self):
#         """Calculate the misfit between streams."""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             misfit = simps((tr_synth.data - tr_obs.data) ** 2, dx=tr_obs.stats.delta)
#             out[tr_obs.id] = misfit
#         return out
#
#     @cache
#     def get_adjoint_sources(self):
#         """Return the adjoint source trace."""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             new = tr_obs.copy()
#             new.data = tr_synth.data - tr_obs.data
#             out[tr_synth.id] = new
#         return out
#
#
# class TravelTimeMisFit(_BaseMisFit):
#     """
#     Manager to calculate misfit and ajoints for traveltime misfit.
#
#     Parameters
#     ----------
#     observed_path
#         The directory containing the observed data.
#     synthetic_path
#         The directory containing the synthetic data.
#     trace_min
#         A ratio to zero-out very low amplitude traces
#         to avoid numerical noise. If the max value of a trace
#         divided by the max value of a maximum trace is less than
#         this ratio, zero its adjoint.
#     """
#
#     def __init__(self, observed_path, synthetic_path, trace_min=0.01):
#         """Read in the streams for each directory."""
#         self._trace_min = trace_min
#         super().__init__(observed_path, synthetic_path)
#
#     @cache
#     def calc_trace_absmax(self):
#         """Get a dict of the absolute max of a trace."""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             out[tr_obs.id] = {
#                 "observed": np.abs(tr_obs.data).max(),
#                 "synthetic": np.abs(tr_synth.data).max(),
#             }
#         return pd.DataFrame(out).T
#
#     @cache
#     def calc_normalization(self):
#         """Calculate the normalization waveforms."""
#         out = {}
#         for _, tr_synth in self.iterate_streams():
#             dt = 1 / tr_synth.stats.sampling_rate
#             double_t_diff = tr_synth.copy().differentiate().differentiate()
#             norm = double_t_diff.data * tr_synth.data
#             norm_sum = simps(norm, dx=dt)
#             out[tr_synth.id] = norm_sum
#         return out
#
#     @cache
#     def calc_tt_diff(self) -> dict:
#         """Calculate the travel time differences"""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             cor = correlate(tr_obs, tr_synth, 100)
#             shift, value = xcorr_max(cor)
#             values = shift / tr_obs.stats.sampling_rate
#             out[tr_obs.id] = values
#         return out
#
#     @cache
#     def calc_misfit(self):
#         """Calculate the misfit between streams."""
#         return {i: v**2 for i, v in self.calc_tt_diff().items()}
#
#     @cache
#     def get_adjoint_sources(self):
#         """Return the adjoint source trace."""
#         out = {}
#         tt_diffs = self.calc_tt_diff()
#         norms = self.calc_normalization()
#         for tr_obs, tr_synth in self.iterate_streams():
#             out_tr = tr_synth.copy()
#             dt = 1 / tr_synth.stats.sampling_rate
#             tid = tr_synth.id
#             diff = tt_diffs[tr_synth.id]
#             norm = 1 / norms[tr_synth.id]
#             tdff = np.gradient(tr_synth.data, dt)
#             data = -diff * norm * tdff
#             if self._should_zero_trace(tid):
#                 data = np.zeros_like(data)
#             out_tr.data = data
#             out[tid] = out_tr
#         return out
#
#     def _should_zero_trace(self, tid):
#         """Return true if data should be zeroes"""
#         df = self.calc_trace_absmax()
#         max_vals = df.max()
#         current = df.loc[tid]
#         return ((current / max_vals) < self._trace_min).any()
#
#
# class AmplitudeMisFit(_BaseMisFit):
#     """
#     Manager to calculate misfit and ajoints for amplitude misfit.
#
#     Parameters
#     ----------
#     observed_path
#         The directory containing the observed data.
#     synthetic_path
#         The directory containing the synthetic data.
#     """
#
#     def __init__(self, observed_path, synthetic_path, trace_min=0.01):
#         """Read in the streams for each directory."""
#         self._trace_min = trace_min
#         super().__init__(observed_path, synthetic_path)
#
#     @cache
#     def calc_trace_absmax(self):
#         """Get a dict of the absolute max of a trace."""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             out[tr_obs.id] = {
#                 "observed": np.abs(tr_obs.data).max(),
#                 "synthetic": np.abs(tr_synth.data).max(),
#             }
#         return pd.DataFrame(out).T
#
#     @cache
#     def calc_normalization(self):
#         """Calculate the normalization waveforms."""
#         out = {}
#         for _, tr_synth in self.iterate_streams():
#             data = tr_synth.data**2
#             tid = tr_synth.id
#             out[tid] = simps(data, dx=1 / tr_synth.stats.sampling_rate)
#         return out
#
#     @cache
#     def calc_amp_ratio(self) -> dict:
#         """Calculate the travel time differences"""
#         out = {}
#         for tr_obs, tr_synth in self.iterate_streams():
#             rms_obs = np.sqrt(np.mean(tr_obs.data**2))
#             rms_synth = np.sqrt(np.mean(tr_synth.data**2))
#             out[tr_obs.id] = np.log(rms_obs / rms_synth)
#         return out
#
#     @cache
#     def calc_misfit(self):
#         """Calculate the misfit between streams."""
#         return {i: v**2 for i, v in self.calc_amp_ratio().items()}
#
#     @cache
#     def get_adjoint_sources(self):
#         """Return the adjoint source trace."""
#         out = {}
#         amp_ratios = self.calc_amp_ratio()
#         norms = self.calc_normalization()
#         for tr_obs, tr_synth in self.iterate_streams():
#             out_tr = tr_synth.copy()
#             norm = 1 / norms[tr_obs.id]
#             amp = amp_ratios[tr_obs.id]
#             synth_data = tr_synth.data
#             data = -norm * amp * synth_data
#             if self._should_zero_trace(tr_obs.id):
#                 data = np.zeros_like(data)
#             out_tr.data = data
#             out[out_tr.id] = out_tr
#         return out
#
#     def _should_zero_trace(self, tid):
#         """Return true if data should be zeroes"""
#         df = self.calc_trace_absmax()
#         max_vals = df.max()
#         current = df.loc[tid]
#         return ((current / max_vals) < self._trace_min).any()
