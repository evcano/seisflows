#!/usr/bin/env python3
"""
Similar to SeisFlows Default Preprocessing module but for noise correlations
"""
import numpy as np
import os
from glob import glob
from obspy import Stream
from scipy.signal import tukey
from seisflows.tools import unix
from seisflows.preprocess.default import Default
from seisflows.plugins.preprocess.window import window_correlation


class DefaultFwani(Default):
    """
    Default Noise Preprocess
    ------------------
    Dummy

    Parameters
    ----------
    :type apply_window: bool
    :param apply_window: mute the entire waveform except on a window centered
        at the maximum of the waveform envelope
    :type window_len: float
    :param window_len: window length in seconds
    :type window_cc_thr: float
    :param window_cc_thr: minimum normalized cross-correlation coefficient
        between observations and synthetics to accept a window
    :type window_delay_thr: float
    :param window_delay_thr: maximum traveltime delay between observations and
        synthetics to accept a window
    :type window_snr_thr: float
    :param window_snr_thr: minimum SNR in dB to accept a window
    :type data_uncertainty: float
    :param data_uncertainty: observed data uncertainty in seconds

    Paths
    -----
    :type dummy: dummy
    :param dummy: dummy
    ***
    """
    __doc__ = Default.__doc__ + __doc__

    def __init__(self, apply_window=False, window_len=None, window_cc_thr=None,
                 window_delay_thr=None, window_snr_thr=None,
                 data_uncertainty=None, ntask=1, **kwargs):
        """
        Noise Preprocessing module parameters

        :type dummy: dummy
        :param dummy: dummy

        """
        super().__init__(**kwargs)

        self.apply_window = apply_window
        self.window_len = window_len
        self.window_cc_thr = window_cc_thr
        self.window_delay_thr = window_delay_thr
        self.window_snr_thr = window_snr_thr
        self.data_uncertainty = data_uncertainty
        self._ntask = ntask

    def quantify_misfit(self, source_name=None, save_residuals=None,
                        export_residuals=None, save_adjsrcs=None, iteration=1,
                        step_count=0, **kwargs):
        """
        Prepares solver for gradient evaluation by writing residuals and
        adjoint traces. Meant to be called by solver.eval_func().

        Reads in observed and synthetic waveforms, applies optional
        preprocessing, assesses misfit, and writes out adjoint sources and
        STATIONS_ADJOINT file.

        TODO use concurrent futures to parallelize this

        :type source_name: str
        :param source_name: name of the event to quantify misfit for. If not
            given, will attempt to gather event id from the given task id which
            is assigned by system.run()
        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type save_adjsrcs: str
        :param save_adjsrcs: if not None, path to write adjoint sources to
        :type iteration: int
        :param iteration: current iteration of the workflow, information should
            be provided by `workflow` module if we are running an inversion.
            Defaults to 1 if not given (1st iteration)
        :type step_count: int
        :param step_count: current step count of the line search. Information
            should be provided by the `optimize` module if we are running an
            inversion. Defaults to 0 if not given (1st evaluation)
        """
        observed, synthetic = self._setup_quantify_misfit(source_name)

        # The names of the source and the reference station are the same
        sta1 = source_name

        # Make sure residuals files does not exist so we dont append residuals
        # from previous simulations
        if os.path.exists(save_residuals):
            unix.rm(save_residuals)

        for obs_fid, syn_fid in zip(observed, synthetic):
            obs = self.read(fid=obs_fid, data_format=self.obs_data_format)
            syn = self.read(fid=syn_fid, data_format=self.syn_data_format)

            sta2 = os.path.basename(syn_fid).split(".")[0:2]
            sta2 = f"{sta2[0]}.{sta2[1]}"

            # Skip autocorrelations
            if sta1 == sta2:
                continue

            # Process observations and synthetics identically
            if self.filter:
                obs = self._apply_filter(obs)
                syn = self._apply_filter(syn)
            if self.mute:
                obs = self._apply_mute(obs)
                syn = self._apply_mute(syn)
            if self.normalize:
                obs = self._apply_normalize(obs)
                syn = self._apply_normalize(syn)

            # Write the residuals/misfit and adjoint sources for each component
            for tr_obs, tr_syn in zip(obs, syn):
                # Simple check to make sure zip retains ordering
                assert(tr_obs.stats.component == tr_syn.stats.component)

                if save_adjsrcs and self._generate_adjsrc:
                    adjsrc = tr_syn.copy()
                    adjsrc.data *= 0.0

                if self.apply_window:
                    win_neg, win_pos = window_correlation(tr_obs,
                                                          tr_syn,
                                                          self.window_len,
                                                          self.window_snr_thr,
                                                          self.window_cc_thr,
                                                          self.window_delay_thr
                                                         )
                else:
                    zero_lag = int((tr_syn.stats.npts-1)/2)
                    win_neg = [0, zero_lag]
                    win_pos = [zero_lag, tr_syn.stats.npts]

                for win in [win_neg, win_pos]:
                    if not win:
                        continue

                    obs_win = tr_obs.data[win[0]:win[1]].copy()
                    syn_win = tr_syn.data[win[0]:win[1]].copy()

                    obs_win *= tukey(obs_win.size, 0.2)
                    syn_win *= tukey(syn_win.size, 0.2)

                    # Calculate the misfit value and write to file
                    if save_residuals and self._calculate_misfit:
                        residual = self._calculate_misfit(
                            obs=obs_win, syn=syn_win,
                            nt=syn_win.size, dt=tr_syn.stats.delta
                        )
                        if self.data_uncertainty:
                            residual *= (1.0 / self.data_uncertainty)
                        with open(save_residuals, "a") as f:
                            f.write(f"{residual:.2E}\n")

                    # Generate an adjoint source trace, write to file
                    if save_adjsrcs and self._generate_adjsrc:
                        adjsrc_win = self._generate_adjsrc(
                            obs=obs_win, syn=syn_win,
                            nt=syn_win.size, dt=tr_syn.stats.delta
                        )
                        if self.data_uncertainty:
                            # the adjoint source was multiplied by the residual
                            # but not by the data uncertainty
                            adjsrc_win *= (1.0 / self.data_uncertainty)
                            if np.abs(residual) <= self.data_uncertainty:
                                adjsrc_win *= 0.0
                        adjsrc_win *= tukey(adjsrc_win.size, 0.2)
                        adjsrc.data[win[0]:win[1]] = adjsrc_win.copy()

                if save_adjsrcs and self._generate_adjsrc:
                    adjsrc = Stream(adjsrc)
                    if self.filter:
                        adjsrc = self._apply_filter(adjsrc)
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid)
                    self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, fid))

        # Exporting residuals to disk (output/) for more permanent storage
        if export_residuals:
            if not os.path.exists(export_residuals):
                unix.mkdir(export_residuals)
            unix.cp(src=save_residuals, dst=export_residuals)

        if save_adjsrcs and self._generate_adjsrc:
            self._check_adjoint_traces(source_name, save_adjsrcs, synthetic)

    def sum_residuals(self, residuals):
        return np.sum(np.square(residuals)) / residuals.size
