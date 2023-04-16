#!/usr/bin/env python3
"""
Similar to SeisFlows Default Preprocessing module but for noise correlations
"""
import numpy as np
import os
from glob import glob
from obspy import Stream
from scipy.signal import correlate, correlation_lags
from seisflows.tools import unix
from seisflows.preprocess.default import Default


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

    Paths
    -----
    :type dummy: dummy
    :param dummy: dummy
    ***
    """
    __doc__ = Default.__doc__ + __doc__

    def __init__(self, apply_window=False, window_len=None, window_cc_thr=None,
                 window_delay_thr=None, **kwargs):
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

                # treat correlation branches separately
                nt = tr_syn.stats.npts
                zero_samp = int((nt - 1) / 2)

                for i in range(0,2):
                    if i == 0:
                        idx1 = 0
                        idx2 = zero_samp + 1
                    elif i == 1:
                        idx1 = zero_samp
                        idx2 = nt

                    tr_obs_branch = tr_obs.data[idx1:idx2].copy()
                    tr_syn_branch = tr_syn.data[idx1:idx2].copy()

                    if self.apply_window:
                        tr_obs_branch, tr_syn_branch, skip = self.max_env_win(
                            tr_obs_branch, tr_syn_branch, tr_syn_branch.size,
                            tr_syn.stats.delta)

                        if skip:
                            continue

                    # Calculate the misfit value and write to file
                    if save_residuals and self._calculate_misfit:
                        residual = self._calculate_misfit(
                            obs=tr_obs_branch, syn=tr_syn_branch,
                            nt=tr_syn_branch.size, dt=tr_syn.stats.delta
                        )
                        with open(save_residuals, "a") as f:
                            f.write(f"{residual:.2E}\n")

                    # Generate an adjoint source trace, write to file
                    if save_adjsrcs and self._generate_adjsrc:
                        adjsrc_branch = self._generate_adjsrc(
                            obs=tr_obs_branch, syn=tr_syn_branch,
                            nt=tr_syn_branch.size, dt=tr_syn.stats.delta
                        )
                        adjsrc.data[idx1:idx2] = adjsrc_branch.copy()

                if save_adjsrcs and self._generate_adjsrc:
                    adjsrc = Stream(adjsrc)
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid)
                    self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, fid))

        if save_adjsrcs and self._generate_adjsrc:
            self._check_adjoint_traces(source_name, save_adjsrcs, synthetic)

        # Exporting residuals to disk (output/) for more permanent storage
        if export_residuals:
            if not os.path.exists(export_residuals):
                unix.mkdir(export_residuals)
            unix.cp(src=save_residuals, dst=export_residuals)

    def max_env_win(self, obs, syn, nt, dt):

        def _xcorr(s1, s2, dt):
            corr = correlate(s1, s2, mode='full')
            corr /= np.linalg.norm(s1) * np.linalg.norm(s2)
            cc = np.max(corr)
            lags = correlation_lags(s1.size, s2.size, mode='full')
            max_lag = lags[np.argmax(corr)] * dt
            return cc, max_lag

        max_samp = np.argmax(obs ** 2.0)
        win_ext = int((self.window_len / 2.0) / dt)
        ind_lo = int(max_samp - win_ext)
        ind_hi = int(max_samp + win_ext)

        if ind_lo > 0 and ind_hi < nt:
            win = np.zeros(nt)
            win[ind_lo:ind_hi+1] += np.hanning(ind_hi + 1 - ind_lo)

            obs *= win
            syn *= win

            cc, max_lag = _xcorr(obs, syn, dt)

            if cc < self.window_cc_thr or np.abs(max_lag) > self.window_delay_thr:
                skip =  True
            else:
                skip = False
        else:
            skip = True

        return obs, syn, skip
