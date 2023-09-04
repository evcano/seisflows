#!/usr/bin/env python3
"""
The SeisFlows Preprocessing module is in charge of interacting with seismic
data (observed and synthetic). It should contain functionality to read and write
seismic data, apply preprocessing such as filtering, quantify misfit,
and write adjoint sources that are expected by the solver.
"""
import os
import numpy as np
from glob import glob
from obspy import read as obspy_read
from obspy import Stream, Trace, UTCDateTime
from scipy.signal import tukey

from seisflows import logger
from seisflows.tools import signal, unix
from seisflows.tools.config import Dict, get_task_id

from seisflows.plugins.preprocess import misfit as misfit_functions
from seisflows.plugins.preprocess import adjoint as adjoint_sources
from seisflows.plugins.preprocess.window import window_waveform


class Default:
    """
    Default Preprocess
    ------------------
    Data processing for seismic traces, with options for data misfit,
    filtering, normalization and muting.

    Parameters
    ----------
    :type obs_data_format: str
    :param obs_data_format: data format for reading observed traces into
        memory. Available formats: 'su', 'ascii', 'sac'
    :type unit_output: str
    :param unit_output: Data units. Must match the synthetic output of
        external solver. Available: ['DISP': displacement, 'VEL': velocity,
        'ACC': acceleration, 'PRE': pressure]
    :type misfit: str
    :param misfit: misfit function for waveform comparisons. For available
        see seisflows.plugins.preprocess.misfit
    :type backproject: str
    :param backproject: backprojection function for migration, or the
        objective function in FWI. For available see
        seisflows.plugins.preprocess.adjoint
    :type normalize: str
    :param normalize: Data normalization parameters used to normalize the
        amplitudes of waveforms. Choose from two sets:
        ENORML1: normalize per event by L1 of traces; OR
        ENORML2: normalize per event by L2 of traces;
        &
        TNORML1: normalize per trace by L1 of itself; OR
        TNORML2: normalize per trace by L2 of itself
    :type filter: str
    :param filter: Data filtering type, available options are:
        BANDPASS (req. MIN/MAX PERIOD/FREQ);
        LOWPASS (req. MAX_FREQ or MIN_PERIOD);
        HIGHPASS (req. MIN_FREQ or MAX_PERIOD)
    :type min_period: float
    :param min_period: Minimum filter period applied to time series.
        See also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they
        will overwrite PERIOD parameters.
    :type max_period: float
    :param max_period: Maximum filter period applied to time series. See
        also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they will
        overwrite PERIOD parameters.
    :type min_freq: float
    :param min_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type max_freq: float
    :param max_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type mute: list
    :param mute: Data mute parameters used to zero out early / late
        arrivals or offsets. Choose any number of:
        EARLY: mute early arrivals;
        LATE: mute late arrivals;
        SHORT: mute short source-receiver distances;
        LONG: mute long source-receiver distances

    Paths
    -----
    :type path_preprocess: str
    :param path_preprocess: scratch path for all preprocessing processes,
        including saving files
    ***
    """
    def __init__(self, syn_data_format="ascii", obs_data_format="ascii",
                 unit_output="VEL", misfit="waveform",
                 adjoint="waveform", normalize=None, filter=None,
                 min_period=None, max_period=None, min_freq=None, max_freq=None,
                 mute=None, early_slope=None, early_const=None, late_slope=None,
                 late_const=None, short_dist=None, long_dist=None,
                 workdir=os.getcwd(), path_preprocess=None, path_solver=None,
                 apply_window=False, window_len=None, window_cc_thr=None,
                 window_delay_thr=None, window_snr_thr=None, ntask=1,
                 data_uncertainty=None, **kwargs):
        """
        Preprocessing module parameters

        .. note::
            Paths and parameters listed here are shared with other modules and 
            so are not included in the class docstring.

        :type syn_data_format: str
        :param syn_data_format: data format for reading synthetic traces into
            memory. Shared with solver module. Available formats: 'su', 'ascii'
        :type workdir: str
        :param workdir: working directory in which to look for data and store
        results. Defaults to current working directory
        :type path_preprocess: str
        :param path_preprocess: scratch path for all preprocessing processes,
            including saving files
        """
        self.syn_data_format = syn_data_format.upper()
        self.obs_data_format = obs_data_format.upper()
        self.unit_output = unit_output.upper()
        self.misfit = misfit
        self.adjoint = adjoint
        self.normalize = normalize

        self.filter = filter
        self.min_period = min_period
        self.max_period = max_period
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mute = mute or []
        self.normalize = normalize or []

        # Mute arrivals sub-parameters
        self.early_slope = early_slope
        self.early_const = early_const
        self.late_slope = late_slope
        self.late_const = late_const
        self.short_dist = short_dist
        self.long_dist = long_dist

        self.path = Dict(
            scratch=path_preprocess or os.path.join(workdir, "scratch",
                                                    "preprocess"),
            solver=path_solver or os.path.join(workdir, "scratch", "solver")
        )

        # The list <_obs_acceptable_data_formats> always includes
        # <_syn_acceptable_data_formats> in addition to more formats
        self._syn_acceptable_data_formats = ["SU", "ASCII"]
        self._obs_acceptable_data_formats = ["SU", "ASCII", "SAC"]

        self._acceptable_unit_output = ["DISP", "VEL", "ACC", "PRE"]

        # Misfits and adjoint sources are defined by the available functions
        # in each of these plugin files. Drop hidden variables from dir()
        self._acceptable_misfits = [_ for _ in dir(misfit_functions)
                                    if not _.startswith("_")]
        self._acceptable_adjsrcs = [_ for _ in dir(adjoint_sources)
                                    if not _.startswith("_")]

        # Internal attributes used to keep track of inversion workflows
        self._iteration = None
        self._step_count = None
        self._source_names = None

        self.apply_window = apply_window
        self.window_len = window_len
        self.window_cc_thr = window_cc_thr
        self.window_delay_thr = window_delay_thr
        self.window_snr_thr = window_snr_thr
        self.data_uncertainty = data_uncertainty
        self._ntask = ntask

    def check(self):
        """ 
        Checks parameters and paths
        """
        if self.misfit:
            assert(self.misfit in self._acceptable_misfits), \
                f"preprocess.misfit must be in {self._acceptable_misfits}"
        if self.adjoint:
            assert(self.adjoint in self._acceptable_adjsrcs), \
                f"preprocess.misfit must be in {self._acceptable_adjsrcs}"

        # Data normalization option
        if self.normalize:
            acceptable_norms = {"TNORML1", "TNORML2", "ENORML1", "ENORML2"}
            chosen_norms = [_.upper() for _ in self.normalize]
            assert(set(chosen_norms).issubset(acceptable_norms))

        # Data muting options
        if self.mute:
            acceptable_mutes = {"EARLY", "LATE", "LONG", "SHORT"}
            chosen_mutes = [_.upper() for _ in self.mute]
            assert(set(chosen_mutes).issubset(acceptable_mutes))
            if "EARLY" in chosen_mutes:
                assert(self.early_slope is not None)
                assert(self.early_slope >= 0.)
                assert(self.early_const is not None)
            if "LATE" in chosen_mutes:
                assert(self.late_slope is not None)
                assert(self.late_slope >= 0.)
                assert(self.late_const is not None)
            if "SHORT" in chosen_mutes:
                assert(self.short_dist is not None)
                assert (self.short_dist > 0)
            if "LONG" in chosen_mutes:
                assert(self.long_dist is not None)
                assert (self.long_dist > 0)

#        # Data filtering options that will be passed to ObsPy filters
#        if self.filter:
#            acceptable_filters = ["BANDPASS", "LOWPASS", "HIGHPASS"]
#            assert self.filter.upper() in acceptable_filters, \
#                f"self.filter must be in {acceptable_filters}"
#
#            # Set the min/max frequencies and periods, frequency takes priority
#            if self.min_freq is not None:
#                self.max_period = 1 / self.min_freq
#            elif self.max_period is not None:
#                self.min_freq = 1 / self.max_period
#
#            if self.max_freq is not None:
#                self.min_period = 1 / self.max_freq
#            elif self.min_period is not None:
#                self.max_freq =  1 / self.min_period
#
#            # Check that the correct filter bounds have been set
#            if self.filter.upper() == "BANDPASS":
#                assert(self.min_freq is not None and
#                       self.max_freq is not None), \
#                    ("BANDPASS filter PAR.MIN_PERIOD and PAR.MAX_PERIOD or " 
#                     "PAR.MIN_FREQ and PAR.MAX_FREQ")
#            elif self.filter.upper() == "LOWPASS":
#                assert(self.max_freq is not None or
#                       self.min_period is not None),\
#                    "LOWPASS requires PAR.MAX_FREQ or PAR.MIN_PERIOD"
#            elif self.filter.upper() == "HIGHPASS":
#                assert(self.min_freq is not None or
#                       self.max_period is not None),\
#                    "HIGHPASS requires PAR.MIN_FREQ or PAR.MAX_PERIOD"
#
#            # Check that filter bounds make sense, by this point, MIN and MAX
#            # FREQ and PERIOD should be set, so we just check the FREQ
#            assert(0 < self.min_freq < np.inf), "0 < PAR.MIN_FREQ < inf"
#            assert(0 < self.max_freq < np.inf), "0 < PAR.MAX_FREQ < inf"
#            assert(self.min_freq < self.max_freq), (
#                "PAR.MIN_FREQ < PAR.MAX_FREQ"
#            )

        assert(self.syn_data_format.upper() in self._syn_acceptable_data_formats), \
            f"synthetic data format must be in {self._syn_acceptable_data_formats}"

        assert(self.obs_data_format.upper() in self._obs_acceptable_data_formats), \
            f"observed data format must be in {self._obs_acceptable_data_formats}"

        assert(self.unit_output.upper() in self._acceptable_unit_output), \
            f"unit output must be in {self._acceptable_unit_output}"

    def setup(self):
        """
        Sets up data preprocessing machinery by dynamicalyl loading the
        misfit, adjoint source type, and specifying the expected file type
        for input and output seismic data.
        """
        unix.mkdir(self.path.scratch)

    def read(self, fid, data_format):
        """
        Waveform reading functionality. Imports waveforms as Obspy streams

        :type fid: str
        :param fid: path to file to read data from
        :type data_format: str
        :param data_format: format of the file to read data from
        :rtype: obspy.core.stream.Stream
        :return: ObsPy stream containing data stored in `fid`
        """
        st = None
        if data_format.upper() == "SU":
            st = obspy_read(fid, format="SU", byteorder="<")
        elif data_format.upper() == "SAC":
            st = obspy_read(fid, format="SAC")
        elif data_format.upper() == "ASCII":
            st = read_ascii(fid)
        return st

    def write(self, st, fid):
        """
        Waveform writing functionality. Writes waveforms back to format that
        SPECFEM recognizes

        :type st: obspy.core.stream.Stream
        :param st: stream to write
        :type fid: str
        :param fid: path to file to write stream to
        """
        if self.syn_data_format.upper() == "SU":
            for tr in st:
                # Work around for ObsPy data type conversion
                tr.data = tr.data.astype(np.float32)
            max_delta = 0.065535
            dummy_delta = max_delta

            if st[0].stats.delta > max_delta:
                for tr in st:
                    tr.stats.delta = dummy_delta

            # Write data to file
            st.write(fid, format="SU")

        elif self.syn_data_format.upper() == "ASCII":
            for tr in st:
                # Float provides time difference between starttime and default
                time_offset = float(tr.stats.starttime)
                data_out = np.vstack((tr.times() + time_offset, tr.data)).T
                np.savetxt(fid, data_out, ["%13.7f", "%17.7f"])

    def _calculate_misfit(self, **kwargs):
        """Wrapper for plugins.preprocess.misfit misfit/objective function"""
        if self.misfit is not None:
            return getattr(misfit_functions, self.misfit)(**kwargs)
        else:
            return None

    def _generate_adjsrc(self, **kwargs):
        """Wrapper for plugins.preprocess.adjoint source/backproject function"""
        if self.adjoint is not None:
            return getattr(adjoint_sources, self.adjoint)(**kwargs)
        else:
            return None

    def initialize_adjoint_traces(self, data_filenames, output,
                                  data_format=None):
        """
        SPECFEM requires that adjoint traces be present for every matching
        synthetic seismogram. If an adjoint source does not exist, it is
        simply set as zeros. This function creates all adjoint traces as
        zeros, to be filled out later

        Appends '.adj. to the solver filenames as expected by SPECFEM (if they
        don't already have that extension)

        TODO there are some sem2d and 3d specific tasks that are not carried
        TODO over here, were they required?

        :type data_filenames: list of str
        :param data_filenames: existing solver waveforms (synthetic) to read.
            These will be copied, zerod out, and saved to path `save`. Should
            come from solver.data_filenames
        :type output: str
        :param output: path to save the new adjoint traces to.
        """
        for fid in data_filenames:
            st = self.read(fid=fid, data_format=self.syn_data_format).copy()
            fid = os.path.basename(fid)  # drop any path before filename
            for tr in st:
                tr.data *= 0

            adj_fid = self._rename_as_adjoint_source(fid)

            # Write traces back to the adjoint trace directory
            self.write(st=st, fid=os.path.join(output, adj_fid))

    def _check_adjoint_traces(self, source_name, save_adjsrcs, synthetic):
        """Check that all adjoint traces required by SPECFEM exist"""
        source_name = source_name or self._source_names[get_task_id()]
        specfem_data_path = os.path.join(self.path.solver, source_name, "DATA")

        # since <STATIONS_ADJOINT> is generated only when using SPECFEM3D
        # by copying <STATIONS>, check adjoint stations in <STATIONS>
        adj_stations = np.loadtxt(os.path.join(specfem_data_path,
                                               "STATIONS"), dtype="str")

        if not isinstance(adj_stations[0], np.ndarray):
            adj_stations = [adj_stations]

        adj_template = "{net}.{sta}.{chan}.adj"

        channels = [os.path.basename(syn).split('.')[2] for syn in synthetic]
        channels = list(set(channels))

        st = self.read(fid=synthetic[0], data_format=self.syn_data_format)
        for tr in st:
            tr.data *= 0.

        for adj_sta in adj_stations:
            sta = adj_sta[0]
            net = adj_sta[1]
            for chan in channels:
                adj_trace = adj_template.format(net=net, sta=sta, chan=chan)
                adj_trace = os.path.join(save_adjsrcs, adj_trace)
                if not os.path.isfile(adj_trace):
                    self.write(st=st, fid=adj_trace)

    def _rename_as_adjoint_source(self, fid):
        """
        Rename synthetic waveforms into filenames consistent with how SPECFEM
        expects adjoint sources to be named. Usually this just means adding
        a '.adj' to the end of the filename
        """
        if not fid.endswith(".adj"):
            if self.syn_data_format.upper() == "SU":
                fid = f"{fid}.adj"
            elif self.syn_data_format.upper() == "ASCII":
                # Differentiate between SPECFEM3D and 3D_GLOBE
                # SPECFEM3D: NN.SSSS.CCC.sem?
                # SPECFEM3D_GLOBE: NN.SSSS.CCC.sem.ascii
                ext = os.path.splitext(fid)[-1]  
                # SPECFEM3D
                if ".sem" in ext:
                    fid = fid.replace(ext, ".adj")
                # GLOBE (!!! Hardcoded to only work with ASCII format)
                elif ext == ".ascii":
                    root, ext1 = os.path.splitext(fid)  # .ascii
                    root, ext2 = os.path.splitext(root)  # .sem
                    fid = fid.replace(f"{ext2}{ext1}", ".adj")

        return fid

    def _setup_quantify_misfit(self, source_name):
        """
        Gather waveforms from the Solver scratch directory which will be used
        for generating adjoint sources
        """
        source_name = source_name or self._source_names[get_task_id()]

        obs_path = os.path.join(self.path.solver, source_name, "traces", "obs")
        syn_path = os.path.join(self.path.solver, source_name, "traces", "syn")

        observed = sorted(os.listdir(obs_path))
        synthetic = sorted(os.listdir(syn_path))

        assert(len(observed) != 0 and len(synthetic) != 0), \
            f"cannot quantify misfit, missing observed or synthetic traces"

        # verify observed traces format
        obs_ext = list(set([os.path.splitext(x)[-1] for x in observed]))

        if self.obs_data_format.upper() == "ASCII":
            obs_ext_ok = obs_ext[0].upper() == ".ASCII" or \
                         obs_ext[0].upper() == f".SEM{self.unit_output[0]}"
        else:
            obs_ext_ok = obs_ext[0].upper() == f".{self.obs_data_format}"

        assert(len(obs_ext) == 1 and obs_ext_ok), (
            f"observed traces have more than one format or their format "
            f"is not the one defined in parameters.yaml"
        )

        # verify synthetic traces format
        syn_ext = list(set([os.path.splitext(x)[-1] for x in synthetic]))

        if self.syn_data_format == "ASCII":
            syn_ext_ok = syn_ext[0].upper() == ".ASCII" or \
                         syn_ext[0].upper() == f".SEM{self.unit_output[0]}"
        else:
            syn_ext_ok = syn_ext[0].upper() == f".{self.syn_data_format}"

        assert(len(syn_ext) == 1 and syn_ext_ok), (
            f"synthetic traces have more than one format or their format "
            f"is not the one defined in parameters.yaml"
        )

        # remove data format
        observed = [os.path.splitext(x)[0] for x in observed]
        synthetic = [os.path.splitext(x)[0] for x in synthetic]

        # only return traces that have both observed and synthetic files
        matching_traces = sorted(list(set(synthetic).intersection(observed)))

        assert(len(matching_traces) != 0), (
            f"there are no traces with both observed and synthetic files for "
            f"source: {source_name}; verify that observations and synthetics "
            f"have the same name including channel code"
        )

        observed.clear()
        synthetic.clear()

        for file_name in matching_traces:
            observed.append(os.path.join(obs_path, f"{file_name}{obs_ext[0]}"))
            synthetic.append(os.path.join(syn_path, f"{file_name}{syn_ext[0]}"))

        assert(len(observed) == len(synthetic)), (
            f"number of observed traces does not match length of synthetic for "
            f"source: {source_name}"
        )

        return observed, synthetic

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

                for k in range(0, len(self.window_len)):
                    if save_adjsrcs and self._generate_adjsrc:
                        band_adjsrc = tr_syn.copy()
                        band_adjsrc.data *= 0.0

                    st_obs_band = Stream(tr_obs.copy())
                    st_obs_band = self._apply_filter(st_obs_band, k)
                    tr_obs_band = st_obs_band[0]

                    st_syn_band = Stream(tr_syn.copy())
                    st_syn_band = self._apply_filter(st_syn_band, k)
                    tr_syn_band = st_syn_band[0]

                    if self.apply_window:
                        win = window_waveform(tr_obs_band,
                                              tr_syn_band,
                                              self.window_len[k],
                                              self.window_snr_thr[k],
                                              self.window_cc_thr[k],
                                              self.window_delay_thr[k],
                                              )
                    else:
                        win = [0, tr_syn_band.stats.npts]

                    if not win:
                        continue

                    obs_win = tr_obs_band.data[win[0]:win[1]].copy()
                    syn_win = tr_syn_band.data[win[0]:win[1]].copy()

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
                        with open(f"{save_residuals}.names", "a") as f:
                            f.write(f"{sta1}_{sta2}\n")

                    # Generate an adjoint source trace
                    if save_adjsrcs and self._generate_adjsrc:
                        adjsrc_win = self._generate_adjsrc(
                            obs=obs_win, syn=syn_win,
                            nt=syn_win.size, dt=tr_syn.stats.delta
                        )
                        if self.data_uncertainty:
                            adjsrc_win *= (1.0 / self.data_uncertainty)
                        adjsrc_win *= tukey(adjsrc_win.size, 0.2)
                        band_adjsrc.data[win[0]:win[1]] = adjsrc_win.copy()
                        band_adjsrc = Stream(band_adjsrc)
                        band_adjsrc = self._apply_filter(band_adjsrc, k)
                        adjsrc.data += band_adjsrc[0].data

                if save_adjsrcs and self._generate_adjsrc:
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid)
                    adjsrc = Stream(adjsrc)
                    self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, fid))

        # Exporting residuals to disk (output/) for more permanent storage
        if export_residuals:
            if not os.path.exists(export_residuals):
                unix.mkdir(export_residuals)
            unix.cp(src=save_residuals, dst=export_residuals)

        if save_adjsrcs and self._generate_adjsrc:
            self._check_adjoint_traces(source_name, save_adjsrcs, synthetic)

    def finalize(self):
        """Teardown procedures for the default preprocessing class"""
        pass

    def sum_residuals(self, residuals):
        """
        Returns the summed square of residuals for each event. Following
        Tape et al. 2007

        :type residuals: np.array
        :param residuals: list of residuals from each NTASK event
        :rtype: float
        :return: sum of squares of residuals
        """
        return np.sum(np.square(residuals)) / residuals.size

    def _apply_filter(self, st, i):
        """
        Apply a filter to waveform data using ObsPy

        :type st: obspy.core.stream.Stream
        :param st: stream to be filtered
        :rtype: obspy.core.stream.Stream
        :return: filtered traces
        """
        # Pre-processing before filtering
        st.detrend("demean")
        st.detrend("linear")
        st.taper(0.05, type="hann")

        if self.filter.upper() == "BANDPASS":
            st.filter("bandpass", zerophase=True, freqmin=self.min_freq[i],
                      freqmax=self.max_freq[i], corners=2)
        elif self.filter.upper() == "LOWPASS":
            st.filter("lowpass", zerophase=True, freq=self.max_freq[i])
        elif self.filter.upper() == "HIGHPASS":
            st.filter("highpass", zerophase=True, freq=self.min_freq[i])

        return st

    def _apply_mute(self, st):
        """
        Apply mute on data based on early or late arrivals, and short or long
        source receiver distances

        .. note::
            The underlying mute functions have been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: stream to mute
        :rtype: obspy.core.stream.Stream
        :return: muted stream object
        """
        mute_choices = [_.upper() for _ in self.mute]
        if "EARLY" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.early_slope,
                                      const=self.early_const, choice="EARLY")
        if "LATE" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.late_slope,
                                      const=self.late_const, choice="LATE")
        if "SHORT" in mute_choices:
            st = signal.mute_offsets(st, dist=self.short_dist, choice="SHORT")
        if "LONG" in mute_choices:
            st = signal.mute_offsets(st, dist=self.long_dist, choice="LONG")

        return st

    def _apply_normalize(self, st):
        """
        Normalize the amplitudes of waveforms based on user choice

        .. note::
            The normalization function has been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: All of the data streams to be normalized
        :rtype: obspy.core.stream.Stream
        :return: stream with normalized traces
        """
        st_out = st.copy()
        norm_choices = [_.upper() for _ in self.normalize]

        # Normalize an event by the L1 norm of all traces
        if 'ENORML1' in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=1)
            for tr in st_out:
                tr.data /= w
        # Normalize an event by the L2 norm of all traces
        elif "ENORML2" in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=2)
            for tr in st_out:
                tr.data /= w
        # Normalize each trace by its L1 norm
        if "TNORML1" in norm_choices:
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=1)
                if w > 0:
                    tr.data /= w
        elif "TNORML2" in norm_choices:
            # normalize each trace by its L2 norm
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=2)
                if w > 0:
                    tr.data /= w

        return st_out

    def read_residuals(self, residuals_files):
        residuals = np.array([])
        for residuals_file in residuals_files:
            tmp = np.loadtxt(residuals_file)
            residuals = np.append(residuals, tmp)
        return residuals

def read_ascii(fid, origintime=None):
    """
    Read waveforms in two-column ASCII format. This is copied directly from
    pyatoa.utils.read.read_sem()
    """
    try:
        times = np.loadtxt(fname=fid, usecols=0)
        data = np.loadtxt(fname=fid, usecols=1)

    # At some point in 2018, the Specfem developers changed how the ascii files
    # were formatted from two columns to comma separated values, and repeat
    # values represented as 2*value_float where value_float represents the data
    # value as a float
    except ValueError:
        times, data = [], []
        with open(fid, 'r') as f:
            lines = f.readlines()
        for line in lines:
            try:
                time_, data_ = line.strip().split(',')
            except ValueError:
                if "*" in line:
                    time_ = data_ = line.split('*')[-1]
                else:
                    raise ValueError
            times.append(float(time_))
            data.append(float(data_))

        times = np.array(times)
        data = np.array(data)

    if origintime is None:
        origintime = UTCDateTime("1970-01-01T00:00:00")

    # We assume that dt is constant after 'precision' decimal points
    delta = round(times[1] - times[0], 4)

    # Honor that Specfem doesn't start exactly on 0
    origintime += times[0]

    # Write out the header information. Deal with the fact that SPECFEM2D/3D and
    # 3D_GLOBE have slightly different formats for their filenames
    net, sta, cha, *fmt = os.path.basename(fid).split('.')
    stats = {"network": net, "station": sta, "location": "",
             "channel": cha, "starttime": origintime, "npts": len(data),
             "delta": delta, "mseed": {"dataquality": 'D'},
             "time_offset": times[0], "format": fmt[0]
             }
    st = Stream([Trace(data=data, header=stats)])

    return st
