import numpy as np
from scipy.signal import tukey


def check_window_limits(window, llim, rlim, minlen):
    window[0] = max(llim, window[0])
    window[1] = min(rlim, window[1])
    if (window[1]-window[0]) < minlen:
        window = None
    return window

def compute_snr(u, wsignal, wnoise):
    signal = u[wsignal[0]:wsignal[1]].copy()
    noise = u[wnoise[0]:wnoise[1]].copy()
    snr = np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(noise)))
    if snr < 0:
        snr = -10.0 * np.log10(-snr)
    elif snr > 0:
        snr = 10.0 * np.log10(snr)
    return snr

def compute_cc_tshift(u1, u2, dt):
    corr = np.correlate(u1, u2, mode="full")
    corr /= (np.linalg.norm(u1, ord=2) * np.linalg.norm(u2, ord=2))
    cc = np.max(corr)
    tshift = (np.argmax(corr) - len(u1) + 1) * dt
    return cc, tshift

def window_correlation(tr_obs, tr_syn, wdur, snr_thr, cc_thr, tshift_thr):
    dt = tr_syn.stats.delta
    nt = tr_syn.stats.npts

    zero_lag = int((nt-1)/2)
    whlen = int((wdur/2.0)/dt)
    wnoiseoffset = int((wdur/4.0)/dt)

    # negative branch
    obs_neg = tr_obs.data.copy()
    obs_neg[zero_lag:] *= 0.0

    wcen = np.argmax(np.square(obs_neg))
    wsig_neg = [int(wcen-whlen), int(wcen+whlen)]

    wsig_neg = check_window_limits(wsig_neg, 0, zero_lag, whlen)

    if wsig_neg:
        wnoise_neg = [wsig_neg[0]-wnoiseoffset-2*whlen, wsig_neg[0]-wnoiseoffset]
        wnoise_neg = check_window_limits(wnoise_neg, 0, wsig_neg[0]-wnoiseoffset, whlen)
        if wnoise_neg:
            snr_neg = compute_snr(tr_obs.data, wsig_neg, wnoise_neg)
            d = tr_obs.data[wsig_neg[0]:wsig_neg[1]].copy()
            s = tr_syn.data[wsig_neg[0]:wsig_neg[1]].copy()
            d *= tukey(d.size, 0.2)
            s *= tukey(s.size, 0.2)
            cc_neg, tshift_neg = compute_cc_tshift(d, s, dt)
            if (snr_neg < snr_thr or
                cc_neg < cc_thr or
                abs(tshift_neg) > tshift_thr): 
                wsig_neg = None
        else:
            wsig_neg = None
    else:
        wsig_neg = None

    # positive branch
    obs_pos = tr_obs.data.copy()
    obs_pos[0:zero_lag] *= 0.0

    wcen = np.argmax(np.square(obs_pos))
    wsig_pos = [int(wcen-whlen), int(wcen+whlen)]

    wsig_pos = check_window_limits(wsig_pos, zero_lag, nt, whlen)

    if wsig_pos:
        wnoise_pos = [wsig_pos[1]+wnoiseoffset, wsig_pos[1]+wnoiseoffset+2*whlen]
        wnoise_pos = check_window_limits(wnoise_pos, wsig_pos[1]+wnoiseoffset, nt, whlen)
        if wnoise_pos:
            snr_pos = compute_snr(tr_obs.data, wsig_pos, wnoise_pos)
            d = tr_obs.data[wsig_pos[0]:wsig_pos[1]].copy()
            s = tr_syn.data[wsig_pos[0]:wsig_pos[1]].copy()
            d *= tukey(d.size, 0.2)
            s *= tukey(s.size, 0.2)
            cc_pos, tshift_pos = compute_cc_tshift(d, s, dt)
            if (snr_pos < snr_thr or
                cc_pos < cc_thr or
                abs(tshift_pos) > tshift_thr): 
                wsig_pos = None
        else:
            wsig_pos = None
    else:
        wsig_pos = None

    return wsig_neg, wsig_pos

def window_waveform(tr_obs, tr_syn, wdur, snr_thr, cc_thr, tshift_thr):
    dt = tr_syn.stats.delta
    nt = tr_syn.stats.npts

    whlen = int((wdur/2.0)/dt)
    wnoiseoffset = int((wdur/4.0)/dt)
    wcen = np.argmax(np.square(tr_obs.data))
    wsig = [int(wcen-whlen), int(wcen+whlen)]

    wsig = check_window_limits(wsig, 0, nt, whlen)

    if wsig:
        wnoise = [wsig[1]+wnoiseoffset, wsig[1]+wnoiseoffset+2*whlen]
        wnoise = check_window_limits(wnoise, wsig[1]+wnoiseoffset, nt, whlen)
        if wnoise:
            snr = compute_snr(tr_obs.data, wsig, wnoise)
            d = tr_obs.data[wsig[0]:wsig[1]].copy()
            s = tr_syn.data[wsig[0]:wsig[1]].copy()
            d *= tukey(d.size, 0.2)
            s *= tukey(s.size, 0.2)
            cc, tshift = compute_cc_tshift(d, s, dt)
            if (snr < snr_thr or
                cc < cc_thr or
                abs(tshift) > tshift_thr):
                wsig = None
        else:
            wsig = None
    else:
        wsig = None

    return wsig

