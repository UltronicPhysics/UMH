"""
UMH_Ligo_Compiler.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Ligo Compiler, for use with UMH_Chirp_Generator.

Parameters:
- OUTPUT_FOLDER

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""
import numpy as np
import os, time
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import sys, math
import json
from numpy.fft import rfft, irfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt, sosfiltfilt, correlate, iirnotch, tf2sos, welch, hilbert, get_window, savgol_filter
#from scipy.fftpack import fft, fftfreq
from scipy.signal import spectrogram, resample, resample_poly, fftconvolve
from matplotlib.ticker import ScalarFormatter
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from fractions import Fraction
import argparse
from matplotlib.colors import Normalize

def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LIGO_DATA":{
            "Hanford":    "PlanckData/H-H1_LOSC_16_V1-1126259446-32.hdf5",
            "Livingston": "PlanckData/L-L1_LOSC_16_V1-1126259446-32.hdf5",
            #"Virgo": "PlanckData/V-V1_LOSC_16_V1-1126259446-32.hdf5"
        },

        "ALLOW_GLOBAL_POLARITY_FLIP":    True,

        "ALLOW_LAG_MAX":                False,
        
        "PHYSICS_STRICT":                True,

        "ENABLE_FINE_STRETCH":          False,          # Allow Stretching, PHYSICS_STRICT must be False for this to be True.

        "ALLOW_PER_DETECTOR_POLARITY":  False,

        "USE_RIDGE_DIAGNOSTIC":          True,

        "USE_MERGE_LOC_FOR_PEAK":       False,


        "NOTCH_LINES": [60,120,180,240,331,500],

        "PSD_MODE": "window",
        "PSD_GUARD_SEC": 2.0,
        "PSD_PRE_START_SEC": 12.0,   # use ~4–12 s if event at 16 s
        "PSD_PRE_END_SEC": 4.0,
        "PSD_USE_POST_WINDOW": False,  # turn on only if you have enough data after event
        "PSD_DROP_FILTER_TRANSIENT_SEC": 1.0,


        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.
        
        "LIGO_OUTPUT_ROOT_FOLDER": os.path.join(base, "Output"),
        "INPUT_FOLDER": os.path.join(base, "Output", "UMH_vs_LIGO"),
        "OUTPUT_FOLDER": os.path.join(base, "Output"),
    }


EPS_SAFE_FLOOR = 1e-24
EPS_FLOOR      = 1e-40
PSD_FLOOR      = 1e-48

# ---- Helper Functions ----
def next_pow2(n): 
    n = int(n)
    if n <= 1: return 1
    return 1 << (n - 1).bit_length()

def _round_pow2(n: int) -> int:
    # nearest power of 2 (>=1)
    if n <= 1: return 1
    p = 1 << (n - 1).bit_length()
    # choose nearest of p and p//2
    lo = p >> 1
    if lo >= 1 and abs(n - lo) < abs(p - n): return lo
    return p

def notch(freq, Q, fs): b, a = iirnotch(w0=freq/(fs/2), Q=Q); return b, a

def apply_notches(sig, fs, lines=(60,120,180,240,331,500), Q=30):
    x = sig
    for f0 in lines:
        b, a = iirnotch(w0=f0/(fs/2), Q=Q)
        sos = tf2sos(b, a)
        x = sosfiltfilt(sos, x)
    return x

def bandpass(sig, fs, f_lo=30.0, f_hi=300.0, order=4):
    wn = [f_lo/(fs/2), f_hi/(fs/2)]
    wn[0] = max(wn[0], 1e-6)
    wn[1] = min(wn[1], 0.999999)
    sos = butter(order, wn, btype="band", output="sos")
    return sosfiltfilt(sos, sig)

def sanitize(x, name=""):
    """
    Physics-safe sanitize: ONLY removes NaN/Inf
    This will NOT change healthy results.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        n_bad = int((~np.isfinite(x)).sum())
        print(f"[WARN] {name}: {n_bad} non-finite samples -> set to 0")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def stable_rms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return float("nan")
    m = np.max(np.abs(x))
    if m == 0: return 0.0
    y = x / m
    return float(m * np.sqrt(np.mean(y*y)))

def primary_peak(env, N, edge_exclude):
    i0p = edge_exclude
    i1p = max(i0p + 1, N - edge_exclude)
    sub = env[i0p:i1p]
    med = np.median(sub)
    mad = np.median(np.abs(sub - med)) + EPS_SAFE_FLOOR
    thr = med + 4.0 * mad
    mask = sub > thr
    idx_local = np.argmax(sub * mask) if np.any(mask) else np.argmax(sub)
    return i0p + idx_local

def find_peak_loudest_significant(fs, env, idx_expected, half_width, edge_exclude,
    smooth_ms=12.0, k_mad=6.0, max_offset_sec=0.08, tie_radius_sec=0.01):
    """
    Expected-locked peak picker (data-only):
    - Search in [idx_expected-half_width, idx_expected+half_width] with edge exclusion
    - Smooth envelope to suppress micro-ripples
    - Find significant local maxima above median + k_mad*MAD
    - PRIMARY: choose the significant peak closest to idx_expected
    - TIE-BREAK: within tie_radius_sec of the closest distance, choose the loudest
    - If closest significant peak is still too far (>max_offset_sec), clamp to idx_expected neighborhood
    - Fallback: loudest point in the window
    """
    N = len(env)
    idx_expected = int(idx_expected); half_width = int(half_width); edge_exclude = int(edge_exclude)

    j0 = max(0, idx_expected - half_width)
    j1 = min(N, idx_expected + half_width)
    if (j1 - j0) < 5: return int(np.clip(idx_expected, 0, N - 1)), 0

    k0 = min(j0 + edge_exclude, j1 - 1)
    k1 = max(j1 - edge_exclude, k0 + 1)
    w  = np.asarray(env[k0:k1], dtype=float)

    # Smooth envelope
    win = max(1, int(round((smooth_ms / 1000.0) * fs)))
    if win > 1: kernel = np.ones(win, dtype=float) / float(win); w = np.convolve(w, kernel, mode="same")

    # Robust threshold
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)) + EPS_SAFE_FLOOR)
    thr = med + float(k_mad) * mad

    if w.size < 3: return int(k0 + np.argmax(w)), 0

    core  = w[1:-1]
    peaks = np.where((core > w[:-2]) & (core >= w[2:]) & (core >= thr))[0] + 1
    if peaks.size == 0: return int(k0 + np.argmax(w)), 0

    idxs = (k0 + peaks).astype(int)
    vals = w[peaks]

    # PRIMARY: closest to expected
    d = np.abs(idxs - idx_expected)
    dmin = int(d.min())
    close_mask = (d == dmin)

    # TIE-BREAK: if multiple at same distance, pick loudest
    cand_idxs = idxs[close_mask]
    cand_vals = vals[close_mask]
    idx_best  = int(cand_idxs[int(np.argmax(cand_vals))])

    # Optional: allow “almost as close” candidates within tie_radius_sec and pick loudest
    tie_rad = int(round(float(tie_radius_sec) * fs))
    if tie_rad > 0:
        near_mask = (d <= (dmin + tie_rad))
        near_idxs = idxs[near_mask]
        near_vals = vals[near_mask]
        # prefer loudest among the near set
        idx_loud_near = int(near_idxs[int(np.argmax(near_vals))])
        # but don’t let it jump too far
        idx_best = idx_loud_near

    # Guardrail: do not jump far away (glitch capture)
    max_off = int(round(float(max_offset_sec) * fs))
    if abs(idx_best - idx_expected) > max_off:
        # clamp back to the best *within* max_off if any exist; else just return idx_expected
        in_mask = np.abs(idxs - idx_expected) <= max_off
        if np.any(in_mask): in_idxs = idxs[in_mask]; in_vals = vals[in_mask]; idx_best = int(in_idxs[int(np.argmax(in_vals))])
        else: idx_best = int(np.clip(idx_expected, 0, N - 1))

    # ---- Sub-sample refinement (parabolic fit) ----
    # Use the ORIGINAL env (not w) so we refine on the true envelope.
    i = int(np.clip(idx_best, 1, N - 2))
    y0 = float(env[i - 1]); y1 = float(env[i]); y2 = float(env[i + 1])

    den = (y0 - 2.0 * y1 + y2)
    if abs(den) > EPS_FLOOR: frac = 0.5 * (y0 - y2) / den; frac = float(np.clip(frac, -0.5, 0.5))   # in [-0.5, +0.5] for a nice peak
    else: frac = 0.0

    return idx_best, float(i) + frac

def Sub_Align(N, umh_w_full, idx_ligo_peak, search_half):
    # Recompute UMH envelope and refine peak near LIGO peak
    env_umh = np.abs(hilbert(umh_w_full))
    j0c     = max(0, idx_ligo_peak - search_half)
    j1c     = min(N, idx_ligo_peak + search_half)
    idx_center = j0c + int(np.argmax(env_umh[j0c:j1c]))

    # Sub-sample refinement around idx_center (parabolic fit)
    k = idx_center
    if 1 <= k < N-1:
        y0, y1, y2 = env_umh[k-1], env_umh[k], env_umh[k+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) > EPS_SAFE_FLOOR:
            delta = 0.5 * (y0 - y2) / denom   # in samples, between ~[-0.5, 0.5]
            idx_center_sub   = k + delta
        else: idx_center_sub = float(k)
    else: idx_center_sub     = float(k)

    return idx_center, idx_center_sub


def choose_ridge_stft_params(fs: float, seg_len: int, ridge_win_sec: float | None = None,
                            T_win: float = 0.04, T_hop: float = 0.004, n_bins_min: int = 20) -> tuple[int, int]:
    """
    Choose STFT params for ridge tracking: emphasize time resolution + enough time bins in the ridge window.
    Returns (NPER_RIDGE, NOVER_RIDGE).
    """
    if ridge_win_sec is None: ridge_win_sec = seg_len / fs

    # Window length from target, rounded to power-of-2 for FFT efficiency
    nper = _round_pow2(int(round(fs * T_win)))
    # Keep in a stable chirp-friendly range (avoid segment-length coupling)
    nper = int(max(256, min(nper, 1024, seg_len)))
    # Hop from target, but clamp for ridge stability (2–10 ms typical)
    hop = int(round(fs * T_hop))
    hop = int(max(int(round(0.002 * fs)), min(hop, int(round(0.010 * fs)), nper - 1)))
    # Ensure enough bins in the ridge window
    hop_max = max(1, int((ridge_win_sec / float(n_bins_min)) * fs))
    hop = min(hop, hop_max)
    hop = max(1, min(hop, nper - 1))
    return nper, int(nper - hop)


def meas_delay_xcorr_sec(fs, a1_w, b1_w, idx_center, halfwin_sec=0.15, maxlag_sec=0.01):
    N = len(a1_w)
    hw = int(round(halfwin_sec * fs))
    i0 = max(0, int(idx_center) - hw)
    i1 = min(N, int(idx_center) + hw)

    x = np.asarray(a1_w[i0:i1], float)
    y = np.asarray(b1_w[i0:i1], float)

    # normalize to prevent amplitude dominating
    x = x - x.mean(); y = y - y.mean()
    x /= (np.linalg.norm(x) + 1e-24)
    y /= (np.linalg.norm(y) + 1e-24)

    maxlag = int(round(maxlag_sec * fs))
    lags = np.arange(-maxlag, maxlag + 1, dtype=int)

    # correlation at each lag (y shifted relative to x)
    corr_vals = np.empty(len(lags), dtype=float)

    best_k = 0
    best_val = -1.0
    for k, lag in enumerate(lags):
        if lag < 0: val = float(np.dot(x[-lag:], y[:len(y)+lag]))
        elif lag > 0: val = float(np.dot(x[:len(x)-lag], y[lag:]))
        else: val = float(np.dot(x, y))

        aval = abs(val); corr_vals[k] = aval      
        if aval > best_val: best_val = aval; best_k = k

    best_lag = int(lags[best_k])

    # ---- Sub-sample refinement (parabolic fit around the peak) ----
    # Use the objective we maximized (abs correlation) for refinement consistency.
    delta = 0.0
    if 0 < best_k < (len(corr_vals) - 1):
        c1 = float(corr_vals[best_k - 1])
        c2 = float(corr_vals[best_k])
        c3 = float(corr_vals[best_k + 1])

        den = (c1 - 2.0*c2 + c3)
        if abs(den) > 1e-30:
            delta = 0.5 * (c1 - c3) / den
            # clamp to keep it sane
            if delta < -0.5: delta = -0.5
            if delta >  0.5: delta =  0.5

    best_lag_refined = float(best_lag) + float(delta)

    # return refined delay + peak strength
    return best_lag / fs, best_lag_refined / fs, best_val
# ---- End Helper Functions ----


# ------------ Fractional-delay and Coarse Alignment utilities ------------
def fractional_delay_fft(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """
    Apply a fractional time shift x(t - tau_s) via frequency-domain phase ramp.
    Positive tau_s delays (shifts to the right in time).
    """
    N = len(x)
    # zero-pad to reduce circular wrap when shifting
    pad = N // 2
    xpad = np.pad(x, (pad, pad), mode='constant')
    X = rfft(xpad)
    freqs = rfftfreq(len(xpad), d=1.0/fs)
    phase = np.exp(-1j * 2.0 * np.pi * freqs * tau_s)
    ypad = irfft(X * phase, n=len(xpad)).real
    # unpad back to original length
    return ypad[pad:-pad]


def coarse_align_template(ligo_w_full, umh_w, fs, t_min=10.0, t_max=22.0):
    """
    Coarse alignment for GW150914-like event.
    Treat umh_w as a short template; slide it over ligo_w_full (whitened)
    and find the best start_idx in [t_min, t_max].

    Returns (corr_max, start_index), with corr_max in [0,1].
    """
    x = np.asarray(ligo_w_full, float)
    h = np.asarray(umh_w,  float)

    # zero-mean, unit-std globally
    x = (x - np.mean(x)) / (np.std(x) + EPS_FLOOR)
    h = (h - np.mean(h)) / (np.std(h) + EPS_FLOOR)

    N_x = len(x)
    N_h = len(h)
    if N_h >= N_x: return 0.0, 0 # fallback: no sliding possible

    # valid-mode correlation: r[start_idx] = sum_{i=0..N_h-1} x[start_idx+i]*h[i]
    r = correlate(x, h, mode="valid")  # length = N_x - N_h + 1

    # convert to correlation coefficient; both are unit-variance
    denom = (N_h + EPS_FLOOR)
    rho = r / denom

    # allowed start_idx indices by time
    starts = np.arange(len(rho))
    t = starts / fs
    mask = (t >= t_min) & (t <= t_max)
    if not np.any(mask): mask = slice(None)  # if config is weird, fall back to global

    rho_sub = rho[mask]
    starts_sub = starts[mask]

    if rho_sub.size == 0: return 0.0, 0

    k = int(np.argmax(np.abs(rho_sub)))
    start_idx = int(starts_sub[k])
    coeff = float(rho_sub[k])

    # polarity: if negative, flip sign only
    if coeff < 0: coeff = -coeff  # we won't flip template here; sign can be absorbed later

    # bound to [0,1]
    coeff = max(min(coeff, 1.0), 0.0)

    return coeff, start_idx

def align_umh_to_global(config, detector, fs, N_ligo, start_idx, t_merge_obs, umh_w_full, umh_cond_full,
                        anchor_i0_snr, anchor_i1_snr, anchor_idx_center, geom_delay_sec_eff):

    delta_geom_sec       = float(geom_delay_sec_eff)

    # Target event time for this detector: anchor event time + geometric delay
    t_target     = (float(anchor_idx_center) / float(fs)) + delta_geom_sec
    idx_target_f = t_target * float(fs)           # float sample index (exact)
    idx_target   = int(round(idx_target_f))       # integer index (for windows/diagnostics only)

    # Deterministic UMH merge location on the LIGO timeline:
    # UMH template t=0 placed at start_idx; merge happens at t_merge_obs
    idx_merge_f = float(start_idx) + float(t_merge_obs) * float(fs)   # NO rounding

    # Exact fractional delay needed so UMH merge lands on geometry target (no rounding)
    tau = (idx_target_f - idx_merge_f) / float(fs)
    # Diagnostic: nearest-sample lag equivalent (do not use for shifting)
    peak_lag = int(round(idx_target_f - idx_merge_f))

    # Apply the fractional delay to both whitened and conditioned UMH
    if abs(tau) > 1e-15:
        umh_w_full    = fractional_delay_fft(umh_w_full,    fs, tau)
        umh_cond_full = fractional_delay_fft(umh_cond_full, fs, tau)

    # For plotting/diagnostics only:
    idx_merge_loc = int(round(idx_merge_f + tau * fs))
    idx_merge_loc = int(np.clip(idx_merge_loc, 0, len(umh_w_full) - 1))

    # Window placement: shift windows by the *actual* integer target relative to anchor center
    shift_samples = int(idx_target - int(anchor_idx_center))
    i0_snr     = int(anchor_i0_snr)     + shift_samples
    i1_snr     = int(anchor_i1_snr)     + shift_samples
    idx_center = int(anchor_idx_center) + shift_samples

    # Purely diagnostic: measure residual lag near target (should be ~0 if everything is consistent)
    search_half_sec   = float(config.get("FIT_SEARCH_HALF_SEC", 0.2))
    search_half       = int(search_half_sec * fs)
    _, idx_center_sub = Sub_Align(N_ligo, umh_w_full, idx_target, search_half)
    lag_meas_sec      = (idx_center_sub - idx_target) / fs

    return umh_w_full, umh_cond_full, peak_lag, i0_snr, i1_snr, idx_center, idx_merge_loc, tau, shift_samples, delta_geom_sec, lag_meas_sec
# ------------ Fractional-delay and Coarse Alignment utilities ------------


# ------------ Not used in Physics Strict Mode: Optional: tiny time-stretch search (+/-2%) ------------
def time_stretch_about_anchor(x, s, anchor_idx, dtype=np.float64):
    """
    Stretch x by factor s about anchor_idx, keeping the same length.
    If s>1, the signal is dilated (chirp slows); if s<1, compressed.
    """
    n = len(x)
    t  = np.arange(n, dtype=dtype) #dtype
    # map destination samples back to source locations, centered at anchor
    t_src = (t - anchor_idx) / s + anchor_idx
    # clamp to [0, n-1] and interpolate
    t_src = np.clip(t_src, 0.0, n - 1.0)
    return np.interp(t_src, t, x)


def best_stretch_by_corr(target, template, fs, s_min=0.95, s_max=1.05, n_steps=41, dtype=np.float64):
    """
    Search over stretch factors s in [s_min, s_max], warping `template` in time
    while keeping length = len(target). Returns (best_s, warped_template, best_corr).
    Correlation is cosine similarity (zero-mean, unit-norm) to avoid amplitude bias.
    """
    target   = np.asarray(target, dtype=dtype)
    template = np.asarray(template, dtype=dtype)

    N = len(target)
    if len(template) < N:
        # pad template if shorter
        pad = np.zeros(N, dtype=dtype)
        pad[:len(template)] = template
        template = pad
    elif len(template) > N: template = template[:N]

    # zero-mean target once
    tgt = target - np.mean(target)
    tgt_norm = np.linalg.norm(tgt) + EPS_FLOOR

    t      = np.arange(N) / fs
    t_mid  = t.mean()
    s_grid = np.linspace(s_min, s_max, n_steps)

    best_corr = -np.inf
    best_s    = 1.0
    best_warp = template.copy()

    for s in s_grid:
        # time-warp about the center so the merger doesn’t drift to an edge
        t_src = (t - t_mid) / s + t_mid
        # keep indices in range to avoid NaNs
        t_src = np.clip(t_src, 0.0, (N - 1) / fs)

        # linear interpolation (keeps length)
        warped = np.interp(t, t_src, template)

        # zero-mean both and compute cosine similarity
        wzm   = warped - np.mean(warped)
        denom = (np.linalg.norm(wzm) + EPS_FLOOR) * tgt_norm
        corr  = float(np.dot(wzm, tgt) / denom)

        if corr > best_corr:
            best_corr = corr
            best_s    = s
            best_warp = warped

    return best_s, best_warp.astype(np.float64, copy=False), best_corr
# ------------ End Not used in Physics Strict Mode: Optional: tiny time-stretch search (+/-2%) ------------


def compute_asd(signal_data, fs, nperseg=4096, noverlap=2048): #0.875
    """
    Compute one-sided amplitude spectral density (ASD) in strain/√Hz.
    """
    window = get_window("hann", nperseg)
    #noverlap = int(nperseg * overlap)
    freqs, psd = welch(signal_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density", average="mean") #median
    # Convert two-sided PSD to one-sided ASD
    asd = np.sqrt(psd)
    return freqs, asd


def amp_proxy_unwhitened(sig, pctl=99.0):
    """
    Robust amplitude proxy for PSI/IOTA scoring.
    Uses Hilbert envelope percentile on *unwhitened* (but bandpassed/conditioned) strain.
    """
    try:
        sig = np.asarray(sig, dtype=float)
        if sig.size < 8: return 0.0
        env = np.abs(hilbert(sig))
        env = env[np.isfinite(env)]
        if env.size == 0: return 0.0
        p = float(np.clip(pctl, 50.0, 100.0))
        return float(np.percentile(env, p))
    except Exception: return 0.0


def inst_freq_hz(sig, fs, smooth_sec=0.012, polyorder=3, min_win_sec=0.004, max_win_sec=0.050): # 12 ms smoothing time (configurable), minimum smoothing (4 ms), cap smoothing (50 ms)
    """
    Instantaneous frequency via Hilbert phase. Phase is smoothed in time (seconds) before differentiation to reduce jitter.
    """
    sig = np.asarray(sig, float)
    analytic = hilbert(sig)
    phase = np.unwrap(np.angle(analytic))

    n = len(phase)
    if n < 8: dphi_dt = np.gradient(phase) * fs; return dphi_dt / (2.0*np.pi)

    # Choose window in samples from physical time
    win_sec = float(np.clip(smooth_sec, min_win_sec, max_win_sec))
    win_len = int(round(win_sec * fs))
    # force odd and at least polyorder+2
    win_len = max(win_len | 1, polyorder + 2 + (polyorder + 2) % 2)

    # ensure valid
    if win_len >= n: win_len = (n - 1) | 1
    if win_len < (polyorder + 2): phase_s = phase
    else: phase_s = savgol_filter(phase, win_len, polyorder, mode="interp")

    dphi_dt = np.gradient(phase_s) * fs
    return dphi_dt / (2.0*np.pi)


def inst_freq_hz_trusted(config, fs, sig_l, sig_u, f_lo, f_hi):
    """
    Return (f_L, f_U, mask_L, mask_U) where masks select reliable instfreq samples.
    Mask criteria:
      - both analytic envelopes "loud enough" (fraction-of-peak or dB-below-peak)
      - away from edges (seconds or fraction)
      - instfreq within [f_lo, f_hi]
      - (optional) strictly positive
    """
    env_frac          = float(config.get("if_envfrac", 0.10))
    env_db_below_peak = config.get("if_env_db_below_peak", None)

    edge_sec          = float(config.get("if_edge_sec", 0.05))
    edge_frac         = config.get("if_edge_frac", None)

    require_positive  = bool(config.get("if_require_positive", True))

    smooth_sec        = float(config.get("if_smooth_sec", 0.012))
    polyorder         = int(config.get("if_polyorder", 3))
    min_win_sec       = float(config.get("if_min_win_sec", 0.004))
    max_win_sec       = float(config.get("if_max_win_sec", 0.050))

    sig_u = np.asarray(sig_u, float); sig_l = np.asarray(sig_l, float)

    f_L = inst_freq_hz(sig_l, fs, smooth_sec=smooth_sec, polyorder=polyorder, min_win_sec=min_win_sec, max_win_sec=max_win_sec)
    f_U = inst_freq_hz(sig_u, fs, smooth_sec=smooth_sec, polyorder=polyorder, min_win_sec=min_win_sec, max_win_sec=max_win_sec)

    env_U = np.abs(hilbert(sig_u)); env_L = np.abs(hilbert(sig_l))
    peak_U = float(env_U.max() + EPS_FLOOR); peak_L = float(env_L.max() + EPS_FLOOR)

    if env_db_below_peak is not None:
        scale = 10.0 ** (-float(env_db_below_peak) / 20.0)
        thr_U = peak_U * scale; thr_L = peak_L * scale
    else: thr_U = env_frac * peak_U; thr_L = env_frac * peak_L

    base_mask = (env_U >= thr_U) & (env_L >= thr_L)
    mask_L = base_mask.copy(); mask_U = base_mask.copy()

    # Edge exclusion
    n_L = len(f_L); n_U = len(f_U)
    if edge_frac is not None: e_L = int(float(edge_frac) * n_L); e_U = int(float(edge_frac) * n_U)
    else: e_L = e_U = int(round(edge_sec * fs))

    if e_L > 0 and 2 * e_L < n_L: mask_L[:e_L] = False; mask_L[-e_L:] = False
    if e_U > 0 and 2 * e_U < n_U: mask_U[:e_U] = False; mask_U[-e_U:] = False

    # Band sanity
    mask_L &= np.isfinite(f_L) & (f_L >= float(f_lo)) & (f_L <= float(f_hi))
    mask_U &= np.isfinite(f_U) & (f_U >= float(f_lo)) & (f_U <= float(f_hi))
    if require_positive: mask_L &= (f_L > 0.0); mask_U &= (f_U > 0.0)

    return f_L, f_U, mask_L, mask_U


def estimate_psd_from_ligo(x, fs, nperseg=1024, noverlap=768, dtype=np.float64):
    """
    Welch PSD, one-sided, units: strain^2/Hz (density).
    """
    x = np.asarray(x, dtype=dtype)
    if noverlap is None: noverlap = nperseg // 2
    # Force zero-mean before PSD to avoid a huge DC spike
    x = x - np.mean(x)
    # Hann window, density scaling (power/Hz)
    f_psd, Pxx = welch(
        x, fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="median",
    )
    # Floor to avoid zeros in whitening
    Pxx = np.maximum(Pxx, PSD_FLOOR)
    return f_psd, Pxx


def whiten_with_psd(x, fs, f_psd, Pxx, dtype=np.float64):
    """
    Whiten a real signal using a *one-sided* PSD (Welch density).
    Result should have ~unit variance if PSD is accurate.
    """
    x = np.asarray(x, dtype=dtype)
    N = len(x)

    # Remove residual DC before FFT
    x = x - np.mean(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Interpolate PSD onto rfft frequency grid
    P_i = np.interp(freqs, f_psd, Pxx, left=Pxx[0], right=Pxx[-1])
    P_i = np.maximum(P_i, PSD_FLOOR)

    # Correct normalization for one-sided PSD with rfft:
    # PSD is power/Hz; rfft bins correspond to Δf = fs/N; amplitude scale needs √(PSD * fs/2)
    denom = np.sqrt(P_i * fs / 2.0)
    Xw = X / denom

    xw = np.fft.irfft(Xw, n=N)

    # Final clean-up
    xw -= np.mean(xw)
    return xw


# --- Visual Enhancements Not Applied to Physics ---
def _aligodesign_asd(f: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Approximate aLIGO design amplitude spectral density (Hanford-like), in 1/sqrt(Hz).
    Valid-ish over ~10–2000 Hz. Outside that, we just clamp.
    """
    PSD_NORM  = 1e-49       # physical normalization

    f = np.asarray(f, dtype=dtype) #dtype
    # Avoid division by zero
    f = np.maximum(f, 1.0)

    # Analytic fit (Ajith et al style) — gives the 'bucket' near ~100 Hz
    x = f / 215.0
    psd = (PSD_NORM) * (x**(-4.14) - 5.0 * x**(-2.0) + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))

    # We only care about a smooth positive curve
    psd = np.maximum(psd, PSD_NORM)
    return np.sqrt(psd)  # ASD = sqrt(PSD)


# Optional Utility: generate approximate aLIGO-like colored noise.
# Used ONLY for visualization overlays; never for setting normalization.
def make_ligo_psd_noise(N: int, dt: float, target_rms: float = 1.0, rng: np.random.Generator | None = None, dtype=np.float64) -> np.ndarray:
    if rng is None: rng = np.random.default_rng(None)

    fs = 1.0 / float(dt)
    freqs = np.fft.rfftfreq(N, d=dt)

    # aLIGO design ASD and PSD
    asd = _aligodesign_asd(freqs).astype(dtype)
    psd = asd**2

    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / max(N, 1)

    re = rng.normal(0.0, 1.0, len(freqs))
    im = rng.normal(0.0, 1.0, len(freqs))
    im[0] = 0.0  # DC real

    sigma = np.sqrt(0.5 * psd * df)
    coeffs = (re + 1j * im) * sigma

    noise = np.fft.irfft(coeffs, n=N)

    #rms = float(np.sqrt(np.mean(noise**2)))
    rms = float(stable_rms(noise))
    if not np.isfinite(rms) or rms < EPS_FLOOR or target_rms <= 0.0: noise[:] = 0.0
    else: noise *= (target_rms / rms)

    return noise.astype(float)


# Optional Utility: simple whitening for plots and sanity checks.
# IMPORTANT: Not used when writing the physics NPZ; does not affect strain_records saved for analysis.
def whiten_for_display(y: np.ndarray, dt: float) -> np.ndarray:
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dt)
    asd = _aligodesign_asd(freqs)
    Yw = Y / (asd + EPS_SAFE_FLOOR)
    return np.fft.irfft(Yw, n=len(y))


# Provide a better CMAP feel for Spectrogram Visual. Custom colormap for spectrogram plots only; purely aesthetic.
def ligo_cmap():
    base = plt.get_cmap("viridis")
    colors = base(np.linspace(0, 1, 256))

    # --- Gentle adjustments ---
    # Dark end: shift purple → blue-green
    colors[:40, 0] *= 0.6      # reduce red
    colors[:40, 1] *= 0.8      # keep some green for balance
    colors[:40, 2] *= 1.15     # boost blue → cooler tone

    # Midrange: keep viridis shape but slightly brighter
    colors[80:160, :] *= 1.03

    # Highlights: preserve yellow/green but avoid clipping to white
    colors[200:, 2] *= 0.95
    colors = np.clip(colors, 0, 1)

    return mcolors.LinearSegmentedColormap.from_list("ligo_soft", colors)
# --- End Visual Enhancements Not Applied to Physics ---

def condition_time_domain(x, fs, f_lo, f_hi, notch_lines=(), dtype=np.float64):
    """
    Conditioning used for 'physics' overlays and residuals:
      - zero-mean
      - zero-phase bandpass [f_lo, f_hi]
      - apply the same narrow notches as used for LIGO data
      - zero-mean again

    This keeps signals in (approximately) physical strain units,
    with identical filtering for UMH and LIGO.
    """
    x = np.asarray(x, dtype); x = x - np.mean(x)
    # small taper reduces filtfilt edge transients
    #x = x * tukey(len(x), alpha=0.05)
    x = bandpass(x, fs, f_lo=f_lo, f_hi=f_hi, order=4)
    if notch_lines: x = apply_notches(x, fs, lines=notch_lines, Q=30)
    x = x - np.mean(x)
    # critical: prevent absurd spikes from propagating
    x = sanitize(x, name="cond_full")
    return x

def estimate_event_time_seconds_hilbert(x, fs, t_bounds=(10.0, 22.0), debug_tag="", dtype=np.float64):
    """
    Event-time via Hilbert magnitude |analytic(x)| within [t_min, t_max].
    More selective than moving-RMS for loud, short bursts.
    """
    x = np.asarray(x, dtype) #dtype?
    N = len(x)
    if N < 8: return (N/2)/float(fs), int(N/2)

    # Hilbert envelope on *conditioned* (bandpassed+notched) data
    env = np.abs(hilbert(x))

    t_min, t_max = float(t_bounds[0]), float(t_bounds[1])
    i0 = int(np.clip(round(t_min * fs), 0, N-1))
    i1 = int(np.clip(round(t_max * fs), i0+1, N))

    sub = env[i0:i1]
    if sub.size == 0:
        idx = int(np.argmax(env))
        return idx/float(fs), idx

    # pick the single largest local maximum in [i0,i1)
    j = int(np.argmax(sub))
    idx = i0 + j

    #if debug_tag: print(f"[{debug_tag}] hilbert-event: i0={i0}, i1={i1}, idx={idx}, t={idx/fs:.6f}")

    return idx/float(fs), int(idx)


def matched_filter_snr_window(config, detector, fs, N_ligo, ligo_w_full, umh_w_full, idx_center, tau, i0, i1, dtype=np.float64):
    i0_snr = i0; i1_snr = i1

    if(False):
        # --- AUTO SNR WINDOW based on UMH signal support ---
        snr_auto = bool(config.get("SNR_AUTO_FULL_CHIRP", False))
        if snr_auto:
            # Use the (conditioned) template to locate where the chirp actually lives
            sig = np.asarray(umh_w_full, dtype=float)
            env = np.abs(hilbert(sig))

            if np.all(env == 0.0):
                # Fallback: no visible template (shouldn't happen), revert to old config-based window
                snr_before_sec = float(config.get("SNR_WIN_BEFORE_SEC", 0.25))
                snr_after_sec  = float(config.get("SNR_WIN_AFTER_SEC", 0.35))
                snr_win_before = int(snr_before_sec * fs)
                snr_win_after  = int(snr_after_sec  * fs)
                i0_snr = int(round(max(0, idx_center - snr_win_before)))
                i1_snr = int(round(min(N_ligo, idx_center + snr_win_after)))
            else:
                # Threshold: use points where envelope is at least, say, 5% of its max
                thr = 0.05 * float(env.max())
                active = env >= thr

                if not np.any(active):
                    # Extremely conservative fallback: treat as above
                    snr_before_sec = float(config.get("SNR_WIN_BEFORE_SEC", 0.25))
                    snr_after_sec  = float(config.get("SNR_WIN_AFTER_SEC", 0.35))
                    snr_win_before = int(snr_before_sec * fs)
                    snr_win_after  = int(snr_after_sec  * fs)
                    i0_snr = int(round(max(0, idx_center - snr_win_before)))
                    i1_snr = int(round(min(N_ligo, idx_center + snr_win_after)))
                else:
                    # First/last index where template is "on"
                    j0 = int(np.argmax(active))
                    j1 = int(len(active) - 1 - np.argmax(active[::-1]))

                    # Add padding (e.g. 50 ms on each side)
                    pad_samples = int(0.05 * fs)  # 50 ms
                    i0_snr = int(round(max(0, j0 - pad_samples)))
                    i1_snr = int(round(min(N_ligo, j1 + pad_samples)))
                    #print(f"[{detector}] SNR Auto")

        else:
            # --- CONFIG-BASED WINDOW ---
            snr_before_sec = float(config.get("SNR_WIN_BEFORE_SEC", 0.25))
            snr_after_sec  = float(config.get("SNR_WIN_AFTER_SEC", 0.35))
            snr_win_before = int(snr_before_sec * fs)
            snr_win_after  = int(snr_after_sec  * fs)
            #print(f"N_ligo={N_ligo}, idx_center={idx_center}, snr_before_sec={snr_before_sec}, snr_after_sec={snr_after_sec}, snr_win_before={snr_win_before}, snr_win_after={snr_win_after}")
            i0_snr = int(round(max(0, idx_center - snr_win_before)))
            i1_snr = int(round(min(N_ligo, idx_center + snr_win_after)))

    #print(f"[{detector}] i0_snr={i0_snr:.3f} i1_snr={i1_snr:.3f}")

    # Window the data/template (whitened)
    L = np.asarray(ligo_w_full[i0_snr:i1_snr], dtype)
    U = np.asarray(umh_w_full[i0_snr:i1_snr], dtype)

    # Full linear cross-correlation (whitened domain)
    xc = np.correlate(L, U, mode="full")
    lags = np.arange(-len(U)+1, len(L))

    # Peak lag
    k = int(np.argmax(np.abs(xc)))
    lag_samp = int(lags[k])
    
    if(bool(config.get("ALLOW_LAG_MAX", False))): lag_samp_eff = lag_samp
    else: lag_samp_eff = 0 #-int(round(tau * fs))

    # ----- (A) TRUE SNR at the peak lag -----
    # Align template at lag for a single-point inner product SNR
    if lag_samp_eff > 0: U_al = np.r_[np.zeros(lag_samp_eff), U[:-lag_samp_eff]]
    elif lag_samp_eff < 0: k2 = -lag_samp_eff; U_al = np.r_[U[k2:], np.zeros(k2)]
    else: U_al = U

    # SNR = (L|U_al) / ||U_al||
    num_snr = float(np.vdot(L, U_al).real)
    den_snr = float(np.linalg.norm(U_al) + EPS_FLOOR)
    rho_peak_signed = num_snr / den_snr
    rho_peak_abs    = abs(rho_peak_signed)

    # ----- (B) MATCH (cosine similarity) for diagnostics -----
    num_m = float(np.vdot(L, U_al).real)
    den_m = float((np.linalg.norm(L) * np.linalg.norm(U_al)) + EPS_FLOOR)
    match_lagged = num_m / den_m

    # Pick a reference index inside the (windowed) template U: envelope peak
    envU = np.abs(hilbert(U))
    edge_frac = float(config.get("FIT_EDGE_EXCLUDE_FRAC", 0.05))  # Small.
    edge      = int(edge_frac * len(U))
    if edge*2 < len(U): idx_ref = int(edge + np.argmax(envU[edge:-edge]))
    else: idx_ref = int(np.argmax(envU))
    # lag_samp_eff definition: U[0] aligns with L[lag_samp_eff]
    idx_in_window = idx_ref + lag_samp_eff
    # clamp to window range to avoid weird negatives/overflow
    idx_in_window = int(np.clip(idx_in_window, 0, len(L)-1))
    t_peak = (i0_snr + idx_in_window) / fs

    print(f"[{detector}] i0={i0_snr/fs:.3f} i1={i1_snr/fs:.3f} idx_ref={idx_ref} lag={lag_samp} lag_eff={lag_samp_eff} t_peak={t_peak:.6f}")
    print(f"tau={tau} idx_center={idx_center} lag={lag_samp}")

    return rho_peak_signed, rho_peak_abs, t_peak, match_lagged, lag_samp, i0_snr, i1_snr


def chirp_mass_from_track(t, f, mask, f_min=30.0, f_max=90.0, trim_lo=10.0, trim_hi=90.0,
                          nbin=16, minbinpts=6, minbins=6):
    """
    Robust single-track chirp-mass proxy from an IF track.

    Uses the standard inspiral scaling:
        df/dt = K * Mc^(5/3) * f^(11/3)
    =>  Mc_tilde ∝ median( (df/dt) / f^(11/3) )^(3/5)
    """
    t = np.asarray(t, float); f = np.asarray(f, float)
    mask = np.asarray(mask, bool)

    m = mask & np.isfinite(t) & np.isfinite(f) & (f >= f_min) & (f <= f_max)
    idx = np.where(m)[0]
    if idx.size < 12: return float("nan"), int(idx.size)

    # Sort by time, keep alignment
    tt = t[idx]; ff = f[idx]
    o = np.argsort(tt)
    tt = tt[o]; ff = ff[o]

    # Guard against non-increasing time (rare but possible)
    dt = np.diff(tt)
    keep = np.ones_like(tt, dtype=bool)
    keep[1:] = dt > 0
    tt = tt[keep]; ff = ff[keep]
    if tt.size < 12: return float("nan"), int(tt.size)

    # df/dt
    try: dfdt = np.gradient(ff, tt)
    except Exception: return float("nan"), int(tt.size)

    # y = (df/dt) / f^(11/3)
    y = dfdt / np.power(np.maximum(ff, 1e-9), 11.0/3.0)

    # Finite + positive only (inspiral chirp should have df/dt > 0)
    ok = np.isfinite(y) & np.isfinite(ff) & (y > 0)
    ff = ff[ok]; y = y[ok]
    if y.size < 12: return float("nan"), int(y.size)

    # Trim/winsorize y to kill catastrophic spikes (track holes -> crazy df/dt)
    if y.size >= 20: lo, hi = np.percentile(y, [trim_lo, trim_hi]); y = np.clip(y, lo, hi)

    # Frequency binning of median(y) to suppress local garbage dominating
    nbin = max(8, int(nbin))
    edges = np.linspace(float(np.min(ff)), float(np.max(ff)), nbin + 1)

    yb, wb = [], []
    for i in range(nbin):
        mm = (ff >= edges[i]) & (ff < edges[i+1])
        cnt = int(np.count_nonzero(mm))
        if cnt >= int(minbinpts): yb.append(float(np.median(y[mm]))); wb.append(float(cnt))

    yb = np.asarray(yb, float); wb = np.asarray(wb, float)
    if yb.size < int(minbins):
        # Fall back: median of trimmed y
        y_med = float(np.median(y))
        Mc_tilde = float(np.power(max(y_med, EPS_FLOOR), 3.0/5.0))
        return Mc_tilde, int(y.size)

    # Weighted median-ish: take median of binned medians (simple + robust)
    y_med = float(np.median(yb)); Mc_tilde = float(np.power(max(y_med, EPS_FLOOR), 3.0/5.0))

    # n_used = total points used before binning (still meaningful)
    return Mc_tilde, int(y.size)


def chirp_mass_ratio_from_tracks(t, fL, fU, mask, f_min_for_mass=30.0, f_max_for_mass=180.0,
                                 trim_lo=10.0, trim_hi=90.0, nbin=16, minbinpts=6, minbins=6):
    """
    Robust ratio of chirp-mass proxies between two tracks.
    Computes y = (df/dt)/f^(11/3) for each track, trims, bins by frequency,
    then forms ratio from per-bin medians:
        r_bin = yU_bin / yL_bin
        r_med = median(r_bin)
        Mc_ratio = r_med^(3/5)
    """
    t  = np.asarray(t, float); fL = np.asarray(fL, float); fU = np.asarray(fU, float)
    mask = np.asarray(mask, bool)

    m = mask & np.isfinite(t) & np.isfinite(fL) & np.isfinite(fU)
    m &= (fL >= f_min_for_mass) & (fL <= f_max_for_mass)
    m &= (fU >= f_min_for_mass) & (fU <= f_max_for_mass)
    idx = np.where(m)[0]
    if idx.size < 12: return float("nan"), float("nan"), int(idx.size)

    # Sort by time (aligns gradients), keep alignment
    tt = t[idx]; fLl = fL[idx]; fUu = fU[idx]; o = np.argsort(tt)
    tt = tt[o]; fLl = fLl[o]; fUu = fUu[o]

    # Ensure strictly increasing time
    dt = np.diff(tt)
    keep = np.ones_like(tt, dtype=bool); keep[1:] = dt > 0
    tt = tt[keep]; fLl = fLl[keep]; fUu = fUu[keep]
    if tt.size < 12: return float("nan"), float("nan"), int(tt.size)

    # Gradients
    try: dfLdt = np.gradient(fLl, tt); dfUdt = np.gradient(fUu, tt)
    except Exception: return float("nan"), float("nan"), int(tt.size)

    yL = dfLdt / np.power(np.maximum(fLl, 1e-9), 11.0/3.0)
    yU = dfUdt / np.power(np.maximum(fUu, 1e-9), 11.0/3.0)

    ok = np.isfinite(yL) & np.isfinite(yU) & np.isfinite(fLl) & np.isfinite(fUu) & (yL > 0) & (yU > 0)
    fLl = fLl[ok]; fUu = fUu[ok]; yL = yL[ok]; yU = yU[ok]
    if yL.size < 12: return float("nan"), float("nan"), int(yL.size)

    # Trim yL and yU independently (kills spikes)
    if yL.size >= 20:
        loL, hiL = np.percentile(yL, [trim_lo, trim_hi])
        loU, hiU = np.percentile(yU, [trim_lo, trim_hi])
        yL = np.clip(yL, loL, hiL); yU = np.clip(yU, loU, hiU)

    # Bin edges based on LIGO track freq (stable reference)
    nbin = max(8, int(nbin))
    edges = np.linspace(float(np.min(fLl)), float(np.max(fLl)), nbin + 1)

    r_bins = []
    for i in range(nbin):
        mm = (fLl >= edges[i]) & (fLl < edges[i+1])
        cnt = int(np.count_nonzero(mm))
        if cnt >= int(minbinpts):
            yLm = float(np.median(yL[mm])); yUm = float(np.median(yU[mm]))
            if np.isfinite(yLm) and np.isfinite(yUm) and (yLm > 0): r_bins.append(yUm / max(yLm, EPS_FLOOR))

    r_bins = np.asarray(r_bins, float)
    if r_bins.size < int(minbins):
        # Fallback: median of pointwise ratio but safer (clip denom)
        r = yU / np.maximum(yL, EPS_FLOOR); r = r[np.isfinite(r) & (r > 0)]
        if r.size < 8: return float("nan"), float("nan"), int(r.size)
        r_med = float(np.median(r))
        Mc_ratio = float(np.power(r_med, 3.0/5.0))
        return Mc_ratio, r_med, int(r.size)

    r_med = float(np.median(r_bins))
    Mc_ratio = float(np.power(max(r_med, EPS_FLOOR), 3.0/5.0))
    return Mc_ratio, r_med, int(r_bins.size)


def deredshift_tf(t_obs, f_obs, z):
    """
    Standard mapping:
      f_src = f_obs * (1+z)
      t_src = t_obs / (1+z)
    Works as a clean diagnostic even if UMH interprets z differently physically.
    """
    z = float(z)
    fac = (1.0 + z)
    if fac <= 0: return t_obs, f_obs
    return (np.asarray(t_obs, float) / fac), (np.asarray(f_obs, float) * fac)



def chirp_diagnostics(config, detector, fs, f_min, f_merge, f_ref, ligo_seg, umh_seg, ligo_w_full, umh_w_full, k_phys, idx_merge_loc,
                          rho_signed, lag_samp, i0, i1, geom_delay_sec_raw=0.0, geom_delay_sec_eff=0.0, t_peak_abs=None, t_event_sec=None,
                          lag_meas_sec=None, anchor_t_peak_abs=None, BINARY_IOTA_DEG=None, pol_psi_deg=None, F_plus=None, F_cross=None, 
                          sign_pred_gen=None, global_pol=None, sign_corr=None, detector_polarity_flip_applied=False, lc_win=None, uc_win=None, 
                          ligo_xcorr_delay_sec=None, ligo_xcorr_strength=None, umh_xcorr_delay_sec=None, umh_xcorr_strength=None, 
                          ligo_cond_full = None, umh_cond_full = None, 
                          anchor_dt_env2_ligo = None, anchor_dt_env3_ligo = None, anchor_dt_env2_umh = None, anchor_dt_env3_umh = None, 
                          dtype=np.float64):

    # Optional: df/dt diagnostic band (can be narrower than main band)
    f_min_diag   = float(config.get("f_min_diag",   f_min))
    f_merge_diag = float(config.get("f_merge_diag", f_merge))

    NPER_RIDGE, NOVER_RIDGE = choose_ridge_stft_params(fs, len(ligo_seg), (len(umh_seg)/fs))
    NPER_RIDGE =  config.get("NPER_RIDGE",  NPER_RIDGE)
    NOVER_RIDGE = config.get("NOVER_RIDGE", NOVER_RIDGE)

    print(f"[{detector}] NPER_RIDGE:{NPER_RIDGE}, NOVER_RIDGE:{NOVER_RIDGE}")

    rho_abs = abs(rho_signed)

    seg_t0_abs = float(i0) / float(fs); N_ligo_seg = len(ligo_seg); t_vec = np.arange(N_ligo_seg) / fs

    if (anchor_t_peak_abs is None): t_peak_geom_abs = None
    else: t_peak_geom_abs = anchor_t_peak_abs + geom_delay_sec_eff

    # Defaults
    diag = {"fs": fs, "rho_peak_abs": rho_abs, "rho_signed": rho_signed, "rho_peak_time": None, 
            "geom_delay_sec_raw": geom_delay_sec_raw, "geom_delay_sec_eff": geom_delay_sec_eff,
            "sign_pol": None, "sign_pred_gen": sign_pred_gen, "sign_ant_pred": None, "sign_ant_match": None, "sign_vs_global": None, 
            "detector_polarity_flip_applied": detector_polarity_flip_applied, "BINARY_IOTA_DEG": BINARY_IOTA_DEG, "pol_psi_deg": pol_psi_deg, 
            "F_plus": F_plus, "F_cross": F_cross, "f_min": f_min, "f_merge": f_merge, "f_min_diag": f_min_diag, "f_merge_diag": f_merge_diag, "f_ref": f_ref, 
            "rms_ratio": None, "rms_ratio_resid": None, "ridge_conf": None, "f_mismatch_rms": None, "f_mismatch_mean": None, "f_mismatch_mean_rel": None, 
            "f_mismatch_rms_ridge": None, "f_mismatch_mean_ridge": None, "f_mismatch_rms_use": None, "f_mismatch_mean_use": None, 
            "t_event_est_diag": t_event_sec, "fit_window_s": None, "t_peak_abs": t_peak_abs, "anchor_t_peak_abs": anchor_t_peak_abs, 
            "t_peak_geom_abs": t_peak_geom_abs, "seg_t0_abs": seg_t0_abs, "j_peak_used": None, "t_peak_local_used": None, "lag_samples_peak": lag_samp, 
            "lag_meas_sec": lag_meas_sec, "k_phys": k_phys, "k_ls": None, "corr_window_signed": None, "corr_window": None, "corr_envelope": None, 
            "match_psd_window_signed": None, "match_psd_window": None, 
            "amp_obs_pctl": None, "amp_obs_unwhite_bp": None, "amp_umh_unwhite_bp": None,
            "USE_RIDGE_DIAGNOSTIC": None, "Mass_Slope_Ratio_Diagnostics": None, "Amplitude_Diagnostics": None, "Distance_Diagnostics": None,
            "Polarization_Inclination_Diagnostics": None, "Sky_Position_Diagnostics": None}
    
    resid_w = ligo_w_full - umh_w_full
    if(ligo_cond_full is not None and umh_cond_full is not None): resid_cond = ligo_cond_full - umh_cond_full
    else: resid_cond = None

    # --- Instantaneous frequency tracks (Hilbert, phase-smoothed) ---
    band_lo = config["band_lo"]; band_hi = config["band_hi"]
    f_L_s, f_U_s, mask_L_if, mask_U_if = inst_freq_hz_trusted(config, fs, ligo_seg, umh_seg, band_lo, band_hi)

    # --- Amplitude gating: only trust freq where both are "loud" ---
    A_L = np.abs(hilbert(ligo_seg))
    A_U = np.abs(hilbert(umh_seg))
    A_all = 0.5 * (A_L + A_U)
    
    idx_merge_seg = int(round(idx_merge_loc - i0))
    i0_tim = int(idx_merge_seg - 0.10 * fs)
    i1_tim = int(idx_merge_seg + 0.04 * fs)
    i0_tim = max(0, i0_tim); i1_tim = min(len(ligo_seg), i1_tim)
    if i1_tim <= i0_tim: i0_tim, i1_tim = 0, len(ligo_seg)
    xL = ligo_seg[i0_tim:i1_tim].astype(float)
    xU = umh_seg[i0_tim:i1_tim].astype(float)
    print(f"i0_tim:{i0_tim} i1_tim:{i1_tim} idx_merge_seg:{idx_merge_seg} seg_t0_abs:{seg_t0_abs}")
    A_L_twin = np.abs(hilbert(xL)); A_U_twin = np.abs(hilbert(xU))
    t_seg2 = seg_t0_abs + (i0_tim / fs) + np.arange(len(xL)) / fs; p2 = 2; p3 = 3
    dt_env2_ligo = np.sum(t_seg2 * (A_L_twin**p2)) / (np.sum(A_L_twin**p2) + EPS_FLOOR)
    dt_env3_ligo = np.sum(t_seg2 * (A_L_twin**p3)) / (np.sum(A_L_twin**p3) + EPS_FLOOR)
    dt_env2_umh  = np.sum(t_seg2 * (A_U_twin**p2)) / (np.sum(A_U_twin**p2) + EPS_FLOOR)
    dt_env3_umh  = np.sum(t_seg2 * (A_U_twin**p3)) / (np.sum(A_U_twin**p3) + EPS_FLOOR)
    if(anchor_dt_env2_ligo is None): anchor_dt_env2_ligo = dt_env2_ligo
    if(anchor_dt_env3_ligo is None): anchor_dt_env3_ligo = dt_env3_ligo
    if(anchor_dt_env2_umh is None): anchor_dt_env2_umh   = dt_env2_umh
    if(anchor_dt_env3_umh is None): anchor_dt_env3_umh   = dt_env3_umh

    # Base gating parameters (same for all detectors)
    A_all_percent_base =   float(config.get("A_all_percent",  65.00))
    time_half_width_base = float(config.get("time_half_width", 0.04))

    # --- Frequency band + finite values (diagnostic band) ---
    mask_f = (f_L_s > f_min_diag) & (f_L_s < f_merge_diag)
    mask_f &= np.isfinite(f_L_s) & np.isfinite(f_U_s)

    # --- Peak indices (keep BOTH) ---
    # Observed LIGO peak inside this segment
    if (t_peak_abs is not None) and np.isfinite(t_peak_abs):
        j_peak_ligo = int(round((float(t_peak_abs) - seg_t0_abs) * float(fs)))
        j_peak_ligo = int(np.clip(j_peak_ligo, 0, N_ligo_seg - 1))
    else: j_peak_ligo = int(np.argmax(np.abs(ligo_seg)))

    # Geometry-predicted peak inside this segment (optional)
    if (t_peak_geom_abs is not None) and np.isfinite(t_peak_geom_abs):
        j_peak_geom = int(round((float(t_peak_geom_abs) - seg_t0_abs) * float(fs)))
        j_peak_geom = int(np.clip(j_peak_geom, 0, N_ligo_seg - 1))
    else: j_peak_geom = None

    # Choose which one to use for df/dt gating around the peak
    USE_GEOM_PEAK_FOR_GATING = bool(config.get("USE_GEOM_PEAK_FOR_GATING", False))
    if USE_GEOM_PEAK_FOR_GATING and (j_peak_geom is not None): j_peak_gate = j_peak_geom
    else: j_peak_gate = j_peak_ligo
    t_peak_local = float(j_peak_gate) / float(fs)
    diag["j_peak_used"] = int(j_peak_gate); diag["t_peak_local_used"] = float(t_peak_local)
    
    # --- Adaptive gating loop: relax if too few points survive ---
    df_min_points = int(config.get("df_min_points", 256))

    # (amp_factor, time_factor) pairs: start tight, then progressively relax
    relax_plan = [
        (1.00, 1.0),
        (0.85, 1.5),
        (0.70, 2.0),
        (0.60, 2.5),
    ]

    mask = None
    for amp_factor, time_factor in relax_plan:
        A_percent = max(0.0, min(100.0, A_all_percent_base * amp_factor))
        A_thr = np.percentile(A_all, A_percent)
        mask_amp = (A_all >= A_thr)

        time_half_width = time_half_width_base * time_factor
        mask_t = np.abs(t_vec - t_peak_local) <= time_half_width

        mask_candidate = mask_L_if & mask_U_if & mask_amp & mask_f & mask_t
        n_candidate = int(mask_candidate.sum())

        if n_candidate >= df_min_points: mask = mask_candidate; break
        if mask is None or n_candidate > int(mask.sum()): mask = mask_candidate

    if mask is None: mask = np.zeros_like(f_L_s, dtype=bool)

    # Last-resort fallbacks
    if not np.any(mask):
        print(f"[{detector}] df/dt diag: primary mask empty; freq={mask_f.sum()}.")
        mask = mask_f & (np.abs(t_vec - t_peak_local) <= time_half_width_base * 2.0)
        print(f"[{detector}] fallback freq+time; count={mask.sum()}")

    if not np.any(mask): mask = mask_f; print(f"[{detector}] fallback freq-only; count={mask.sum()}")
    if not np.any(mask): mask = np.isfinite(f_L_s) & np.isfinite(f_U_s); print(f"[{detector}] fallback all-finite; count={mask.sum()}")

    # --- Amplitude-weighted RMS mismatch + hybrid ridge ---
    f_mismatch_rms, f_mismatch_mean, f_mismatch_mean_rel = float("nan"), float("nan"), float("nan")

    n_mask = int(mask.sum())
    if n_mask > 0:
        df = f_U_s[mask] - f_L_s[mask]
        A  = A_all[mask].astype(float)
        # keep only entries where both df and A are finite
        finite = np.isfinite(df) & np.isfinite(A)
        df = df[finite]; A  = A[finite]

        if df.size == 0: print(f"[{detector}] df/dt diag: no finite points after filtering.")
        else:
            # If no usable amplitude weights, fall back to unweighted
            if A.size == 0: f_mismatch_mean = float(np.mean(df)); f_mismatch_rms = float(stable_rms(df)) #float(np.sqrt(np.mean(df * df)))
            else:
                # normalize before squaring to prevent overflow
                Amax = float(np.max(A)) + EPS_SAFE_FLOOR
                An = A / Amax; w = An * An

                w_sum = float(w.sum())
                if w_sum > 0.0:
                    w /= w_sum
                    f_mismatch_mean = float(np.sum(w * df))
                    f_mismatch_rms  = float(np.sqrt(np.sum(w * (df * df))))
                else:
                    f_mismatch_mean = float(np.mean(df))
                    f_mismatch_rms  = float(stable_rms(df)) #float(np.sqrt(np.mean(df * df)))

            f_mismatch_mean_rel = float(f_mismatch_mean / max(f_ref, 1.0))
    else: print(f"[{detector}] df/dt diag: no valid points after fallbacks.")

    f_mismatch_rms_ridge, f_mismatch_mean_ridge, f_L_ridge = None, None, None
    ridge_conf = 0.0
    USE_RIDGE_DIAGNOSTIC = bool(config.get("USE_RIDGE_DIAGNOSTIC", True))
    if USE_RIDGE_DIAGNOSTIC:
        try:
            rho_lo  = float(config.get("RIDGE_RHO_LO",      6.0))
            rho_hi  = float(config.get("RIDGE_RHO_HI",     10.0))
            s_lo    = float(config.get("RIDGE_SMOOTH_LO",  0.18))
            s_mid   = float(config.get("RIDGE_SMOOTH_MID", 0.10))
            s_hi    = float(config.get("RIDGE_SMOOTH_HI",   0.0))   # no smoothing for loud events
            # Adaptive ridge smoothing based on SNR
            def apply_ridge_denoise(x, strength=0.2):
                """
                Very light smoothing to reduce vertical noise streaks. Preserves chirp shape.
                """
                smoothed = gaussian_filter1d(x, sigma=1.0)
                return (1-strength)*x + strength*smoothed
            if rho_abs < rho_lo: ligo_seg_ridge = apply_ridge_denoise(ligo_seg, strength=s_lo)
            elif rho_abs < rho_hi: ligo_seg_ridge = apply_ridge_denoise(ligo_seg, strength=s_mid)
            else: ligo_seg_ridge = apply_ridge_denoise(ligo_seg, strength=s_hi)

            f_L_ridge = spectrogram_ridge_track(ligo_seg_ridge, fs, f_ref_track=f_U_s, f_band=80.0, nperseg=NPER_RIDGE, noverlap=NOVER_RIDGE,
                                                ridge_prom_ratio_min=float(config.get("RIDGE_PROM_RATIO_MIN", 1.5)),
                                                ridge_prom_db_min=float(config.get("RIDGE_PROM_DB_MIN", 1.0)),
                                                ridge_peak2_db_min=float(config.get("RIDGE_PEAK2_DB_MIN", 1.0)),
                                                ridge_prom_ref=str(config.get("RIDGE_PROM_REF", "median")),
                                                ridge_smooth=float(config.get("RIDGE_SMOOTH", 7.0)), 
                                                max_jump_hz=float(config.get("RIDGE_MAX_JUMP_HZ", 30.0)), prefer_previous=True)

            ridge_mask = mask & np.isfinite(f_L_ridge) & np.isfinite(f_U_s)
            denom = max(int(mask.sum()), 1)
            n_r   = int(ridge_mask.sum())
            if n_r >= int(config.get("RIDGE_MIN_PTS", 12)):
                df_r = f_U_s[ridge_mask] - f_L_ridge[ridge_mask]
                f_mismatch_rms_ridge  = float(stable_rms(df_r))
                f_mismatch_mean_ridge = float(np.mean(df_r))
                ridge_conf = float(n_r) / float(denom)
            else:
                # Not enough trustworthy ridge points -> treat as low confidence and don't lean on ridge mismatch
                ridge_conf = 0.0
                if n_r > 0:
                    df_r = f_U_s[ridge_mask] - f_L_ridge[ridge_mask]
                    f_mismatch_rms_ridge  = float(stable_rms(df_r))
                    f_mismatch_mean_ridge = float(np.mean(df_r))
                else: print(f"{detector}: ridge_mask is empty or below RIDGE_MIN_PTS.")
        except Exception as e: USE_RIDGE_DIAGNOSTIC = False; print(f"{detector}: ridge diagnostic failed: {e}")
    diag["USE_RIDGE_DIAGNOSTIC"] = USE_RIDGE_DIAGNOSTIC

    # --- Hybrid mismatch: combine ridge + RMS by confidence ---
    if ridge_conf != 0.0 and f_mismatch_rms_ridge is not None:
        alpha = min(1.0, max(0.0, ridge_conf))
        f_mismatch_rms_use  = alpha * f_mismatch_rms_ridge  + (1 - alpha) * f_mismatch_rms
        f_mismatch_mean_use = alpha * f_mismatch_mean_ridge + (1 - alpha) * f_mismatch_mean
    else: f_mismatch_rms_use  = f_mismatch_rms; f_mismatch_mean_use = f_mismatch_mean

    print(f"{detector}: fit_window=({i0/fs:.4f}, {i1/fs:.4f}) mask_count={mask.sum()} RMS={f_mismatch_rms_use:.3f} ridge_conf={ridge_conf:.3f}")

    L = np.asarray(ligo_w_full[i0:i1], dtype); U = np.asarray(umh_w_full[i0:i1],  dtype)

    # Correlation ON normalized vectors (no amplitude shrink)
    Lz = L - L.mean(); Uz = U - U.mean()
    w  = tukey(len(Lz), alpha=0.2)
    Ln = Lz * w; Un = Uz * w
    num = float(np.vdot(Ln, Un).real)
    den = np.sqrt(np.vdot(Ln, Ln).real * np.vdot(Un, Un).real) + EPS_FLOOR
    match_psd_signed = float(num / den)
    match_psd_abs    = abs(match_psd_signed)

    # Keep LS gain only for residual magnitude (so reviewers can see %leftover)
    den_ls = np.vdot(U, U).real if np.vdot(U, U).real > 0 else 1.0
    k_ls   = float(np.vdot(L, U).real / den_ls)
    resid  = L - k_ls * U
    rms_L  = float(stable_rms(L)) #float(np.sqrt(np.mean(L**2)))
    rms_resid = float(stable_rms(resid)) #float(np.sqrt(np.mean(resid**2)))

    corr_window_signed = float(np.corrcoef(ligo_w_full[i0:i1], umh_w_full[i0:i1])[0, 1])
    corr_window = abs(corr_window_signed)
    # Envelope-only correlation (phase-insensitive) within the final window
    def corr_envelope(x, y):
        ex = np.abs(hilbert(x)); ey = np.abs(hilbert(y))
        ex = (ex - np.mean(ex)) / (np.std(ex) + EPS_SAFE_FLOOR)
        ey = (ey - np.mean(ey)) / (np.std(ey) + EPS_SAFE_FLOOR)
        return float(np.dot(ex, ey) / (len(ex) - 1))

    if (t_peak_abs is not None) and not (i0/fs <= t_peak_abs <= i1/fs):
        print(f"[{detector}][DIAG] WARNING: SNR peak at t={t_peak_abs:.4f}s is outside chirp window [{i0/fs:.4f}, {i1/fs:.4f}]s")
    else: print(f"[{detector}][DIAG] k_ls={k_ls:.3e}, corr_window_signed={corr_window_signed:.3f}, rms_resid/rms_L={rms_resid/rms_L:.3f}")

    # --- Global polarity convention ---
    if global_pol is None:
        if (sign_corr is not None) and np.isfinite(sign_corr): global_pol = 1.0 if (float(sign_corr) > 0.0) else -1.0
        else: global_pol = 1.0

    # --- Observed sign (comparison-window sign) ---
    sign_pol = None
    if np.isfinite(match_psd_signed) and (abs(match_psd_signed) > EPS_SAFE_FLOOR): sign_pol = 1.0 if (match_psd_signed > 0.0) else -1.0
    elif np.isfinite(rho_signed) and (abs(rho_signed) > EPS_SAFE_FLOOR): sign_pol = 1.0 if (rho_signed > 0.0) else -1.0

    # --- Geometry-only predicted sign (fallback only) ---
    sign_ant_pred = 1.0
    if (pol_psi_deg is not None) and (BINARY_IOTA_DEG is not None) and (F_plus is not None) and (F_cross is not None):
        try:
            #cos_iota = float(np.cos(np.deg2rad(float(BINARY_IOTA_DEG))))
            #h_plus   = 0.5 * (1.0 + cos_iota*cos_iota)
            #h_cross  = cos_iota
            #sgn_ant  = float(F_plus) * h_plus + float(F_cross) * h_cross
            #sign_ant_pred = 1.0 if (sgn_ant >= 0.0) else -1.0

            psi = np.deg2rad(pol_psi_deg)
            c2 = float(np.cos(2.0 * psi)); s2 = float(np.sin(2.0 * psi))
            # Rotate antenna patterns by ψ
            Fp_psi =  F_plus * c2 + F_cross * s2; Fx_psi = -F_plus * s2 + F_cross * c2
            # Use Fp_psi/Fx_psi instead of F_plus/F_cross
            cos_iota = float(np.cos(np.deg2rad(float(BINARY_IOTA_DEG))))
            h_plus   = 0.5 * (1.0 + cos_iota*cos_iota)
            h_cross  = cos_iota
            sgn_ant = Fp_psi * h_plus + Fx_psi * h_cross
            sign_ant_pred = 1.0 if (sgn_ant >= 0.0) else -1.0
        except Exception: sign_ant_pred = 1.0

    # --- Generator-provided predicted sign (preferred) ---
    if (sign_pred_gen is not None) and np.isfinite(sign_pred_gen): sign_pred_eff = 1.0 if float(sign_pred_gen) >= 0.0 else -1.0
    else: sign_pred_eff = sign_ant_pred

    # --- Match between observed sign and expected sign ---
    sign_ant_match = -1.0
    if (sign_pol is not None) and (sign_pred_eff is not None): sign_ant_match = 1.0 if (float(sign_pol) * float(sign_pred_eff) > 0.0) else -1.0

    # --- Observed vs global convention (diagnostic only) ---
    sign_vs_global = None
    if (global_pol is not None) and (sign_pol is not None) and np.isfinite(global_pol) and np.isfinite(sign_pol):
        sign_vs_global = 1.0 if (float(sign_pol) * float(global_pol) > 0.0) else -1.0

    diag["sign_pol"] = sign_pol; diag["sign_ant_pred"] = sign_ant_pred; diag["sign_ant_match"] = sign_ant_match; diag["sign_vs_global"] = sign_vs_global
    diag["k_ls"] = k_ls; diag["corr_window_signed"] = corr_window_signed; diag["corr_window"] = corr_window
    diag["corr_envelope"] = corr_envelope(ligo_w_full[i0:i1], umh_w_full[i0:i1])
    diag["match_psd_window_signed"] = match_psd_signed
    diag["match_psd_window"] = match_psd_abs
    diag["rms_ratio_resid"] = rms_resid / (rms_L + EPS_SAFE_FLOOR)
    diag["f_mismatch_rms"]  = f_mismatch_rms; diag["f_mismatch_mean"] = f_mismatch_mean
    diag["f_mismatch_mean_rel"] = f_mismatch_mean_rel = float(f_mismatch_mean / max(f_ref, 1.0))
    #Ridge-based cross-check (secondary)
    diag["f_mismatch_rms_ridge"]  = (None if f_mismatch_rms_ridge is None else float(f_mismatch_rms_ridge))
    diag["f_mismatch_mean_ridge"] = (None if f_mismatch_mean_ridge is None else float(f_mismatch_mean_ridge))
    diag["f_mismatch_rms_use"]  = float(f_mismatch_rms_use)
    diag["f_mismatch_mean_use"] = float(f_mismatch_mean_use)
    diag["ridge_conf"] = float(ridge_conf)
    diag["rms_ratio"] =  float(np.std(ligo_w_full[i0:i1] - umh_w_full[i0:i1]) / (np.std(ligo_w_full[i0:i1]) + EPS_SAFE_FLOOR))
    
    diag["fit_window_s"]= [i0/fs, i1/fs]
    diag["rho_peak_time"] = (None if (t_peak_abs is None or not np.isfinite(t_peak_abs)) else float(t_peak_abs))

    if mask.sum() > 0:
        fL_tr = f_L_s[mask]; fU_tr = f_U_s[mask]
        print(f"{detector}: f_mismatch_rms={f_mismatch_rms}, f_mismatch_mean={f_mismatch_mean}, "
              f"fL[min,max]=[{fL_tr.min():.2f},{fL_tr.max():.2f}] fU[min,max]=[{fU_tr.min():.2f},{fU_tr.max():.2f}] n={mask.sum()}")
    else: print(f"{detector}: instfreq trusted mask empty.")

    if(lc_win is not None and uc_win is not None):
        diag["amp_obs_pctl"] = AMP_PCTL = float(config.get("AMP_OBS_PCTL", 99.0))
        amp_obs_unwhite_bp = amp_proxy_unwhitened(lc_win, pctl=AMP_PCTL)
        diag["amp_obs_unwhite_bp"] = float(amp_obs_unwhite_bp)
        amp_umh_unwhite_bp = amp_proxy_unwhitened(uc_win, pctl=AMP_PCTL)
        diag["amp_umh_unwhite_bp"] = float(amp_umh_unwhite_bp)

    ###########################################
    # UMH DIAGNOSTIC SUITE (per detector)
    ###########################################
    d_ms  = {"mass_slope_Hz_per_s": None, "mass_offset_Hz": None, "mass_slope_Hz_per_Hz": None, "mass_offset_Hz_at_f0": None,
             "mass_slope_note": "n/a", "massratio_curv_diff": None, "massratio_curv_diff_raw_sign": None, 
             "massratio_curv_diff_raw": None, "massratio_curv_valid": False, "massratio_note": "n/a"}
    d_amp = {"on_i0": None, "on_i1": None, "off_i0": None, "off_i1": None, "rms_L_on_cond": None, "rms_L_off_cond": None, "rms_U_on_cond": None,
             "rms_U_off_cond": None, "rms_R_on_cond": None, "rms_R_off_cond": None, "est_sig_rms_from_LIGO": None, "amp_ratio_umh_to_estsig": None,
             "rms_L_on_w": None, "rms_U_on_w": None, "rms_L_off_w": None, "rms_R_on_w": None, "rms_R_off_w": None, "resid_whitened_inflation": None,           
             "alpha_star": None, "alpha_star_num_sh": None, "alpha_star_den_hh": None,
             "alpha_star_off": None, "alpha_star_off_num_sh": None, "alpha_star_off_den_hh": None, 
             "amplitude_note": None}
    d_ds  = {"distance_ratio_mean": None, "distance_ratio_std": None, "distance_note": "n/a"}
    d_pl  = {"pol_norm_diff": None, "pol_note": "n/a"}
    d_rd  = {"meas_delay_sec": None, "meas_delay_geom_sec": None, "pred_delay_sec": None, "lag_residual_sec": None, "lag_residual_geom_sec": None,
             "align_gate_failed": None, "umh_vs_ligo_peak_align_sec": None, "ligo_xcorr_delay_sec": None, "ligo_xcorr_strength": None, 
             "umh_xcorr_delay_sec": None, "umh_xcorr_strength": None, "resid_xcorr_delay_sec": None, "lag_note": "n/a"}

    
    # ---------------------------------------------------------
    # TOTAL MASS DIAGNOSTIC — slope of Δf(t)
    # ---------------------------------------------------------
    #f_track_L = f_L_s; f_track_U = f_U_s

    seg_L = ligo_w_full[i0:i1];   seg_U = umh_w_full[i0:i1]
    maxL  = np.max(np.abs(seg_L)); maxU = np.max(np.abs(seg_U))

    f_track_L = np.asarray(f_L_s, dtype=float); f_track_U = np.asarray(f_U_s, dtype=float)

    USE_RIDGE_MASS = bool(config.get("USE_RIDGE_MASS_DIAG", True))
    if USE_RIDGE_MASS:
        FB_RM                = float(config.get("RIDGE_MASS_F_BAND",   40.0)) #220
        NPER_RM              = int(config.get("RIDGE_MASS_NPERSEG",   256))
        NOVR_RM              = int(config.get("RIDGE_MASS_NOVERLAP",  192))
        ridge_prom_ratio_min = float(config.get("RIDGE_PROM_RATIO_MIN", 1.5))
        ridge_prom_db_min    = float(config.get("RIDGE_PROM_DB_MIN",    1.0))
        ridge_peak2_db_min   = float(config.get("RIDGE_PEAK2_DB_MIN",   1.0))
        ridge_prom_ref       = str(config.get("RIDGE_PROM_REF",    "median"))
        ridge_smooth         = float(config.get("RIDGE_SMOOTH",         7.0))
        max_jump_hz          = float(config.get("RIDGE_MAX_JUMP_HZ",   30.0))
        ridge_f_lo           = float(config.get("RIDGE_F_LO",          25.0))
        ridge_f_hi           = float(config.get("RIDGE_F_HI",         120.0))
        #ligo_seg, umh_seg
        try:
            fL_r = spectrogram_ridge_track(seg_L, fs, f_ref_track=f_L_s, f_band=FB_RM, nperseg=NPER_RM, noverlap=NOVR_RM,
                                        ridge_prom_ratio_min=ridge_prom_ratio_min, ridge_prom_db_min=ridge_prom_db_min,
                                        ridge_peak2_db_min=ridge_peak2_db_min, ridge_prom_ref=ridge_prom_ref, ridge_smooth=ridge_smooth, 
                                        max_jump_hz=max_jump_hz, f_lo=ridge_f_lo, f_hi=ridge_f_hi, prefer_previous=True)
        except Exception as e: fL_r = None; print(f"[RIDGE_MASS] L ridge exception: {e}")
        try:
            fU_r = spectrogram_ridge_track(seg_U, fs, f_ref_track=f_U_s, f_band=FB_RM, nperseg=NPER_RM, noverlap=NOVR_RM, #f_band=FB_RM, nperseg=NPER_RM, noverlap=NOVR_RM
                                        ridge_prom_ratio_min=ridge_prom_ratio_min, ridge_prom_db_min=ridge_prom_db_min,
                                        ridge_peak2_db_min=ridge_peak2_db_min, ridge_prom_ref=ridge_prom_ref, ridge_smooth=ridge_smooth, 
                                        max_jump_hz=max_jump_hz, f_lo=ridge_f_lo, f_hi=ridge_f_hi, prefer_previous=True)
        except Exception as e: fU_r = None; print(f"[RIDGE_MASS] U ridge exception: {e}")

        if (fL_r is not None) and (np.isfinite(fL_r).sum() >= 16): f_track_L = np.asarray(fL_r, dtype=float)
        if (fU_r is not None) and (np.isfinite(fU_r).sum() >= 16): f_track_U = np.asarray(fU_r, dtype=float)
    
    try:
        trim_lo, trim_hi = config.get("MASS_ISO_TRIM_LO",10), config.get("MASS_ISO_TRIM_HI",90)
        nbin      = config.get("MASS_ISO_NBIN",     16)
        minbinpts = config.get("MASS_ISO_MINBINPTS", 6)
        minbins   = config.get("MASS_ISO_MINBINS",   6)
        
        # --- Chirp-mass isolation diagnostic (amplitude-independent) ---
        t_full = np.arange(len(f_track_L)) / fs
        
        mass_fmin = float(config.get("MASS_ISO_FMIN",  35.0))
        mass_fmax = float(config.get("MASS_ISO_FMAX", 130.0)) #90.0
        
        mask_mass = np.isfinite(f_track_L) & np.isfinite(f_track_U) #& mask_L_if & mask_U_if
        mask_mass &= (f_track_L >= mass_fmin) & (f_track_L <= mass_fmax)
        mask_mass &= (f_track_U >= mass_fmin) & (f_track_U <= mass_fmax)
        # --- Monotonicity gate (critical) ---
        # Compute a smoothed derivative df/dt for each track (central diff), then require positive slope.
        def _dfdt_ok(t, f, mask, min_dfdt=0.0):
            idx = np.where(mask)[0]
            if len(idx) < 8: return mask, None
            tt = t[idx]; ff = f[idx]
            # light smoothing for derivative stability
            ff_s = ff.copy()
            if len(ff_s) >= 9: k = 9; ff_s = np.convolve(ff_s, np.ones(k)/k, mode="same")
            dfdt = np.gradient(ff_s, tt)
            thr = float(config.get("MASS_MIN_DFDT", min_dfdt))
            ok = np.isfinite(dfdt) & (dfdt > thr)
            mask2 = mask.copy(); mask2[idx] = ok
            return mask2, float(np.median(dfdt[ok])) if np.any(ok) else None

        print("[RIDGE_MASS] fs=", fs, "seg_L_len=", len(seg_L))
        print("[RIDGE_MASS] finite L:", int(np.isfinite(f_track_L).sum()), "finite U:", int(np.isfinite(f_track_U).sum()))
        if np.isfinite(f_track_L).any(): print("[RIDGE_MASS] L min/med/max:", float(np.nanmin(f_track_L)), float(np.nanmedian(f_track_L)), float(np.nanmax(f_track_L)))
        if np.isfinite(f_track_U).any(): print("[RIDGE_MASS] U min/med/max:", float(np.nanmin(f_track_U)), float(np.nanmedian(f_track_U)), float(np.nanmax(f_track_U)))
        inband_L = np.isfinite(f_track_L) & (f_track_L >= mass_fmin) & (f_track_L <= mass_fmax)
        inband_U = np.isfinite(f_track_U) & (f_track_U >= mass_fmin) & (f_track_U <= mass_fmax)
        print("[RIDGE_MASS] inband L:", int(inband_L.sum()), "inband U:", int(inband_U.sum()))
        print("[RIDGE_MASS] inband BOTH:", int((inband_L & inband_U).sum()))
        print(f"[RIDGE_MASS] idx = np.where(mask_mass)[0]:{len(np.where(mask_mass)[0])}")

        mask_mass, dfdtL_med = _dfdt_ok(t_full, f_track_L, mask_mass, min_dfdt=0.0)
        mask_mass, dfdtU_med = _dfdt_ok(t_full, f_track_U, mask_mass, min_dfdt=0.0)
        d_ms["dfdtL_med"] = dfdtL_med; d_ms["dfdtU_med"] = dfdtU_med
        print(f"[RIDGE_MASS] dfdtL_med:{dfdtL_med}, dfdtU_med:{dfdtU_med}")

        if np.any(mask_mass):
            # observer-frame chirp mass ratio (UMH vs LIGO)
            Mc_ratio_obs, r_med_obs, n_obs = chirp_mass_ratio_from_tracks(t_full, f_track_L, f_track_U, mask_mass,
                f_min_for_mass=float(config.get("MASS_DIAG_FMIN", 30.0)), f_max_for_mass=float(config.get("MASS_DIAG_FMAX", 180.0)))
            d_ms["Mc_ratio_obs"] = Mc_ratio_obs; d_ms["Mc_rmed_obs"] = r_med_obs; d_ms["Mc_n_obs"] = n_obs

            # Optional: source-frame diagnostic to deconfound distance/redshift time dilation
            if bool(config.get("MASS_DIAG_DEREDSHIFT", True)):
                UMH_z_tension = float(config.get("UMH_z_tension", np.nan))
                if np.isfinite(UMH_z_tension):
                    t_src, fL_src = deredshift_tf(t_full, f_track_L, UMH_z_tension)
                    _,     fU_src = deredshift_tf(t_full, f_track_U, UMH_z_tension)

                    Mc_ratio_src, r_med_src, n_src = chirp_mass_ratio_from_tracks(t_src, fL_src, fU_src, mask_mass,
                        f_min_for_mass=float(config.get("MASS_DIAG_FMIN", 30.0)) * (1.0 + UMH_z_tension),
                        f_max_for_mass=float(config.get("MASS_DIAG_FMAX", 180.0)) * (1.0 + UMH_z_tension))

                    d_ms["Mc_ratio_src"] = Mc_ratio_src; d_ms["Mc_rmed_src"] = r_med_src
                    d_ms["Mc_n_src"] = n_src; d_ms["Mc_z_used"] = UMH_z_tension

                    McL_tilde_src, nL_src = chirp_mass_from_track(t_src, fL_src, mask_mass,
                        f_min=float(config.get("MASS_DIAG_FMIN", 30.0)) * (1.0 + UMH_z_tension),
                        f_max=float(config.get("MASS_DIAG_FMAX_CHIRP", 90.0)) * (1.0 + UMH_z_tension),
                        trim_lo=trim_lo, trim_hi=trim_hi, nbin=nbin, minbinpts=minbinpts, minbins=minbins)
                    McU_tilde_src, nU_src = chirp_mass_from_track(t_src, fU_src, mask_mass,
                        f_min=float(config.get("MASS_DIAG_FMIN", 30.0)) * (1.0 + UMH_z_tension),
                        f_max=float(config.get("MASS_DIAG_FMAX_CHIRP", 90.0)) * (1.0 + UMH_z_tension),
                        trim_lo=trim_lo, trim_hi=trim_hi, nbin=nbin, minbinpts=minbinpts, minbins=minbins)
                    d_ms["McL_tilde_src"] = McL_tilde_src; d_ms["McL_n_src"] = nL_src
                    d_ms["McU_tilde_src"] = McU_tilde_src; d_ms["McU_n_src"] = nU_src

                    if np.isfinite(McL_tilde_src) and np.isfinite(McU_tilde_src) and (McL_tilde_src > 0):
                        d_ms["Mc_ratio_src_trackwise"] = float(McU_tilde_src / McL_tilde_src)
                    else: d_ms["Mc_ratio_src_trackwise"] = None

            # --- Per-track chirp-mass estimates (LIGO-only and UMH-only) ---
            McL_tilde_obs, nL_obs = chirp_mass_from_track(t_full, f_track_L, mask_mass,
                f_min=float(config.get("MASS_DIAG_FMIN", 30.0)), f_max=float(config.get("MASS_DIAG_FMAX_CHIRP", 90.0)),
                            trim_lo=trim_lo, trim_hi=trim_hi, nbin=nbin, minbinpts=minbinpts, minbins=minbins)
            McU_tilde_obs, nU_obs = chirp_mass_from_track(t_full, f_track_U, mask_mass,
                f_min=float(config.get("MASS_DIAG_FMIN", 30.0)), f_max=float(config.get("MASS_DIAG_FMAX_CHIRP", 90.0)),
                            trim_lo=trim_lo, trim_hi=trim_hi, nbin=nbin, minbinpts=minbinpts, minbins=minbins)
            d_ms["McL_tilde_obs"] = McL_tilde_obs; d_ms["McL_n_obs"] = nL_obs
            d_ms["McU_tilde_obs"] = McU_tilde_obs; d_ms["McU_n_obs"] = nU_obs

            if np.isfinite(McL_tilde_obs) and np.isfinite(McU_tilde_obs) and (McL_tilde_obs > 0):
                d_ms["Mc_ratio_obs_trackwise"] = float(McU_tilde_obs / McL_tilde_obs)
            else: d_ms["Mc_ratio_obs_trackwise"] = None

            # ---------------------------------------------------------
            # TOTAL MASS DIAGNOSTIC (FIXED): returns direction
            # ---------------------------------------------------------
            try:
                # Build aligned arrays in the fixed band
                t0  = np.asarray(t_vec[mask_mass], dtype=float)
                fL0 = np.asarray(f_track_L[mask_mass], dtype=float)
                fU0 = np.asarray(f_track_U[mask_mass], dtype=float)
                df0 = fL0 - fU0  # Δf = f_LIGO - f_UMH

                # Basic finite gate (keeps everything aligned)
                finite = np.isfinite(t0) & np.isfinite(fL0) & np.isfinite(fU0) & np.isfinite(df0)
                t_raw = t0[finite]
                x_raw = fL0[finite]     # frequency axis (LIGO track in fixed band)
                y_raw = df0[finite]     # df = fLIGO - fUMH

                n_raw = int(y_raw.size)
                d_ms["mass_iso_n"] = n_raw

                # Defaults (we will try to fill them no matter what)
                d_ms["mass_slope_Hz_per_s"]  = None; d_ms["mass_offset_Hz"] = None; d_ms["mass_slope_Hz_per_Hz"] = None
                d_ms["mass_offset_Hz_at_f0"] = None; d_ms["mass_iso_conf"]  = 0.0; d_ms["mass_iso_note"] = "n/a"

                MIN_PTS = int(config.get("MASS_ISO_MIN_PTS", 60))  # lowered so we still emit direction
                if n_raw < max(12, MIN_PTS): d_ms["mass_slope_note"] = f"mass iso weak (too few finite points: n={n_raw})"
                else:
                    # ---------------------------
                    # Track quality metrics (do NOT invalidate, only set confidence)
                    # ---------------------------
                    # dfL/dt monotonicity proxy
                    try:
                        # Ensure increasing t for gradient
                        o_tq = np.argsort(t_raw)
                        tq = t_raw[o_tq]; xq = x_raw[o_tq]

                        # Guard against duplicate times (shouldn't happen, but cheap)
                        dt = np.diff(tq)
                        if np.any(dt <= 0): keep_t = np.ones_like(tq, dtype=bool); keep_t[1:] = dt > 0; tq = tq[keep_t]; xq = xq[keep_t]
                        if tq.size >= 8:
                            dfLdt = np.gradient(xq, tq)
                            dfdt_min = float(config.get("MASS_ISO_DFLDT_MIN", 5.0))  # Hz/s
                            pos_frac = float(np.mean(dfLdt > dfdt_min))
                        else: pos_frac = float("nan")
                    except Exception: pos_frac = float("nan")

                    d_ms["mass_pos_dfdt_frac"] = (None if not np.isfinite(pos_frac) else float(pos_frac))

                    # Robust df span: p95 - p05 (raw span also for debugging)
                    p_lo = float(config.get("MASS_ISO_SPAN_PLO", 5.0))
                    p_hi = float(config.get("MASS_ISO_SPAN_PHI", 95.0))
                    y_lo, y_hi = np.percentile(y_raw, [p_lo, p_hi])
                    df_span_rob = float(y_hi - y_lo)
                    df_span_raw = float(np.nanmax(y_raw) - np.nanmin(y_raw))
                    d_ms["mass_df_span_raw_Hz"] = df_span_raw
                    d_ms["mass_df_span_rob_Hz"] = df_span_rob

                    # Confidence from monotonicity + robust span
                    mono_min   = float(config.get("MASS_ISO_MONO_FRAC_MIN", 0.90))
                    dfspan_ref = float(config.get("MASS_ISO_DFSPAN_REF", 12.0))  # "good" span target (Hz)

                    # monotonicity score in [0,1]
                    if np.isfinite(pos_frac): s_mono = float(np.clip((pos_frac - 0.50) / (mono_min - 0.50 + EPS_SAFE_FLOOR), 0.0, 1.0))
                    else: s_mono = 0.3  # unknown -> lowish

                    # span score in [0,1] (bigger span => lower score)
                    s_span = float(np.clip(dfspan_ref / (df_span_rob + EPS_SAFE_FLOOR), 0.0, 1.0))

                    # count score in [0,1]
                    n_ref = float(config.get("MASS_ISO_NREF", 200.0))
                    s_n = float(np.clip(n_raw / (n_ref + EPS_SAFE_FLOOR), 0.0, 1.0))

                    conf = float(np.clip(0.50 * s_span + 0.35 * s_mono + 0.15 * s_n, 0.0, 1.0))
                    d_ms["mass_iso_conf"] = conf

                    # ---------------------------
                    # df vs t slope (TIME ORDERED) — diagnostic only
                    # ---------------------------
                    try:
                        o_t = np.argsort(t_raw)
                        t_fit = t_raw[o_t]; y_t = y_raw[o_t]
                        if t_fit.size >= 12:
                            a_t, b_t = np.polyfit(t_fit, y_t, 1)
                            d_ms["mass_slope_Hz_per_s"] = float(a_t); d_ms["mass_offset_Hz"] = float(b_t)
                    except Exception: pass

                    # ---------------------------
                    # df vs f slope — BINNED + ROBUST (this is the sweep direction)
                    # ---------------------------
                    # Sort by frequency, drop plateaus, TRIM df outliers, then bin medians
                    o_f = np.argsort(x_raw)
                    x = x_raw[o_f]; y = y_raw[o_f]

                    # Drop non-increasing repeats/plateaus
                    dx = np.diff(x)
                    keep = np.ones_like(x, dtype=bool)
                    keep[1:] = dx > float(config.get("MASS_ISO_DF_MIN", 1e-3))
                    x = x[keep]; y = y[keep]

                    if x.size < 20: d_ms["mass_slope_note"] = f"mass iso weak (too few points after monotonic filter: n={int(x.size)})"
                    else:
                        # Trim catastrophic df outliers (track holes)
                        trim_lo = float(config.get("MASS_ISO_TRIM_LO", 10.0))
                        trim_hi = float(config.get("MASS_ISO_TRIM_HI", 90.0))
                        ylo, yhi = np.percentile(y, [trim_lo, trim_hi])
                        y_trim = np.clip(y, ylo, yhi)

                        NBIN = int(config.get("MASS_ISO_NBIN", 16))
                        MINBINPTS = int(config.get("MASS_ISO_MINBINPTS", 6))
                        NBIN = max(8, NBIN)

                        edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), NBIN + 1)
                        xb, yb, wb = [], [], []
                        for i in range(NBIN):
                            m = (x >= edges[i]) & (x < edges[i + 1])
                            cnt = int(np.count_nonzero(m))
                            if cnt >= MINBINPTS:
                                xb.append(0.5 * (edges[i] + edges[i + 1]))
                                yb.append(float(np.median(y_trim[m])))
                                wb.append(float(cnt))

                        xb = np.asarray(xb, dtype=float); yb = np.asarray(yb, dtype=float); wb = np.asarray(wb, dtype=float)

                        d_ms["mass_iso_bins_used"] = int(xb.size)
                        d_ms["mass_f_span_Hz"] = float(np.nanmax(x) - np.nanmin(x))
                        d_ms["mass_df_span_Hz"] = float(np.nanmax(yb) - np.nanmin(yb)) if yb.size > 0 else None

                        if xb.size < int(config.get("MASS_ISO_MINBINS", 6)):
                            d_ms["mass_slope_note"] = f"mass iso weak (too few bins: nbins={int(xb.size)})"
                        else:
                            # Weighted linear fit on binned medians
                            a_f, b_f = np.polyfit(xb, yb, 1, w=wb)
                            d_ms["mass_slope_Hz_per_Hz"] = float(a_f); d_ms["mass_offset_Hz_at_f0"] = float(b_f)

                            # Interpretation based on df vs f slope (more stable than df vs t)
                            slope_tol_f = float(config.get("UMH_MASS_SLOPE_TOL_F", 0.02))
                            if not np.isfinite(a_f): d_ms["mass_slope_note"] = "mass iso failed (non-finite slope)"
                            elif abs(a_f) < slope_tol_f: d_ms["mass_slope_note"] = f"mass OK-ish (df vs f slope small), conf={conf:.2f}"
                            elif a_f > 0: d_ms["mass_slope_note"] = f"UMH chirp too slow -> total mass high (df grows with f), conf={conf:.2f}"
                            else: d_ms["mass_slope_note"] = f"UMH chirp too fast -> total mass low (df shrinks with f), conf={conf:.2f}"
            except Exception as e:
                d_ms["mass_slope_Hz_per_s"]  = None; d_ms["mass_offset_Hz"] = None; d_ms["mass_slope_Hz_per_Hz"] = None
                d_ms["mass_offset_Hz_at_f0"] = None; d_ms["mass_iso_conf"]  = 0.0
                d_ms["mass_slope_note"] = f"Exception in Mass Slope diagnostic: {e}."

            # ---------------------------------------------------------
            # MASS RATIO DIAGNOSTIC — curvature in a FIXED LIGO-PEAK-ANCHORED window 
            # Goal: keep the curvature window tied to the OBSERVED LIGO peak (stable across sweeps).
            # ---------------------------------------------------------
            try:
                MR_WIN_SEC      = float(config.get("UMH_MR_WINDOW_SEC", 0.03))
                MR_WIN_SAMP     = int(max(16, round(MR_WIN_SEC * fs)))
                MR_MIN_PTS      = int(config.get("UMH_MR_MIN_PTS", 32))
                MR_SMOOTH_SIGMA = float(config.get("UMH_MR_SMOOTH_SIGMA", 2.0))
                # Where to start the curvature window relative to the LIGO peak (push slightly *after* peak)
                MR_START_OFFSET_SEC  = float(config.get("UMH_MR_START_OFFSET_SEC", 0.005))
                MR_START_OFFSET_SAMP = int(round(MR_START_OFFSET_SEC * fs))
                # How far around the LIGO peak we are willing to "snap" to the local IF maximum on the LIGO track
                MR_PEAK_SEARCH_HW_SEC  = float(config.get("UMH_MR_PEAK_SEARCH_HW_SEC", 0.020))
                MR_PEAK_SEARCH_HW_SAMP = int(max(8, round(MR_PEAK_SEARCH_HW_SEC * fs)))
                # Physical denom floor in Hz/s (prevents explosions when df/dt ~ 0)
                MR_DFDT_FLOOR_HZ_PER_S = float(config.get("UMH_MR_DFDT_FLOOR_HZ_PER_S", 50.0))

                # Use FULL tracks (length == len(ligo_seg)) aligned to t_vec
                fL_all = np.asarray(f_L_s, dtype=np.float64); fU_all = np.asarray(f_U_s, dtype=np.float64); N_all  = int(len(fL_all))

                # --- Build a "usable" mask for IF tracks ---
                finite = np.isfinite(fL_all) & np.isfinite(fU_all)
                band_ok = finite & (fL_all > f_min_diag) & (fL_all < f_merge_diag) & (fU_all > f_min_diag) & (fU_all < f_merge_diag)
                # If band filter is too aggressive (common when a track has holes), fall back to finite-only
                mask_ok = band_ok if int(np.count_nonzero(band_ok)) >= MR_MIN_PTS else finite
                if int(np.count_nonzero(mask_ok)) < MR_MIN_PTS:
                    d_ms["massratio_curv_diff_raw_sign"] = None; d_ms["massratio_curv_diff_raw"] = None; d_ms["massratio_curv_diff"] = None
                    d_ms["massratio_note"] = "mass ratio diagnostic skipped (too few finite IF points in full track)"
                else:
                    # --- Snap reference index to a LOCAL IF maximum on the LIGO track near the LIGO amplitude peak ---
                    a0 = max(0, j_peak_ligo - MR_PEAK_SEARCH_HW_SAMP); a1 = min(N_all, j_peak_ligo + MR_PEAK_SEARCH_HW_SAMP + 1)

                    fL_search = np.where(mask_ok[a0:a1], fL_all[a0:a1], -np.inf)
                    j_if_local = int(np.argmax(fL_search)); j_if = int(a0 + j_if_local)

                    if not np.isfinite(fL_search[j_if_local]) or (fL_search[j_if_local] == -np.inf): j_if = j_peak_ligo  # fallback: use peak directly

                    # Window is anchored AFTER that reference (post-peak curvature)
                    j0 = int(np.clip(j_if + MR_START_OFFSET_SAMP, 0, N_all - 1)); j1 = int(min(N_all, j0 + MR_WIN_SAMP))

                    # If too close to the end, slide left
                    if (j1 - j0) < MR_MIN_PTS: j1 = N_all; j0 = max(0, j1 - MR_WIN_SAMP)

                    # Require enough usable IF points INSIDE the chosen window, otherwise slide left
                    win_ok = mask_ok[j0:j1]
                    if int(np.count_nonzero(win_ok)) < MR_MIN_PTS:
                        # slide window left by up to 2*MR_WIN_SAMP to find a usable segment
                        found = False
                        for shift in (MR_WIN_SAMP, 2*MR_WIN_SAMP):
                            jj1 = max(MR_MIN_PTS, j0)  # keep sane
                            jj0 = max(0, j0 - shift)
                            if int(np.count_nonzero(mask_ok[jj0:jj1])) >= MR_MIN_PTS: j0, j1 = jj0, jj1; found = True; break
                        if not found:
                            d_ms["massratio_note"] = "mass ratio diagnostic skipped (no window with enough usable IF points)"
                            # keep all curv fields None and exit this curvature block
                            raise RuntimeError("No usable IF window for curvature")

                    # Extract window, then drop non-finite points
                    fL_m = fL_all[j0:j1].copy(); fU_m = fU_all[j0:j1].copy()
                    tt   = (np.arange(j0, j1, dtype=np.float64) / float(fs))

                    ok = np.isfinite(fL_m) & np.isfinite(fU_m)
                    fL_m = fL_m[ok]; fU_m = fU_m[ok]; tt = tt[ok]

                    if tt.size < MR_MIN_PTS:
                        d_ms["massratio_curv_diff_raw_sign"] = None; d_ms["massratio_curv_diff_raw"] = None; d_ms["massratio_curv_diff"] = None
                        d_ms["massratio_note"] = f"mass ratio diagnostic skipped (insufficient pts in fixed window: n={int(tt.size)})"
                    else:
                        # Smooth before differentiation
                        if MR_SMOOTH_SIGMA > 0.0:
                            fL_m = gaussian_filter1d(fL_m, sigma=MR_SMOOTH_SIGMA, mode="nearest")
                            fU_m = gaussian_filter1d(fU_m, sigma=MR_SMOOTH_SIGMA, mode="nearest")

                        # Derivatives
                        dfL  = np.gradient(fL_m, tt); d2fL = np.gradient(dfL, tt)
                        dfU  = np.gradient(fU_m, tt); d2fU = np.gradient(dfU, tt)

                        med_dfL = float(np.nanmedian(np.abs(dfL))) if np.any(np.isfinite(dfL)) else 0.0
                        med_dfU = float(np.nanmedian(np.abs(dfU))) if np.any(np.isfinite(dfU)) else 0.0

                        # If the LIGO IF track is locally flat/broken, curvature can't be trusted
                        if (med_dfL < MR_DFDT_FLOOR_HZ_PER_S) or (med_dfU < MR_DFDT_FLOOR_HZ_PER_S):
                            d_ms["massratio_curv_diff_raw_sign"] = None; d_ms["massratio_curv_diff_raw"] = None; d_ms["massratio_curv_diff"] = None
                            d_ms["massratio_note"] = (f"mass ratio diagnostic skipped (df/dt too small in fixed window: med_dfL={med_dfL:.2f}, med_dfU={med_dfU:.2f} Hz/s)")
                        else:
                            df_floor_L = max(MR_DFDT_FLOOR_HZ_PER_S, 0.10 * med_dfL)
                            df_floor_U = max(MR_DFDT_FLOOR_HZ_PER_S, 0.10 * med_dfU)

                            RL = d2fL / np.maximum(np.abs(dfL), df_floor_L)
                            RU = d2fU / np.maximum(np.abs(dfU), df_floor_U)

                            raw_sign = float(np.nanmean(RL - RU))
                            raw_abs  = float(np.nanmean(np.abs(RL - RU)))

                            d_ms["massratio_curv_valid"] = True; d_ms["massratio_curv_diff_raw_sign"] = raw_sign; d_ms["massratio_curv_diff_raw"] = raw_abs

                            MR_CURV_MAX = float(config.get("UMH_MR_CURV_MAX", 25.0))
                            curv_diff = float(np.clip(raw_abs, 0.0, MR_CURV_MAX)) if np.isfinite(raw_abs) else None
                            d_ms["massratio_curv_diff"] = curv_diff

                            curv_tol = float(config.get("UMH_MR_CURV_TOL", 0.5))
                            if (curv_diff is not None) and (curv_diff < curv_tol): d_ms["massratio_note"] = "mass ratio OK (curvature ratio close in fixed post-peak window)"
                            else: d_ms["massratio_note"] = "mass ratio likely off (curvature mismatch in fixed post-peak window)"

            except Exception as e: d_ms["massratio_curv_diff"] = None; d_ms["massratio_note"] = f"Exception in Mass Ratio diagnostic: {e}."
        else: print(f"[RIDGE_MASS] np.any(mask_mass) failed, no mask_mass found.")

        # PN leading-order expects: df/dt ∝ f^(11/3) - So log(df/dt) = (11/3) log(f) + const
        def pn_slope_check(t, f, mask):
            idx = np.where(mask)[0]
            if len(idx) < 12: return None, None
            tt = t[idx]; ff = f[idx]
            if len(ff) >= 9: ff = np.convolve(ff, np.ones(9)/9, mode="same")
            dfdt = np.gradient(ff, tt)
            ok = (dfdt > 0) & np.isfinite(dfdt) & (ff > 0) & np.isfinite(ff)
            if ok.sum() < 10: return None, None
            x = np.log(ff[ok]); y = np.log(dfdt[ok])
            a, b = np.polyfit(x, y, 1)
            return float(a), float(b)

        pn_a_L, pn_b_L = pn_slope_check(t_full, f_track_L, mask_mass)
        pn_a_U, pn_b_U = pn_slope_check(t_full, f_track_U, mask_mass)
        d_ms["pn_slope_L"] = pn_a_L; d_ms["pn_slope_U"] = pn_a_U; d_ms["pn_slope_target"] = 11.0/3.0

    except Exception as e: 
        d_ms["mass_slope_Hz_per_s"], d_ms["mass_offset_Hz"], d_ms["massratio_curv_diff"] = None, None, None; 
        d_ms["mass_slope_note"] = d_ms["massratio_note"] = f"Exception in Mass Slope and Mass Ratio diagnostic: {e}."

    try:
        # -----------------------------------------------------------
        # AMPLITUDE DIAGNOSTIC: noise inflation vs real deficiency (NO fitting, NO scaling)
        # -----------------------------------------------------------
        if(ligo_cond_full is not None and umh_cond_full is not None and resid_cond is not None and resid_w is not None):
            def _rms(x): x = np.asarray(x, float); return float(np.sqrt(np.mean(x*x) + EPS_FLOOR))
            def alpha_star_diag(s_w: np.ndarray, h_w: np.ndarray, mask: np.ndarray):
                #Diagnostic-only amplitude projection: alpha* = (s|h)/(h|h)
                sw = np.asarray(s_w, dtype=float); hw = np.asarray(h_w, dtype=float)
                if mask is None: m = slice(None)
                else: m = mask
                num = float(np.sum(sw[m] * hw[m])); den = float(np.sum(hw[m] * hw[m]))
                if den < EPS_FLOOR: return np.nan, num, den
                return num / den, num, den

            N_ligo = len(ligo_w_full)
            # Use the same window for scoring (SNR window)
            on_i0 = int(i0); on_i1 = int(i1); on_len = on_i1 - on_i0
            on_i0 = max(0, min(on_i0, N_ligo-2)); on_i1 = max(on_i0+1, min(on_i1, N_ligo))
            
            # Off-source window: same length, earlier, with a configurable guard gap
            noise_gap_sec = float(config.get("NOISE_GAP_SEC", 0.30)); gap = int(round(noise_gap_sec * fs)) # guard from chirp

            off_i1 = max(0, on_i0 - gap); off_i0 = max(0, off_i1 - on_len)
            if (off_i1 - off_i0) < max(16, on_len//2): off_i0 = min(N_ligo-1, on_i1 + gap); off_i1 = min(N_ligo, off_i0 + on_len)
            off_i0 = max(0, min(int(off_i0), N_ligo-2)); off_i1 = max(off_i0+1, min(int(off_i1), N_ligo))

            # Conditioned (unwhitened) RMS: signal+noise vs noise
            L_on = ligo_cond_full[on_i0:on_i1]; L_off = ligo_cond_full[off_i0:off_i1]
            U_on = umh_cond_full[on_i0:on_i1];  U_off = umh_cond_full[off_i0:off_i1]   # should be ~0-ish except leakage/windowing
            R_on = resid_cond[on_i0:on_i1];     R_off = resid_cond[off_i0:off_i1]
        
            rms_L_on = _rms(L_on); rms_U_on  = _rms(U_on); rms_L_off = _rms(L_off); rms_U_off = _rms(U_off)
            rms_R_on = _rms(R_on); rms_R_off = _rms(R_off)

            # Estimate LIGO "signal RMS" by variance subtraction (no scaling of UMH)
            est_sig_rms = float(np.sqrt(max(0.0, (rms_L_on*rms_L_on) - (rms_L_off*rms_L_off))))
            if est_sig_rms < EPS_FLOOR:
                est_sig_rms = 0.0; amp_ratio_umh_to_estsig = np.inf
                d_amp_en = ", est_sig_rms clamped to 0 (on-window RMS not above off-window); RMS-based signal estimate not informative here."
            else: amp_ratio_umh_to_estsig = rms_U_on / (est_sig_rms); d_amp_en = ""

            # Compare UMH in-band RMS to the estimated signal RMS
            amp_ratio_umh_to_estsig = rms_U_on / (est_sig_rms + EPS_FLOOR)
            # Whitened-domain residual sanity: residual on-source should look like noise
            Lw_on = ligo_w_full[on_i0:on_i1]; Uw_on = umh_w_full[on_i0:on_i1]; Lw_off = ligo_w_full[off_i0:off_i1]
            Rw_on = resid_w[on_i0:on_i1];    Rw_off = resid_w[off_i0:off_i1]
            rms_Lw_on = _rms(Lw_on); rms_Uw_on = _rms(Uw_on); rms_Lw_off = _rms(Lw_off)
            rms_Rw_on = _rms(Rw_on); rms_Rw_off = _rms(Rw_off)
            # Residual inflation factor (whitened): ~1 means residual consistent with noise
            resid_whitened_inflation = rms_Rw_on / (rms_Rw_off + EPS_FLOOR)
            # ---- alpha* diagnostic (report-only) ----
            on_mask = np.zeros(len(ligo_w_full), dtype=bool); on_mask[i0:i1] = True
            alpha_star, num_sh, den_hh = alpha_star_diag(s_w=ligo_w_full, h_w=umh_w_full, mask=on_mask)
            # Build an "off" window mask (example: a chunk well before the event)
            NOISE_GAP_SEC = float(config.get("NOISE_GAP_SEC", 0.35)); gap = int(round(NOISE_GAP_SEC * fs))
            win_len = int(i1 - i0); off_i1 = int(i0 - gap); off_i0 = int(off_i1 - win_len)
            if off_i0 < 0 or (off_i1 - off_i0) < win_len: off_i0 = int(i1 + gap); off_i1 = int(off_i0 + win_len)
            off_i0 = max(0, min(off_i0, N_ligo - 2)); off_i1 = max(off_i0 + 1, min(off_i1, N_ligo))
            off_mask = np.zeros(len(ligo_w_full), dtype=bool); off_mask[off_i0:off_i1] = True

            alpha_star_off, num_off, den_off = alpha_star_diag(s_w=ligo_w_full, h_w=umh_w_full, mask=off_mask)

            d_amp["on_i0"] = i0; d_amp["on_i1"] = i1; d_amp["off_i0"] = off_i0; d_amp["off_i1"] = off_i1;
            d_amp["rms_L_on_cond"]  = rms_L_on; d_amp["rms_L_off_cond"] = rms_L_off; d_amp["rms_U_on_cond"] = rms_U_on
            d_amp["rms_U_off_cond"] = rms_U_off; d_amp["rms_R_on_cond"] = rms_R_on; d_amp["rms_R_off_cond"] = rms_R_off
            d_amp["est_sig_rms_from_LIGO"] = est_sig_rms; d_amp["amp_ratio_umh_to_estsig"] = amp_ratio_umh_to_estsig
            d_amp["rms_L_on_w"] = rms_Lw_on; d_amp["rms_U_on_w"]  = rms_Uw_on; d_amp["rms_L_off_w"] = rms_Lw_off
            d_amp["rms_R_on_w"] = rms_Rw_on; d_amp["rms_R_off_w"] = rms_Rw_off; d_amp["resid_whitened_inflation"] = resid_whitened_inflation

            d_amp["alpha_star"] = float(alpha_star) if np.isfinite(alpha_star) else None
            d_amp["alpha_star_num_sh"] = float(num_sh); d_amp["alpha_star_den_hh"] = float(den_hh)
            d_amp["alpha_star_off"] = float(alpha_star_off) if np.isfinite(alpha_star_off) else None
            d_amp["alpha_star_off_num_sh"] = float(num_off); d_amp["alpha_star_off_den_hh"] = float(den_off)

            d_amp["amplitude_note"] = f"AMP_DIAG cond: rms_L(on)={rms_L_on:.3e} rms_L(off)={rms_L_off:.3e} " \
                    f"est_sig_rms={est_sig_rms:.3e} rms_U(on)={rms_U_on:.3e} ratio(U/estSig)={amp_ratio_umh_to_estsig:.3f}" \
                    f"AMP_DIAG whitened: rms_R(on)={rms_Rw_on:.3f} rms_R(off)={rms_Rw_off:.3f} inflation={resid_whitened_inflation:.3f}{d_amp_en}"
            # Print a compact, interpretable summary
            print(f"[{detector}] AMP_DIAG cond: rms_L(on)={rms_L_on:.3e} rms_L(off)={rms_L_off:.3e} "
                    f"est_sig_rms={est_sig_rms:.3e} rms_U(on)={rms_U_on:.3e} ratio(U/estSig)={amp_ratio_umh_to_estsig:.3f}")
            print(f"[{detector}] AMP_DIAG whitened: rms_R(on)={rms_Rw_on:.3f} rms_R(off)={rms_Rw_off:.3f} "
                    f"inflation={resid_whitened_inflation:.3f}")
    except Exception as e: d_amp["amplitude_note"] = "Exception in Amplitude diagnostic: {e}."; print(f"[WARN] Failed to store amp_noise_diag: {e}")

    # ---------------------------------------------------------
    # DISTANCE DIAGNOSTIC — amplitude ratio stability
    # ---------------------------------------------------------
    try:
        if ligo_w_full is not None and umh_w_full is not None:
            if maxL > 0 and maxU > 0:
                A_Lw = np.abs(seg_L); A_Uw = np.abs(seg_U)

                Nseg = seg_L.size
                j0 = int(0.1 * Nseg); j1 = int(0.9 * Nseg)
                if j1 > j0:
                    A_ratio = A_Uw[j0:j1] / np.maximum(A_Lw[j0:j1], 1e-12)
                    if A_ratio.size > 0:
                        d_ds["distance_ratio_mean"] = float(np.mean(A_ratio))
                        d_ds["distance_ratio_std"]  = float(np.std(A_ratio))

                        dist_tol = float(config.get("UMH_DIST_RATIO_TOL", 0.05))
                        if d_ds["distance_ratio_std"] < dist_tol: d_ds["distance_note"] = "Distance scaling dominant (amplitude ratio stable)"
                        else: d_ds["distance_note"] = "Amplitude profile mismatch (beyond pure distance)"
    except Exception as e: d_ds["distance_ratio_mean"], d_ds["distance_ratio_std"] = None, None; d_ds["distance_note"] = "Exception in Distance diagnostic: {e}."

    # ---------------------------------------------------------
    # POLARIZATION / INCLINATION DIAGNOSTIC
    # ---------------------------------------------------------
    try:
        if ligo_w_full is not None and umh_w_full is not None:
            if maxL > 0 and maxU > 0:
                Hn = seg_L / maxL; Un = seg_U / maxU
                d_pl["pol_norm_diff"] = float(np.mean(np.abs(Hn - Un)))

                pol_tol = float(config.get("UMH_POL_TOL", 0.15))    #Set to a 15% off tolerance.
                if d_pl["pol_norm_diff"] < pol_tol: d_pl["pol_note"] = f"Projection OK: (within {(pol_tol*100)}%) for this detector (RA/DEC + PSI/IOTA roughly consistent)"
                else: d_pl["pol_note"] = f"Projection mismatch: (not within {(pol_tol*100)}%) for this detector (one or more of RA/DEC/PSI/IOTA or intrinsic waveform)"
    except Exception as e: d_pl["pol_norm_diff"] = None; d_pl["pol_note"] = "Exception in Polarization diagnostic: {e}."

    # ---------------------------------------------------------
    # SKY POSITION (RA/DEC) DIAGNOSTIC — irreducible lag
    # ---------------------------------------------------------
    try:
        if (t_peak_geom_abs is None) or (not np.isfinite(t_peak_geom_abs)) or (t_peak_abs is None) or (anchor_t_peak_abs is None):
            d_rd["meas_delay_sec"] = None; d_rd["pred_delay_sec"] = None; d_rd["lag_residual_sec"] = None;
            d_rd["meas_delay_geom_sec"] = None; d_rd["lag_residual_geom_sec"] = None; 
            #d_rd["lag_note"] = "Missing t_peak_geom_abs, t_peak_abs or anchor_t_peak_abs"
        else:
            d_rd["meas_delay_sec"] = meas_delay = float(t_peak_abs) - float(anchor_t_peak_abs)
            d_rd["meas_delay_geom_sec"] = meas_delay_geom = float(t_peak_geom_abs) - float(anchor_t_peak_abs)

            # Predicted delay from geometry (relative to the same anchor)
            if (geom_delay_sec_eff is None): d_rd["pred_delay_sec"] = pred_delay = None
            else: d_rd["pred_delay_sec"] = pred_delay = float(geom_delay_sec_eff)
            if pred_delay is None: 
                d_rd["lag_residual_sec"] = None; d_rd["align_gate_failed"] = None; d_rd["umh_vs_ligo_peak_align_sec"] = None; 
                #d_rd["lag_note"] = "Missing geom_delay_sec_eff"
            else:
                d_rd["lag_residual_sec"] = meas_delay - pred_delay; d_rd["lag_residual_geom_sec"] = meas_delay_geom - pred_delay
                #lag_tol_sec = float(config.get("UMH_LAG_TOL_SEC", 2e-4))  # ~0.2 ms
                #if abs(d_rd["lag_residual_sec"]) < lag_tol_sec: d_rd["lag_note"] = "Inter-site timing consistent with geometry (RA/DEC OK)"
                #else: d_rd["lag_note"] = "Inter-site timing inconsistent with geometry (RA/DEC / peak-pick issue)"
                
        # lag_meas_sec is the measured UMH↔LIGO peak lag for THIS detector (post-alignment).
        if lag_meas_sec is None: lag_meas_sec = float(lag_samp) / float(fs); d_rd["align_gate_failed"] = False
        else: d_rd["align_gate_failed"] = (abs(float(lag_meas_sec)) > float(config.get("UMH_ALIGN_LAG_GATE_SEC", 2e-3)))
        d_rd["umh_vs_ligo_peak_align_sec"] = float(lag_meas_sec)

        if ligo_xcorr_delay_sec is not None: d_rd["ligo_xcorr_delay_sec"] = ligo_xcorr_delay_sec
        if ligo_xcorr_strength  is not None: d_rd["ligo_xcorr_strength"]  = ligo_xcorr_strength
        if umh_xcorr_delay_sec is not None: d_rd["umh_xcorr_delay_sec"]   = umh_xcorr_delay_sec
        if umh_xcorr_strength  is not None: d_rd["umh_xcorr_strength"]    = umh_xcorr_strength
        if ligo_xcorr_delay_sec is not None and umh_xcorr_delay_sec is not None: 
            d_rd["resid_xcorr_delay_sec"] = resid_xcorr_delay_sec = ligo_xcorr_delay_sec - umh_xcorr_delay_sec
            lag_tol_sec = float(config.get("UMH_LAG_TOL_SEC", 2e-4))  # ~0.2 ms
            if abs(resid_xcorr_delay_sec) < lag_tol_sec: d_rd["lag_note"] = "Inter-site timing consistent with geometry (RA/DEC OK)"
            else: d_rd["lag_note"] = "Inter-site timing inconsistent with geometry (RA/DEC / peak-pick issue)"
        else: d_rd["lag_note"] = None #"Missing ligo_xcorr_delay_sec or umh_xcorr_delay_sec"

        d_rd["t_env_p2_ligo_vs_anchor"] = dt_env2_ligo - anchor_dt_env2_ligo; d_rd["t_env_p3_ligo_vs_anchor"] = dt_env3_ligo - anchor_dt_env3_ligo
        d_rd["t_env_p2_umh_vs_anchor"]  = dt_env2_umh  - anchor_dt_env2_umh;  d_rd["t_env_p3_umh_vs_anchor"]  = dt_env3_umh  - anchor_dt_env3_umh
        d_rd["t_env_p2_resid"] = d_rd["t_env_p2_ligo_vs_anchor"] - d_rd["t_env_p2_umh_vs_anchor"]
        d_rd["t_env_p3_resid"] = d_rd["t_env_p3_ligo_vs_anchor"] - d_rd["t_env_p3_umh_vs_anchor"]
    except Exception as e: 
        d_rd["meas_delay_sec"] = None; d_rd["pred_delay_sec"] = None; d_rd["lag_residual_sec"] = None; 
        d_rd["align_gate_failed"] = None; d_rd["umh_vs_ligo_peak_align_sec"] = None; 
        d_rd["lag_note"] = "Exception in Sky Position: {e}."

    diag["Mass_Slope_Ratio_Diagnostics"] = d_ms
    diag["Amplitude_Diagnostics"] = d_amp
    diag["Distance_Diagnostics"] = d_ds
    diag["Polarization_Inclination_Diagnostics"] = d_pl
    diag["Sky_Position_Diagnostics"] = d_rd

    return diag, resid_cond, resid_w, dt_env2_ligo, dt_env3_ligo, dt_env2_umh, dt_env3_umh


def spectrogram_ridge_track(sig, fs, f_ref_track=None, f_band=80.0, nperseg=256, noverlap=192,
                            ridge_prom_ratio_min=1.5, ridge_prom_db_min=1.0, ridge_peak2_db_min=1.0,
                            ridge_prom_ref="median", ridge_smooth=7, max_jump_hz=30.0, f_lo=None, f_hi=None, prefer_previous=True):
    """
    Estimate a robust time–frequency ridge for a short segment.
    Returns an array of length len(sig) with NaNs where undefined.
    If f_ref_track is provided (same length as sig), the search at each time is restricted to [f_ref(t) - f_band, f_ref(t) + f_band].
    """
    f, t, Sxx = spectrogram(sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density", mode="psd")
    ridge_f = np.full_like(t, np.nan, dtype=float)
    use_median = (str(ridge_prom_ref).lower().strip() == "median")
    prev_f = np.nan

    for k, tau in enumerate(t):
        # ----- band restriction (optional) -----
        if f_ref_track is not None:
            j = int(round(tau * fs))
            f0 = f_ref_track[j] if (0 <= j < len(f_ref_track)) else np.nan
            band_mask = ((f >= f0 - f_band) & (f <= f0 + f_band)) if np.isfinite(f0) else np.ones_like(f, dtype=bool)
            if f_lo is not None: band_mask &= (f >= float(f_lo))
            if f_hi is not None: band_mask &= (f <= float(f_hi))
        else: band_mask = np.ones_like(f, dtype=bool)
        if not np.any(band_mask): continue

        col = Sxx[band_mask, k]
        if col.size < 3 or not np.any(np.isfinite(col)): continue
        col = np.asarray(col, float); col = np.where(np.isfinite(col), col, np.nan)

        # reference floor: median/mean of column (robust)
        ref = float(np.nanmedian(col) if use_median else np.nanmean(col)); ref = max(ref, EPS_FLOOR)

        # Candidate peaks: we’ll take the top few and apply continuity preference (this avoids jumping to a different stripe when two are close)
        finite = np.isfinite(col)
        if not np.any(finite): continue
        idxs = np.where(finite)[0]
        # sort by power descending
        order = idxs[np.argsort(col[idxs])[::-1]]
        K = min(5, len(order)); order = order[:K]
        chosen = None; chosen_peak = None

        for ii in order:
            peak = float(col[ii])

            prom_ratio = peak / ref
            prom_db = 10.0 * np.log10(prom_ratio + EPS_FLOOR)
            if (prom_ratio < float(ridge_prom_ratio_min)) or (prom_db < float(ridge_prom_db_min)): continue

            # second-best check (peak2)
            peak2_db_min = float(ridge_peak2_db_min)
            if peak2_db_min > 0.0:
                col2 = col.copy(); col2[ii] = np.nan
                peak2 = float(np.nanmax(col2)) if np.any(np.isfinite(col2)) else 0.0
                peak2_db = 10.0 * np.log10((peak + EPS_FLOOR) / (peak2 + EPS_FLOOR))
                if peak2_db < peak2_db_min: continue

            f_candidate = float(f[band_mask][ii])

            # continuity gate
            if np.isfinite(prev_f) and (max_jump_hz is not None) and (max_jump_hz > 0):
                if abs(f_candidate - prev_f) > float(max_jump_hz): continue

            # If we reach here, candidate is acceptable.
            if not prefer_previous or not np.isfinite(prev_f): chosen = f_candidate; chosen_peak = peak; break

            # prefer candidate closest to prev_f among acceptable ones
            if chosen is None: chosen = f_candidate; chosen_peak = peak
            elif abs(f_candidate - prev_f) < abs(chosen - prev_f): chosen = f_candidate; chosen_peak = peak

        if chosen is None: continue

        ridge_f[k] = chosen; prev_f = chosen

    # Optional smoothing on the spectrogram-time-grid ridge (ignoring NaNs)
    if ridge_smooth is not None and int(ridge_smooth) >= 3:
        w = int(ridge_smooth)
        if w % 2 == 0: w += 1
        rf = ridge_f.copy(); mask = np.isfinite(rf)
        if mask.sum() >= w:
            rf_fill = rf.copy(); idx = np.where(mask)[0]
            rf_fill[:] = np.interp(np.arange(len(rf)), idx, rf[idx])
            kernel = np.ones(w, dtype=float) / float(w)
            rf_s = np.convolve(rf_fill, kernel, mode="same")
            ridge_f = rf_s

    # interpolate ridge back to full time grid
    t_full = np.arange(len(sig)) / fs
    out = np.full_like(t_full, np.nan, dtype=float)
    valid = np.isfinite(ridge_f)
    if valid.sum() >= 2: out[:] = np.interp(t_full, t[valid], ridge_f[valid])

    # after interpolation / smoothing in spectrogram_ridge_track, before return:
    if f_lo is not None: out = np.where(np.isfinite(out), np.maximum(out, float(f_lo)), out)
    if f_hi is not None: out = np.where(np.isfinite(out), np.minimum(out, float(f_hi)), out)

    return out



def build_spectrogram(y_win, fs_base, fs_spec, i0, NPER, NOVER, p_lo, p_hi, dtype=np.float64):
    """
    Build a log-power spectrogram (optionally upsampled for visualization).
    Returns (f, t_abs_core, S_db_core, vmin, vmax).
    """
    y = np.asarray(y_win, dtype=dtype)

    # --- Optional upsampling for prettier spectrograms ---
    if fs_spec > fs_base:
        up_factor = fs_spec / fs_base
        N_orig = len(y)
        t_orig = np.arange(N_orig) / fs_base

        N_up = int(round(N_orig * up_factor))
        t_up = np.arange(N_up) / fs_spec

        # Linear interpolation is fine for visualization only
        y = np.interp(t_up, t_orig, y)
        fs_use = fs_spec
    else: fs_use = fs_base

    # --- Compute spectrogram ---
    #f, t, S = spectrogram(y, fs=fs_use, window="hann", nperseg=NPER, noverlap=NOVER, detrend=False, scaling="density", mode="psd")
    #S_db = 10.0 * np.log10(np.maximum(S, EPS_FLOOR))

    # --- Compute spectrogram ---
    # Visual-safety: avoid SciPy overflow if y has absurd magnitude.
    # We rescale y ONLY for the spectrogram computation, then add back the exact dB offset so the plotted S_db matches the unscaled result (when finite).
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    scale = 1.0
    ymax = float(np.max(np.abs(y))) if y.size else 0.0
    if np.isfinite(ymax) and ymax > 0.0 and ymax > 1e6: scale = ymax; y = y / scale
    f, t, S = spectrogram(y, fs=fs_use, window="hann", nperseg=NPER, noverlap=NOVER, detrend=False, scaling="density", mode="psd")
    S_db = 10.0 * np.log10(np.maximum(S, EPS_FLOOR))
    # Undo the amplitude rescale in dB so the result matches original units:
    # y_scaled = y/scale => PSD_scaled = PSD/(scale^2) => dB_scaled = dB - 20log10(scale)
    if scale != 1.0: S_db += 20.0 * np.log10(scale)

    # Smooth a bit in (freq, time)
    S_db = gaussian_filter(S_db, sigma=(0.3, 0.15))

    # Local window times → absolute detector timeline
    t_abs = t + i0 / fs_base

    # --- Core region for color scaling AND plotting ---
    if S_db.shape[1] > 6: S_db_core = S_db[:, 3:-3]; t_abs_core = t_abs[3:-3]
    else: S_db_core = S_db; t_abs_core = t_abs
    #S_db_core = S_db; t_abs_core = t_abs

    # Compute vmin/vmax from core region
    vmin, vmax = np.nanpercentile(S_db_core, [p_lo, p_hi])
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        max_range = 95.0
        if (vmax - vmin) > max_range: vmin = vmax - max_range
        vmin = max(vmin, -140.0)
        vmax = min(vmax, -20.0)
    else: vmin, vmax = -120.0, -40.0  # robust fallback

    print(f"Spectrogram: vmin={vmin}, vmax={vmax}")

    return f, t_abs_core, S_db_core, vmin, vmax


def run_ligo_compiler_test(config_overrides=None):
    """
    Main driver enforcing a clean, UMH-faithful apples-to-apples comparison.

    Steps per detector:
      Load LIGO strain and UMH template.
      Resample UMH to LIGO fs (if needed).
      Apply identical bandpass+notches.
      Estimate PSD from LIGO; whiten both with same PSD.
      Coarse alignment (lag + polarity) in whitened domain.
      Fine time-stretch search around s≈1 to maximize correlation.
      Compute matched-filter SNR.
      Generate overlays, residuals, spectrograms, and ASD/FFT diagnostics.
    """
    config        = get_default_config()
    if config_overrides: config.update(config_overrides)

    #Pull ligo data retrieved through the Planck download data Script, to the base shared PlanckData directory.
    ligo_data     = config["LIGO_DATA"]

    PHYSICS_STRICT      = config.get("PHYSICS_STRICT",       True)

    ENABLE_FINE_STRETCH = config.get("ENABLE_FINE_STRETCH", False)
    if(PHYSICS_STRICT): ENABLE_FINE_STRETCH = False

    USE_RIDGE_DIAGNOSTIC = bool(config.get("USE_RIDGE_DIAGNOSTIC", True))

    notch_lines   = tuple(config.get("NOTCH_LINES", [60,120,180,240,331,500]))

    NPER          = 4096 #1024        #4096
    NOVER         = 2048 #768         #1024

    # --- analysis constants ---
    NPER_PSD      = 4096        # for ASD / whitening
    NOVER_PSD     = 2048

    # Alignment quality thresholds
    MIN_PEAK_CORR    = 0.40   # min corr to accept fine peak alignment
    MIN_STRETCH_CORR = 0.15   # min corr to accept non-unity stretch

    dpi       = config["DPI"]
    dtype     = config["DTYPE"]

    indir     = config["INPUT_FOLDER"]
    outdir    = config["OUTPUT_FOLDER"]

    title     = "UMH Ligo Compiler"
    file_root = "UMH_vs_LIGO"
    file_hdr  = "UMH_Ligo_Compiler"

    file_in   = "UMH_Chirp_Generator"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    file_path_in=os.path.join(indir, file_in)

    print(f"{title}: Files Will be Saved to {outdir}.")

    umh_npz_path = f"{file_path_in}_Dynamic.npz"

    # --- Load UMH chirp (single source of truth) ---
    umh = np.load(f"{file_path_in}_Dynamic.npz")

    dt_umh = float(umh["dt_obs"])
    fs_umh = float(umh["fs_obs"])
    Fn_umh = float(umh["Fn_obs"])

    config["UMH_z_tension"] = UMH_z_tension = umh.get("UMH_z_tension", np.nan)

    print(f"[UMH] UMH_z_tension={umh.get('UMH_z_tension', '')}, z_GR={umh.get('z_GR', '')}")
    print(f"[UMH] d_geom_Mpc={umh.get('distance_Mpc')}, M1={umh.get('M1_kg')}, M2={umh.get('M2_kg')}")

    det_names = [str(s) for s in umh["detector_names"]]
    det_names_norm  = [n.strip().lower() for n in det_names]
    name_to_idx     = {name: i for i, name in enumerate(det_names_norm)}

    f_min   = float(umh.get("f_min_obs",    30.0))
    f_merge = float(umh.get("f_merge_obs", 150.0))
    lowcut  = float(umh.get("lowcut_obs",  f_min))
    highcut = float(umh.get("highcut_obs", config.get("highcut", 500.0)))
    f_rd    = float(umh.get("f_rd_obs",    250.0))
    f_ref   = float(umh.get("f_ref_obs",   100.0))
    t_merge_obs = umh.get("t_merge_obs",   None)

    umh_time = umh.get("time", None)
    umh_freq_track_obs_Hz = umh.get("freq_track_obs_Hz", None)
    t_track_obs, f_track_obs = None, None
    if(umh_time is not None): t_track_obs = np.asarray(umh_time, dtype=dtype)              # intrinsic UMH time
    if(umh_freq_track_obs_Hz is not None): f_track_obs = np.asarray(umh_freq_track_obs_Hz, dtype=dtype)     # intrinsic GW track
    
    M1_solar_src = float(umh.get("M1_solar_src", 0.0))
    M2_solar_src = float(umh.get("M2_solar_src", 0.0))
    distance_Mpc = float(umh.get("distance_Mpc", 0.0))
    ra_deg          = float(umh.get("ra_deg", 0.0))
    dec_deg         = float(umh.get("dec_deg", 0.0))
    pol_psi_deg     = float(umh.get("pol_psi_deg", 0.0))
    BINARY_IOTA_DEG = float(umh.get("BINARY_IOTA_DEG", 0.0))

    geom_delay = umh.get("geom_delay", [])
    det_Fp     = umh.get("det_Fp", [])
    det_Fx     = umh.get("det_Fx", [])
    det_Sign   = umh.get("det_Sign", [])

    band_lo = max(20.0, 0.8 * f_min)
    band_hi = min(0.9 * Fn_umh, highcut)
    config["band_lo"] = band_lo; config["band_hi"] = band_hi

    print(f"[UMH] fs={fs_umh:.1f} Hz, f_min={f_min:.1f}, highcut={highcut:.1f}, f_rd={f_rd:.1f}, band=[{band_lo:.1f},{band_hi:.1f}]")

    global_pol = None; network_polarity_flip_applied = False

    # Load LIGO Wave forms and do Pre-Alignment check to find Global Anchor.
    det_fnd = []; det_results = {}
    summary = {"GLOBAL": None, "NETWORK": None}; psd_map_dict = {}
    for detector, ligo_path in ligo_data.items():
        if detector not in det_names: print(f"[warn] No UMH channel for {detector}; skipping."); continue

        print()
        # We make a pass per detector to find the strongest Detector Signal and then in the next phase Globally Anchor around the strongest detector.
        print(f"================ Starting: Alignment Check: {detector} ================")

        # --- Load LIGO strain ---
        try:
            with h5py.File(os.path.join(config["LIGO_OUTPUT_ROOT_FOLDER"], ligo_path), "r") as f:
                ligo_strain = np.array(f["strain"]["Strain"], dtype=dtype)
                dt_ligo = f["strain"]["Strain"].attrs["Xspacing"]
                fs_ligo = 1.0 / dt_ligo
                N_ligo = ligo_strain.size
        except Exception as e: print(f"Skipping {detector} due to LIGO file issue: {e}"); continue

        # --- Load UMH template for this detector & resample to LIGO fs ---
        def UMH_Resample(config, detector, fs_ligo, fs_umh, ligo_strain, umh_raw):
            print(f"LIGO fs (original): {fs_ligo:.3f} Hz | N={len(ligo_strain)} samples")
            print(f"UMH  fs (original): {fs_umh:.3f} Hz | N={len(umh_raw)} samples")
            # If needed, resample UMH to LIGO fs
            if abs(fs_ligo - fs_umh) > 1e-9:
                r = fs_ligo / fs_umh
                frac = Fraction(r).limit_denominator(4096)
                umh_resamp = resample_poly(umh_raw, frac.numerator, frac.denominator)
                fs = fs_ligo
                print("Resampling UMH → LIGO fs:")
                print(f"  Ratio: {r:.6f} ≈ {frac.numerator}/{frac.denominator}")
                print(f"  UMH fs (after): {fs:.3f} Hz | N={len(umh_resamp)} samples")
            else:
                fs = fs_ligo
                umh_resamp = umh_raw.copy()
                print("Sampling rates already match — no resampling performed.")

            N_ligo = len(ligo_strain)
            N_umh  = len(umh_resamp)
            print(f"LIGO N={N_ligo}, UMH N={N_umh} (no padding yet)")
            print(f"Final shared fs: {fs:.3f} Hz | Final UMH length: {len(umh_resamp)} samples\n")

            return N_ligo, N_umh, umh_resamp, fs

        umh_key = f"strain_{detector}"
        umh_raw = np.array(umh[umh_key], dtype=dtype)
        N_ligo, N_umh, umh_resamp, fs = UMH_Resample(config, detector, fs_ligo, fs_umh, ligo_strain, umh_raw)

        # ------------------------------
        # Conditioning + PSD + whitening (common PSD)
        # ------------------------------
        def Condition_Wave(config, detector, fs, strain_full, band_lo, band_hi, notch_lines, f_psd=None, Pxx=None, dtype=dtype, debug="LIGO"):
            cond_full = condition_time_domain(strain_full, fs, band_lo, band_hi, notch_lines, dtype=dtype)
            cond_full = sanitize(cond_full, name=f"{detector}:{debug}:cond_full")
            N_cf_len  = len(cond_full)

            # Event time estimate within a plausible window (GW150914 in 32s LOSC is ~16.4 s)
            # --- Event-time estimate (robust two-stage) ---
            t_event_sec, idx_event = estimate_event_time_seconds_hilbert(cond_full, fs, t_bounds=(15.0, 22.0), debug_tag=detector, dtype=dtype)
            # fallback to full 10–22 window if nothing above threshold
            if t_event_sec < 15.0 or t_event_sec > 22.0: 
                t_event_sec, idx_event = estimate_event_time_seconds_hilbert(cond_full, fs, t_bounds=(10.0, 22.0), debug_tag=detector, dtype=dtype)
            print(f"PreCheck Alignment: [{detector}] estimated t_event ≈ {t_event_sec:.6f} s (from {debug} Hilbert envelope)")
            # End: Event-time estimate.

            # Estimate PSD Whitening based on off-source data LIGO only
            if(f_psd is None and Pxx is None):
                def _slice_indices(t_start, t_end, fs, N):
                    i0 = int(round(t_start * fs))
                    i1 = int(round(t_end   * fs))
                    i0 = max(0, min(N, i0))
                    i1 = max(0, min(N, i1))
                    if i1 <= i0: return None
                    return i0, i1
                
                mask = np.ones(N_cf_len, dtype=bool)
                # Drop initial filter transient region (helps avoid startup artifacts)     
                psd_mode = str(config.get("PSD_MODE", "window")).lower()           
                drop_sec = float(config.get("PSD_DROP_FILTER_TRANSIENT_SEC", 1.0))
                i_drop = int(round(drop_sec * fs))

                if psd_mode == "window":
                    guard     = float(config.get("PSD_GUARD_SEC", 2.0))
                    pre_start = float(config.get("PSD_PRE_START_SEC", 12.0))
                    pre_end   = float(config.get("PSD_PRE_END_SEC",   4.0))

                    # Pre-event window: [t_event - pre_start, t_event - pre_end]
                    t0_pre = t_event_sec - pre_start
                    t1_pre = t_event_sec - pre_end

                    sl_pre = _slice_indices(t0_pre, t1_pre, fs, N_cf_len)
                    chunks = []

                    if sl_pre is not None:
                        i0, i1 = sl_pre
                        i0 = max(i0, i_drop)
                        if i1 > i0: chunks.append(cond_full[i0:i1])

                    # Optional post-event window: [t_event + post_start, t_event + post_end]
                    if bool(config.get("PSD_USE_POST_WINDOW", False)):
                        post_start = float(config.get("PSD_POST_START_SEC", 4.0))
                        post_end   = float(config.get("PSD_POST_END_SEC",   12.0))
                        t0_post = t_event_sec + post_start
                        t1_post = t_event_sec + post_end

                        sl_post = _slice_indices(t0_post, t1_post, fs, N_cf_len)
                        if sl_post is not None:
                            i0, i1 = sl_post
                            i0 = max(i0, i_drop)
                            if i1 > i0: chunks.append(cond_full[i0:i1])

                    if len(chunks) == 0:
                        # Fallback: original mask approach if windows don't fit in the file Exclude guard band around event
                        i0g = max(0, int(round((t_event_sec - guard) * fs)))
                        i1g = min(N_cf_len, int(round((t_event_sec + guard) * fs)))
                        mask[i0g:i1g] = False
                        # Exclude initial transient
                        mask[:i_drop] = False
                        l_for_psd = cond_full[mask]
                        print(f"[{detector}] PSD_MODE=window (fallback->mask), PSD samples={len(l_for_psd)}")
                    else:
                        l_for_psd = np.concatenate(chunks)
                        print(f"[{detector}] PSD_MODE=window, PSD samples={len(l_for_psd)} "
                              f"(pre={'yes' if sl_pre else 'no'}, post={'yes' if bool(config.get('PSD_USE_POST_WINDOW', False)) else 'no'})")
                else:
                    # psd_mode == "mask" -> keep original behavior (but add a bigger guard if desired)
                    guard = float(config.get("PSD_GUARD_SEC", 2.0))
                    i0g = max(0, int(round((t_event_sec - guard) * fs)))
                    i1g = min(N_cf_len, int(round((t_event_sec + guard) * fs)))
                    mask[i0g:i1g] = False
                    mask[:i_drop] = False
                    l_for_psd = cond_full[mask]
                    print(f"[{detector}] PSD_MODE=mask, PSD samples={len(l_for_psd)}")

                # Generate PSD for use in Whitening.
                f_psd, Pxx = estimate_psd_from_ligo(l_for_psd, fs, nperseg=NPER_PSD, noverlap=NOVER_PSD, dtype=dtype)
            # End: Estimate PSD Whitening based on off-source data LIGO only
            
            # ---------------------------------
            # Optional: Very light glitch gating
            # ---------------------------------
            gate_glitches = bool(config.get("GATE_GLITCHES", False))
            if gate_glitches:
                # Hilbert envelope of the conditioned strain_full
                env_full = np.abs(hilbert(cond_full))
                med_env  = np.median(env_full)

                # Define a high threshold so we only touch real spikes e.g. 8× the median envelope
                k_thresh = float(config.get("GATE_ENV_FACTOR", 8.0))
                thresh   = k_thresh * med_env

                # Protect a window around the GW event so we never gate the chirp itself
                protect_before = float(config.get("GATE_PROTECT_BEFORE_SEC", 0.25))
                protect_after  = float(config.get("GATE_PROTECT_AFTER_SEC", 0.35))
                t = np.arange(len(cond_full)) / fs
                protect = (t >= (t_event_sec - protect_before)) & \
                          (t <= (t_event_sec + protect_after))

                # Candidate glitch points: envelope above threshold AND outside the chirp window
                glitch_pts = (env_full > thresh) & (~protect)

                # Build a smooth gate (Tukey-ish) around each glitch point
                gate = np.ones_like(cond_full, dtype=float)
                half_width = int(float(config.get("GATE_HALF_WIDTH_SEC", 0.03)) * fs)  # ~30 ms by default

                idxs = np.where(glitch_pts)[0]
                for idx in idxs:
                    i0 = max(0, idx - half_width)
                    i1 = min(len(gate), idx + half_width)
                    n  = i1 - i0
                    if n <= 3: continue
                    # simple cosine taper from 1 → 0 → 1
                    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
                    gate[i0:i1] = np.minimum(gate[i0:i1], 1.0 - 0.9 * window)

                # Apply gate to the conditioned strain_full
                cond_full *= gate
            # End: Optional: Very light glitch gating


            # Whiten the entire conditioned LIGO time series with that PSD
            w_full = whiten_with_psd(cond_full, fs, f_psd, Pxx, dtype=dtype)
            w_full = sanitize(w_full, name=f"{detector}:{debug}:w_full")

            return w_full, cond_full, f_psd, Pxx, t_event_sec


        ligo_w_full, ligo_cond_full, f_psd, Pxx, t_event_sec = Condition_Wave(config, detector, fs, ligo_strain, 
                                   band_lo, band_hi, notch_lines, f_psd=None, Pxx=None, dtype=dtype, debug="LIGO")
        
        # Store PSD arrays for later use by the generator
        psd_map = {"freqs": f_psd.astype(dtype), "psd": Pxx.astype(dtype)}

        # Condition UMH (resampled to fs) and whiten it with the SAME PSD
        umh_w_short, _, _, _, _ = Condition_Wave(config, detector, fs, umh_resamp, band_lo, band_hi, notch_lines, 
                                                 f_psd=f_psd, Pxx=Pxx, dtype=dtype, debug="UMH")

        # Coarse align (operates on whitened data)
        corr, start_idx = coarse_align_template(ligo_w_full, umh_w_short, fs)
        print(f"PreCheck Alignment: [{detector}] coarse_align_template: corr={corr:.3f}, start_idx={start_idx} samp ({start_idx/fs:.4f} s)")

        # --- Predict UMH merge index on the LIGO timeline using generator-known t_merge_obs ---
        idx_merge_loc = None
        if t_merge_obs is not None:
            idx_merge_loc = int(round(float(start_idx) + float(t_merge_obs) * float(fs)))
            idx_merge_loc = int(np.clip(idx_merge_loc, 0, len(ligo_w_full) - 1))
            print(f"[{detector}] t_merge_obs anchor: t_merge_obs={float(t_merge_obs):.6f}s -> idx_merge_loc={idx_merge_loc} ({idx_merge_loc/fs:.6f}s)")
        else: print(f"[{detector}] t_merge_obs is None; falling back to legacy UMH peak picking.")

        # Construct a full-length UMH vector on the LIGO timeline (optional; handy for plots)
        umh_full = np.zeros_like(ligo_strain, dtype=dtype)
        stop_idx = min(start_idx + N_umh, N_ligo)
        length   = stop_idx - start_idx
        if length > 0: umh_full[start_idx:stop_idx] = umh_resamp[:length]

        # Whitened, timeline-aligned UMH.
        umh_w_full, umh_cond_full, _, _, _ = Condition_Wave(config, detector, fs, umh_full, band_lo, band_hi, notch_lines, 
                                                            f_psd=f_psd, Pxx=Pxx, dtype=dtype, debug="UMH")

        # ------------------------------
        # Envelope-based fine alignment near true peaks
        # ------------------------------
        def Fine_Align(config, detector, fs, ligo_w_full, umh_w_full, umh_cond_full, idx_merge_loc=None):
            N = len(ligo_w_full)
            
            #Find Peak, using coarse peak, exclude areas around peak to stop_idx inadvertantly picking another lobe.
            edge_exclude_crs_sec = float(config.get("FIT_EDGE_EXCLUDE_CRS_SEC", 0.5))
            edge_exclude_crs     = int(edge_exclude_crs_sec * fs)
            #Find Peak, secondary using fine peak detection.
            edge_exclude_sec     = float(config.get("FIT_EDGE_EXCLUDE_SEC", 0.05))
            edge_exclude         = int(edge_exclude_sec * fs)           
            search_half_sec      = float(config.get("FIT_SEARCH_HALF_SEC", 0.2))
            search_half          = int(search_half_sec * fs)

            env_ligo = np.abs(hilbert(ligo_w_full))

            # ------------------------------
            # LIGO peak: data-only, but expected-time locked (preferred)
            # ------------------------------
            if idx_merge_loc is not None:
                idx_expected = int(np.clip(idx_merge_loc, 0, N - 1))
                idx_ligo_peak, idx_ligo_peak_sub = find_peak_loudest_significant(fs, env_ligo, idx_expected=idx_expected, half_width=search_half,
                    edge_exclude=edge_exclude, smooth_ms=float(config.get("PEAK_SMOOTH_MS", 12.0)), k_mad=float(config.get("PEAK_K_MAD", 6.0)),
                    max_offset_sec=float(config.get("PEAK_MAX_OFFSET_SEC", 0.08)), tie_radius_sec=float(config.get("PEAK_TIE_RADIUS_SEC", 0.01)))
                idx_ligo_peak_crs = idx_expected; idx_ligo_peak = int(idx_ligo_peak); idx_ligo_peak_sub = float(idx_ligo_peak_sub)
                print(f"[{detector}] LIGO peak locked to t_merge_obs: idx_expected={idx_expected}, idx_ligo_peak={idx_ligo_peak}, idx_ligo_peak_sub={idx_ligo_peak_sub}")
            else:
                # Fallback to legacy behavior if no t_merge_obs provided
                idx_ligo_peak_crs = primary_peak(env_ligo, N, edge_exclude_crs)
                idx_ligo_peak, idx_ligo_peak_sub = find_peak_loudest_significant(fs, env_ligo, idx_expected=idx_ligo_peak_crs, half_width=search_half,
                    edge_exclude=edge_exclude, smooth_ms=float(config.get("PEAK_SMOOTH_MS", 12.0)), k_mad=float(config.get("PEAK_K_MAD", 6.0)),
                    max_offset_sec=float(config.get("PEAK_MAX_OFFSET_SEC", 0.08)), tie_radius_sec=float(config.get("PEAK_TIE_RADIUS_SEC", 0.01)))
                idx_ligo_peak = int(idx_ligo_peak); idx_ligo_peak_sub = float(idx_ligo_peak_sub)
                print(f"[{detector}] Legacy LIGO peak: idx_ligo_peak_crs={idx_ligo_peak_crs}, idx_ligo_peak={idx_ligo_peak}, idx_ligo_peak_sub={idx_ligo_peak_sub}")

            # ------------------------------
            # UMH peak: DO NOT PICK. Anchor deterministically to t_merge_obs.
            # (Optional tiny local refine could be allowed, but not needed.)
            # ------------------------------
            if idx_merge_loc is not None:
                idx_umh_peak = int(np.clip(idx_merge_loc, 0, N - 1))
                print(f"[{detector}] UMH peak ANCHORED to t_merge_obs: idx_umh_peak={idx_umh_peak}")
            else:
                # Fallback legacy UMH peak picking
                env_umh  = np.abs(hilbert(umh_w_full))
                j0 = max(0, idx_ligo_peak - search_half); j1 = min(N, idx_ligo_peak + search_half)
                idx_umh_peak = j0 + int(np.argmax(env_umh[j0:j1]))
                idx_merge_loc = idx_umh_peak
                print(f"[{detector}] Legacy UMH peak picked: idx_umh_peak={idx_umh_peak}")

            peak_lag         = idx_ligo_peak - idx_umh_peak
            tau              = peak_lag / fs
            umh_w_full_aa    = fractional_delay_fft(umh_w_full, fs, tau)
            umh_cond_full_aa = fractional_delay_fft(umh_cond_full, fs, tau)

            idx_merge_loc = int(round(idx_merge_loc + tau * fs))
            idx_merge_loc = int(np.clip(idx_merge_loc, 0, len(umh_w_full_aa) - 1))

            #Sub Align peak to find exact alignment from coarse peak.
            idx_center, idx_center_sub = Sub_Align(N, umh_w_full_aa, idx_ligo_peak, search_half)

            # Apply residual sub-sample shift so UMH peak sits exactly on LIGO peak
            residual_lag_samples = idx_center_sub - idx_ligo_peak
            tau_resid            = -residual_lag_samples / fs   # note the minus sign: shift UMH toward LIGO

            #lag_meas_sec     = (idx_center_sub - idx_ligo_peak) / fs
            lag_meas_sec_pre  = (idx_center_sub - idx_ligo_peak) / fs
            #t_peak_ligo_abs  = float(idx_ligo_peak) / float(fs)
            t_peak_ligo_abs   = float(idx_ligo_peak_sub) / float(fs)

            if abs(tau_resid) > 1e-9:  # avoid pointless FFT work
                umh_w_full_aa    = fractional_delay_fft(umh_w_full_aa,    fs, tau_resid)
                umh_cond_full_aa = fractional_delay_fft(umh_cond_full_aa, fs, tau_resid)
                idx_center       = idx_ligo_peak  # by construction, we've just aligned peaks
                            
                idx_merge_loc    = int(round(idx_merge_loc + tau_resid * fs))
                idx_merge_loc    = int(np.clip(idx_merge_loc, 0, len(umh_w_full_aa) - 1))

                # recompute sub-align on the corrected UMH
                _, idx_center_sub2 = Sub_Align(N, umh_w_full_aa, idx_ligo_peak, search_half)
                lag_meas_sec  = (idx_center_sub2 - idx_ligo_peak) / fs
            else: idx_center  = int(round(idx_center_sub)); lag_meas_sec = lag_meas_sec_pre; idx_center_sub2 = 0
            
            mf_gate_samp = float(config.get("MF_ALIGN_GATE_SAMP", 0.25))  # quarter-sample default
            dsec_int, dsec_sub, _ = meas_delay_xcorr_sec(fs, ligo_w_full, umh_w_full_aa, idx_center, halfwin_sec=0.15, maxlag_sec=0.01)
            print(f"[{detector}] Fine_Align: lag_meas_sec:{lag_meas_sec} dsec_int:{dsec_int} dsec_sub={dsec_sub}")
            if abs(dsec_sub) * fs >= mf_gate_samp:
                umh_w_full_aa    = fractional_delay_fft(umh_w_full_aa,    fs, -dsec_sub)
                umh_cond_full_aa = fractional_delay_fft(umh_cond_full_aa, fs, -dsec_sub)
                idx_merge_loc    = int(np.clip(int(round(idx_merge_loc + (-dsec_sub * fs))), 0, N-1))
                idx_center       = int(np.clip(int(round(idx_center    + (-dsec_sub * fs))), 0, N-1))
                idx_center_sub2  = np.clip(idx_center_sub2 + (-dsec_sub * fs), 0, N-1)
                dsec_int2, dsec_sub2, _ = meas_delay_xcorr_sec(fs, ligo_w_full, umh_w_full_aa, idx_center, halfwin_sec=0.15, maxlag_sec=0.01)
                print(f"[{detector}] Fine_Align: meas_delay_xcorr_sec: dsec_int2:{dsec_int2} dsec_sub2={dsec_sub2}")

            # Define fit window around idx_center (now effectively aligned with LIGO peak)
            fit_before_sec    = float(config.get("FIT_WIN_BEFORE_SEC", 0.18))
            fit_after_sec     = float(config.get("FIT_WIN_AFTER_SEC", 0.22))
            win_before        = int(fit_before_sec * fs)
            win_after         = int(fit_after_sec  * fs)

            i0 = max(0, idx_center - win_before)
            i1 = min(N, idx_center + win_after)
            if (i1 - i0) < int(0.15 * fs):
                half = int(0.175 * fs)
                i0   = max(0, idx_center - half)
                i1   = min(N, idx_center + half)

            return umh_w_full_aa, umh_cond_full_aa, i0, i1, idx_ligo_peak, idx_center, \
                    idx_merge_loc, lag_meas_sec, tau, tau_resid, t_peak_ligo_abs, dsec_int, dsec_sub

        idx_merge_loc_eff = idx_merge_loc if config.get("USE_MERGE_LOC_FOR_PEAK", False) else None
        umh_w_full_aa, umh_cond_full_aa, i0, i1, idx_ligo_peak, idx_center, idx_merge_loc, lag_meas_sec, tau, tau_resid, \
            t_peak_ligo_abs, dsec_int, dsec_sub = Fine_Align(config, detector, fs, ligo_w_full, umh_w_full, umh_cond_full, 
                                                             idx_merge_loc=idx_merge_loc_eff)
        t_peak_obs = float(idx_ligo_peak) / float(fs)

        # --- Fine time-stretch (optional, gated) - ONLY USED FOR DIAGNOSTIC PURPOSES ---
        stretch_accepted=False
        if ENABLE_FINE_STRETCH and PHYSICS_STRICT is False: # Disabled always under PHYSICS_STRICT Mode.
            
            def Fine_Stretch(config, detector, fs, ligo_w_full, umh_w_full_aa, umh_cond_full_aa, i0, i1, idx_center, dtype=np.float64):
                S_MIN, S_MAX, N_STEPS = 0.98, 1.02, 21
                IMPROVE_EPS, ABS_MIN  = 0.02, 0.15

                lw_win = ligo_w_full[i0:i1].astype(dtype)
                uw_win =  umh_w_full_aa[i0:i1].astype(dtype)

                s_best, uw_warp_win, corr_best = best_stretch_by_corr(lw_win, uw_win, fs, s_min=S_MIN, s_max=S_MAX, n_steps=N_STEPS, dtype=dtype)
                # Baseline (s=1) correlation for gating
                corr_unity = best_stretch_by_corr(lw_win, uw_win, fs, s_min=1.0, s_max=1.0, n_steps=1, dtype=dtype)[2]

                if (np.isfinite(s_best)
                    and abs(s_best - 1.0) > 1e-6
                    and (corr_best - corr_unity) > IMPROVE_EPS
                    and corr_best > ABS_MIN):

                    # Stretch ABOUT the current MF-peak anchor
                    i_peak     = int(idx_center)   # envelope-locked center from step 4
                    umh_w = time_stretch_about_anchor(umh_w_full_aa, s_best, i_peak, dtype=dtype)
                    umh_cond = time_stretch_about_anchor(umh_cond_full_aa, s_best, i_peak, dtype=dtype)
                    stretch_accepted = True

                    print(f"PreCheck Alignment: [{detector}] fine-stretch ACCEPTED: s_best={s_best:.5f}, corr_search={corr_best:.3f}, baseline={corr_unity:.3f}")

                else: print(f"PreCheck Alignment: [{detector}] fine-stretch REJECTED: s_best={s_best:.5f}, corr_search={corr_best:.3f}, baseline={corr_unity:.3f}")

                return umh_w, umh_cond, stretch_accepted, corr_best, corr_unity

            umh_w_full_aa, umh_cond_full_aa, stretch_accepted, corr_best, corr_unity = Fine_Stretch(
                config, detector, fs, ligo_w_full, umh_w_full_aa, umh_cond_full_aa, i0, i1, idx_center, dtype=dtype)
        # --- End Fine time-stretch (optional, gated) - ONLY USED FOR DIAGNOSTIC PURPOSES ---


        lw_win  = ligo_w_full[i0:i1].astype(dtype)
        uw_win  = umh_w_full_aa[i0:i1].astype(dtype)

        num_L = np.vdot(lw_win, lw_win).real
        num_U = np.vdot(uw_win, uw_win).real
        
        #k_phys = np.sqrt(num_L / num_U)
        if not (np.isfinite(num_L) and np.isfinite(num_U)) or num_U <= EPS_SAFE_FLOOR: k_phys = float("nan")
        else: k_phys = float(np.sqrt(num_L / num_U))
        corr = np.vdot(lw_win, uw_win).real
        if abs(corr) > EPS_SAFE_FLOOR: sign_corr = 1.0 if corr > 0 else -1.0
        else: sign_corr = 1.0

        print(f"PreCheck Alignment: [{detector}] k_phys(LS gain, physical, just printed, not scaled.) = {k_phys:.3e}")

        detector_polarity_flip_applied = False  # default: no flip applied
        if (PHYSICS_STRICT is False) and config.get("ALLOW_PER_DETECTOR_POLARITY", False): 
            if sign_corr is not None and float(sign_corr) < 0.0:
                umh_w_full_aa = umh_w_full_aa * (-1.0)
                umh_cond_full_aa = umh_cond_full_aa * (-1.0)
                uw_win = uw_win * (-1.0)
                detector_polarity_flip_applied = True # flip was actually applied
                print(f"PreCheck Alignment: [{detector}] sign_corr = {sign_corr:+.0f} (applied={detector_polarity_flip_applied})")
        else: print(f"PreCheck Alignment: [{detector}] sign_corr = {sign_corr:+.0f} (diagnostic, not applied.)")

        rho_signed, rho_abs, t_peak, match_lagged, lag_samp, i0_snr, i1_snr =  matched_filter_snr_window(config, 
                                        detector, fs, N_ligo, ligo_w_full, umh_w_full_aa, idx_center, tau, i0, i1, dtype=dtype)

        # Single, final recenter pass around MF peak (clamped to envelope center)
        t0_win    = i0_snr / fs
        t1_win    = i1_snr / fs
        t_center  = idx_center / fs
        max_drift = 0.25
        if abs(t_peak - t_center) > max_drift: t_peak = t_center

        recentered_final = False
        if not (t0_win <= t_peak <= t1_win):
            width = (i1_snr - i0_snr) / fs
            ctr   = int(np.clip(t_peak * fs, 0, len(ligo_w_full) - 1))
            half  = int(0.5 * width * fs)
            i0_snr    = max(0, ctr - half)
            i1_snr    = min(len(ligo_w_full), ctr + half)
            recentered_final = True
            print(f"PreCheck Alignment: [{detector}] recentered diagnostics to [{i0_snr/fs:.4f},{i1_snr/fs:.4f}] s (final)")

        print(f"PreCheck Alignment: [{detector}] windowed rho_peak_signed={rho_signed:.3f}, |rho|={rho_abs:.3f} at t={t_peak:.6f}s")

        # Per-detector overlap in PSD metric, aligned at the MF peak lag
        L = np.asarray(ligo_w_full[i0_snr:i1_snr], dtype)
        U = np.asarray(umh_w_full_aa[i0_snr:i1_snr], dtype)

        if lag_samp > 0: U_al = np.r_[np.zeros(lag_samp), U[:-lag_samp]]
        elif lag_samp < 0: k  = -lag_samp; U_al = np.r_[U[k:], np.zeros(k)]
        else: U_al = U

        num = float(np.vdot(L, U_al).real)
        den = float(np.sqrt(np.vdot(L, L).real * np.vdot(U_al, U_al).real) + EPS_FLOOR)
        match_psd_signed = num / den

        # Take the aligned window
        lw_win = ligo_w_full[i0_snr:i1_snr].astype(float)
        uw_win = umh_w_full_aa[i0_snr:i1_snr].astype(float)

        dtr_idx   = name_to_idx[detector.strip().lower()]
        geom_delay_sec_raw = float(umh["geom_delay"][dtr_idx])
        F_plus    = float(umh["det_Fp"][dtr_idx])
        F_cross   = float(umh["det_Fx"][dtr_idx])
        sign_pred_gen = float(umh["det_Sign"][dtr_idx])
        
        print(f"PreCheck Alignment: geom_delay_sec_raw={geom_delay_sec_raw}")

        # Unwhitened (but bandpassed/conditioned) windows for distance-invariant amplitude ratios
        lc_win = ligo_cond_full[i0_snr:i1_snr].astype(float)
        uc_win = umh_cond_full[i0_snr:i1_snr].astype(float)
 
        # Do not use geom_delay_sec in Chirp Diagnostics for geom_delay_sec_eff yet until after we determine Anchor.
        diag, resid_cond, resid_w, dt_env2_ligo, dt_env3_ligo, dt_env2_umh, dt_env3_umh = chirp_diagnostics(config, detector, fs, f_min, f_merge, f_ref, 
                                lw_win, uw_win, ligo_w_full, umh_w_full_aa, k_phys, idx_merge_loc,
                                rho_signed, lag_samp, i0_snr, i1_snr, geom_delay_sec_raw=geom_delay_sec_raw, geom_delay_sec_eff=0.0, t_peak_abs=t_peak_ligo_abs, 
                                t_event_sec=t_event_sec, lag_meas_sec=lag_meas_sec, anchor_t_peak_abs=t_peak_ligo_abs, 
                                BINARY_IOTA_DEG=BINARY_IOTA_DEG, pol_psi_deg=pol_psi_deg, F_plus=F_plus, F_cross=F_cross, sign_pred_gen=sign_pred_gen, global_pol=None, 
                                sign_corr=sign_corr, detector_polarity_flip_applied=detector_polarity_flip_applied, lc_win=lc_win, uc_win=uc_win, 
                                ligo_cond_full = ligo_cond_full, umh_cond_full = umh_cond_full, dtype=dtype)

        # --- Summary record ---
        if(PHYSICS_STRICT is False): # Disabled always under PHYSICS_STRICT Mode.
            if(bool(config.get("ENABLE_FINE_STRETCH", True))):
                diag["fine_stretch"] = {"enabled": True,
                    "accepted": bool(stretch_accepted),
                    "s_best": float(s_best if np.isfinite(s_best) else 1.0),
                    "corr_unity": float(corr_unity),
                    "corr_best": float(corr_best),
                    "corr_gain": float(corr_best - corr_unity),
                    "recentered": bool(recentered_final)},
            else: diag["fine_stretch"] = {"enabled": False}
        
        psd_map_dict[detector] = psd_map

        det_results[detector] = {}
        det_results[detector]["fs"] = fs
        det_results[detector]["f_min"]   = f_min
        det_results[detector]["f_merge"] = f_merge
        det_results[detector]["f_ref"]   = f_ref

        det_results[detector]["N_ligo"] =            N_ligo
        det_results[detector]["idx_ligo_peak"]     = idx_ligo_peak
        det_results[detector]["ligo_w_full"]       = ligo_w_full
        det_results[detector]["ligo_cond_full"]    = ligo_cond_full
        det_results[detector]["lw_win"] =            lw_win

        det_results[detector]["umh_w_short"] =       umh_w_short        #Original UMH Wave for Detector
        det_results[detector]["umh_w_full"] =        umh_w_full         #Full UMH Wave for Detector applied to same timeframe.
        det_results[detector]["umh_cond_full"] =     umh_cond_full      #Full UMH Conditioned Wave for Detector applied to same timeframe.
        det_results[detector]["umh_w_full_aa"] =     umh_w_full_aa      #Full UMH Wave for Detector aligned.
        det_results[detector]["umh_cond_full_aa"] =  umh_cond_full_aa   #Full UMH Conditioned Wave for Detector aligned.
        det_results[detector]["uw_win"]  =           uw_win

        det_results[detector]["resid_cond"] =        resid_cond
        det_results[detector]["resid_w"] =           resid_w

        det_results[detector]["start_idx"] =         start_idx
        det_results[detector]["i0"] =                i0
        det_results[detector]["i1"] =                i1
        det_results[detector]["i0_snr"] =            i0_snr
        det_results[detector]["i1_snr"] =            i1_snr
        det_results[detector]["idx_center"] =        idx_center
        det_results[detector]["idx_merge_loc"] =     idx_merge_loc
        
        det_results[detector]["dt_env2_ligo"] =      dt_env2_ligo
        det_results[detector]["dt_env3_ligo"] =      dt_env3_ligo
        det_results[detector]["dt_env2_umh"] =       dt_env2_umh
        det_results[detector]["dt_env3_umh"] =       dt_env3_umh

        det_results[detector]["detector_polarity_flip_applied"] = detector_polarity_flip_applied

        det_results[detector]["k_phys"] =     k_phys
        det_results[detector]["rho_signed"] = rho_signed
        det_results[detector]["sign_corr"]  = sign_corr

        det_results[detector]["lag_samples_peak"] = lag_samp
        det_results[detector]["t_event_est_diag"] = t_event_sec
        det_results[detector]["t_peak"] =           t_peak
        det_results[detector]["lag_meas_sec"] =     lag_meas_sec
        det_results[detector]["t_peak_obs"] =       t_peak_obs
        det_results[detector]["t_peak_ligo_abs"] =  t_peak_ligo_abs
        
        det_results[detector]["tau"]       =  tau
        det_results[detector]["tau_resid"] =  tau_resid
        det_results[detector]["geom_delay_sec_raw"] = geom_delay_sec_raw
        det_results[detector]["F_plus"]    =   F_plus
        det_results[detector]["F_cross"]   =   F_cross
        det_results[detector]["sign_pred_gen"] = sign_pred_gen

        det_results[detector]["diag"] = diag    # Add Detector Diagnostics to Results.
        det_fnd.append(detector)                # Add Detector to array, to calculate anchor.

        print(f"================ Completed: Alignment Check: {detector} ================")
    #End: Load LIGO Wave forms and do Pre-Alignment check to find Global Anchor.

    # Found Global Anchor to use for all Alignment.
    def snr_for(detector): D = det_results[detector]; return abs(D["rho_signed"])
    anchor_detector = max(det_fnd, key=snr_for)
    print(); print(f"================ Alignment Anchored Globally using: {anchor_detector} ================")
    anchor = det_results[anchor_detector]
    anchor_geom_delay_sec_raw = anchor["geom_delay_sec_raw"]
    anchor_t_peak =          anchor["t_peak"]
    anchor_t_peak_obs =      anchor["t_peak_obs"]
    anchor_t_peak_ligo_abs = anchor["t_peak_ligo_abs"]
    anchor_start_idx =       anchor["start_idx"]
    anchor_i0 =              anchor["i0"]
    anchor_i1 =              anchor["i1"]
    anchor_i0_snr =          anchor["i0_snr"]
    anchor_i1_snr =          anchor["i1_snr"]
    anchor_lag_samp   =      anchor["lag_samples_peak"]
    anchor_idx_center =      anchor["idx_center"]
    anchor_idx_merge_loc =   anchor["idx_merge_loc"]

    anchor_dt_env2_ligo =    anchor["dt_env2_ligo"]
    anchor_dt_env3_ligo =    anchor["dt_env3_ligo"]
    anchor_dt_env2_umh =     anchor["dt_env2_umh"]
    anchor_dt_env3_umh =     anchor["dt_env3_umh"]

    anchor_ligo_w_full =     anchor["ligo_w_full"]
    
    anc_rho_sgn =            anchor["rho_signed"]
    anc_rho_sgn_corr =       anchor["sign_corr"]

    anchor_umh_w_full = anchor["umh_w_full"] = anchor["umh_w_full_aa"]       #Solidify Alignment for anchor for umh_w_full
    anchor["umh_cond_full"] = anchor["umh_cond_full_aa"]    #Solidify Alignment for anchor for umh_cond_full

    #Peform Check on Global Polarity Flip.
    pol_anchor = 1.0 if anc_rho_sgn > 0.0 else -1.0
    def global_polarity_audit(pol, det_fnd, det_results):
        rho_signed_pol, rho_abs = [], []
        for det in det_fnd:
            rho_sign = float(det_results[det].get("rho_signed", 0.0))  # ideally rho_signed_peak
            rho_signed_pol.append(pol * rho_sign)   # changes with flip
            rho_abs.append(abs(rho_sign))           # reporting only
        score_signed_sum = float(np.sum(rho_signed_pol))               # audit score
        rho_net_abs = float(np.sqrt(np.sum(np.square(rho_abs))))       # informational only
        return score_signed_sum, rho_net_abs, rho_abs
    
    # Override only if the network signed-sum is meaningfully better
    s_keep, rho_net_keep, _ = global_polarity_audit(pol_anchor, det_fnd, det_results)
    s_flip, rho_net_flip, _ = global_polarity_audit(-pol_anchor, det_fnd, det_results)
    rel_margin = 0.10; abs_margin = 0.25           # 10% relative improvement, absolute improvement in signed-sum
    delta = (s_flip - s_keep); denom = max(1e-6, abs(s_keep))
    global_pol = (-pol_anchor) if ((delta / denom) > rel_margin and delta > abs_margin) else pol_anchor
    
    #ReAnchor geom_delay_sec from chosen anchor.
    anchor_geom_delay_sec_eff = anchor["geom_delay_sec_eff"] = anchor["diag"]["geom_delay_sec_eff"] #Will be 0.0 from prealign.
    for detector in det_fnd: det_results[detector]["geom_delay_sec_eff"] = det_results[detector]["geom_delay_sec_raw"] - anchor_geom_delay_sec_raw

    print(f"Global Anchor: Index Center={anchor_idx_center / fs}[s]")
    print(f"================ Completed: Alignment Anchored Globally using: {anchor_detector} ================")
    # End: Found Global Anchor to use for all Alignment.

    # ------------------------------------------------------------
    # Check / Apply NETWORK polarity convention ONCE, globally (all dets)
    # ------------------------------------------------------------
    network_polarity_flip_applied = False
    if config.get("ALLOW_GLOBAL_POLARITY_FLIP", True) and (global_pol is not None) and (float(global_pol) < 0.0) and config.get("ALLOW_PER_DETECTOR_POLARITY", False) is False:
        print("[UMH] Applying network-wide polarity flip (global convention)")
        network_polarity_flip_applied = True; global_pol_eff = 1.0

        for detector in det_fnd:
            dr = det_results[detector]
            dr["umh_w_short"] *= -1.0; dr["umh_w_full"] *= -1.0; dr["umh_cond_full"] *= -1.0; 
            dr["sign_corr"] *= -1.0; dr["rho_signed"] *= -1.0 #dr["uw_win"] *= -1.0; 

        # Apply sign to Anchor.
        fs = anchor["fs"]; N_ligo = anchor["N_ligo"]; k_phys = anchor["k_phys"]
        ligo_cond_full = anchor["ligo_cond_full"]; umh_cond_full = anchor["umh_cond_full"]; 
        anchor_umh_w_full = anchor["umh_w_full"];
        lw_win = anchor["lw_win"]; #uw_win = anchor["uw_win"] 
        t_event_sec = anchor["t_event_est_diag"];
        F_plus = anchor["F_plus"]; F_cross = anchor["F_cross"]; sign_pred_gen = anchor["sign_pred_gen"];
        detector_polarity_flip_applied = anchor["detector_polarity_flip_applied"];
        lag_meas_sec = anchor["lag_meas_sec"]; idx_ligo_peak = anchor["idx_ligo_peak"]
        anc_rho_sgn_corr_eff = anchor["sign_corr"]
                
        anc_rho_sgn, rho_abs, anchor_t_peak, match_lagged, anchor_lag_samp, anchor_i0_snr, anchor_i1_snr =  matched_filter_snr_window(config, 
                                            anchor_detector, fs, N_ligo, anchor_ligo_w_full, anchor_umh_w_full, anchor_idx_center, tau, 
                                            anchor_i0, anchor_i1, dtype=dtype)
        anchor["rho_signed"] = anc_rho_sgn; anchor["t_peak"] = anchor_t_peak; anchor["lag_samples_peak"] = anchor_lag_samp;
        anchor["i0_snr"] = anchor_i0_snr; anchor["i1_snr"] = anchor_i1_snr

        anchor["lw_win"] = lw_win = anchor_ligo_w_full[anchor_i0_snr:anchor_i1_snr].astype(dtype)
        anchor["uw_win"] = uw_win = anchor_umh_w_full[anchor_i0_snr:anchor_i1_snr].astype(dtype)

        num_L = np.vdot(lw_win, lw_win).real; num_U = np.vdot(uw_win, uw_win).real
        if not (np.isfinite(num_L) and np.isfinite(num_U)) or num_U <= EPS_SAFE_FLOOR: anchor["k_phys"] = k_phys = float("nan")
        else: anchor["k_phys"] = k_phys = float(np.sqrt(num_L / num_U))
        corr = np.vdot(lw_win, uw_win).real
        if abs(corr) > EPS_SAFE_FLOOR: anchor["sign_corr"] = sign_corr = 1.0 if corr > 0 else -1.0
        else: anchor["sign_corr"] = sign_corr = 1.0
        anc_rho_sgn_corr_eff = sign_corr 

        search_half_sec  = float(config.get("FIT_SEARCH_HALF_SEC", 0.2))
        search_half = int(search_half_sec * fs)
        anchor_idx_center, idx_center_sub = Sub_Align(N_ligo, anchor_umh_w_full, idx_ligo_peak, search_half)
        anchor["lag_meas_sec"] = lag_meas_sec = (idx_center_sub - idx_ligo_peak) / fs
        anchor["idx_center"] = anchor_idx_center; 

        # Unwhitened (but bandpassed/conditioned) windows for distance-invariant amplitude ratios
        lc_win = ligo_cond_full[i0_snr:i1_snr].astype(float)
        uc_win = umh_cond_full[i0_snr:i1_snr].astype(float)

        diag, resid_cond, resid_w, dt_env2_ligo, dt_env3_ligo, dt_env2_umh, dt_env3_umh = chirp_diagnostics(config, anchor_detector, fs, f_min, f_merge, f_ref, 
                    lw_win, uw_win, anchor_ligo_w_full, anchor_umh_w_full, k_phys, anchor_idx_merge_loc,
                    anc_rho_sgn, anchor_lag_samp, anchor_i0_snr, anchor_i1_snr, geom_delay_sec_raw=anchor_geom_delay_sec_raw, geom_delay_sec_eff=anchor_geom_delay_sec_eff, 
                    t_peak_abs=anchor_t_peak_ligo_abs, t_event_sec=t_event_sec, lag_meas_sec=lag_meas_sec, anchor_t_peak_abs=anchor_t_peak_ligo_abs, 
                    BINARY_IOTA_DEG=BINARY_IOTA_DEG, pol_psi_deg=pol_psi_deg, F_plus=F_plus, F_cross=F_cross, sign_pred_gen=sign_pred_gen, global_pol=global_pol_eff, 
                    sign_corr=anc_rho_sgn_corr_eff, detector_polarity_flip_applied=detector_polarity_flip_applied, lc_win=lc_win, uc_win=uc_win, 
                    ligo_cond_full = ligo_cond_full, umh_cond_full = umh_cond_full, dtype=dtype)

        anchor["diag"] = diag; summary[anchor_detector] = diag
        anchor["resid_cond"] = resid_cond; anchor["resid_w"] = resid_w;       
        anchor["dt_env2_ligo"] = anchor_dt_env2_ligo = dt_env2_ligo; anchor["dt_env3_ligo"] = anchor_dt_env3_ligo = dt_env3_ligo;
        anchor["dt_env2_umh"]  = anchor_dt_env2_umh  = dt_env2_umh;  anchor["dt_env3_umh"]  = anchor_dt_env3_umh  = dt_env3_umh;
    else: global_pol_eff = global_pol; anc_rho_sgn_corr_eff = anc_rho_sgn_corr

    # Process each detector using the Global Anchored Alignment.
    for detector in det_fnd:
        print()
        print(f"================ Starting: SNR and Visual: {detector} ================")
        dr = det_results[detector]

        fs =               dr["fs"]

        N_ligo =           dr["N_ligo"]
        idx_ligo_peak =    dr["idx_ligo_peak"]
        ligo_w_full =      dr["ligo_w_full"]
        ligo_cond_full =   dr["ligo_cond_full"]
        lw_win =           dr["lw_win"]
        
        umh_w_short =      dr["umh_w_short"]
        umh_w_full =       dr["umh_w_full"]
        umh_cond_full =    dr["umh_cond_full"]
        #uw_win =           dr["uw_win"]

        resid_cond =       dr["resid_cond"]
        resid_w =          dr["resid_w"]
        
        start_idx =        dr["start_idx"]
        i0 =               dr["i0"]
        i1 =               dr["i1"]
        i0_snr =           dr["i0_snr"]
        i1_snr =           dr["i1_snr"]
        idx_center =       dr["idx_center"]
        idx_merge_loc =    dr["idx_merge_loc"]

        tau =              dr["tau"]
        tau_resid =        dr["tau_resid"]
        geom_delay_sec_raw = dr["geom_delay_sec_raw"]
        geom_delay_sec_eff = dr["geom_delay_sec_eff"]
        t_peak_obs =       dr["t_peak_obs"]
        t_peak_ligo_abs =  dr["t_peak_ligo_abs"]
        F_plus =           dr["F_plus"]
        F_cross =          dr["F_cross"]
        sign_pred_gen =    dr["sign_pred_gen"]
        
        rho_signed =       dr["rho_signed"]
        sign_corr =        dr["sign_corr"]
        detector_polarity_flip_applied = dr["detector_polarity_flip_applied"]
        
        lag_samp =         dr["lag_samples_peak"]
        t_event_sec =      dr["t_event_est_diag"]
        lag_meas_sec =     dr["lag_meas_sec"]

        # ------------------------------------------------------------
        # Align all detectors from Anchor Detector so only one alignment.
        # ------------------------------------------------------------
        if(anchor_detector != detector):
            print(f"Non Anchor {detector}, Re-Align based on {anchor_detector}")
            umh_w_full, umh_cond_full, peak_lag, i0_snr, i1_snr, idx_center, idx_merge_loc, tau, shift_samples, \
            delta_geom_sec, lag_meas_sec = align_umh_to_global(config, detector, fs, N_ligo, start_idx, t_merge_obs, umh_w_full,
                                                                  umh_cond_full, anchor_i0_snr, anchor_i1_snr, anchor_idx_center, 
                                                                  geom_delay_sec_eff)

            dr["umh_w_full"] = umh_w_full; dr["umh_cond_full"] = umh_cond_full;
            dr["peak_lag"] = peak_lag; dr["i0_snr"] = i0_snr; dr["i1_snr"] = i1_snr;
            dr["idx_center"] = idx_center; dr["idx_merge_loc"] = idx_merge_loc; dr["tau"] = tau; dr["lag_meas_sec"] = lag_meas_sec

            rho_signed, rho_abs, t_peak, match_lagged, lag_samp, i0_snr, i1_snr  = matched_filter_snr_window(config,
                                                detector, fs, N_ligo, ligo_w_full, umh_w_full, idx_center, tau, i0, i1, dtype=dtype)
            dr["rho_signed"] = rho_signed; dr["rho_abs"] = rho_abs; dr["t_peak"] = t_peak; dr["lag_samples_peak"] = lag_samp;
            dr["i0_snr"] = i0_snr; dr["i1_snr"] = i1_snr;
            
            dr["lw_win"] = lw_win = ligo_w_full[i0_snr:i1_snr].astype(dtype); dr["uw_win"] = uw_win = umh_w_full[i0_snr:i1_snr].astype(dtype)
            num_L = np.vdot(lw_win, lw_win).real; num_U = np.vdot(uw_win, uw_win).real
            if not (np.isfinite(num_L) and np.isfinite(num_U)) or num_U <= EPS_SAFE_FLOOR: k_phys = float("nan")
            else: k_phys = float(np.sqrt(num_L / num_U))
            corr = np.vdot(lw_win, uw_win).real
            if abs(corr) > EPS_SAFE_FLOOR: sign_corr = 1.0 if corr > 0 else -1.0
            else: sign_corr = 1.0

            dr["lw_win"] = lw_win; dr["uw_win"] = uw_win
            dr["k_phys"] = k_phys; dr["sign_corr"] = sign_corr

            # Unwhitened (but bandpassed/conditioned) windows for distance-invariant amplitude ratios
            lc_win = ligo_cond_full[i0_snr:i1_snr].astype(float)
            uc_win = umh_cond_full[i0_snr:i1_snr].astype(float)
            resid_cond = ligo_cond_full - umh_cond_full

            ligo_xcorr_delay_int_sec, ligo_xcorr_delay_sec, ligo_xcorr_strength = meas_delay_xcorr_sec(fs, anchor_ligo_w_full, ligo_w_full, idx_center=anchor_idx_center)
            umh_xcorr_delay_int_sec, umh_xcorr_delay_sec, umh_xcorr_strength = meas_delay_xcorr_sec(fs, anchor_umh_w_full, umh_w_full, idx_center=anchor_idx_center)

            diag, resid_cond, resid_w, dt_env2_ligo, dt_env3_ligo, dt_env2_umh, dt_env3_umh = chirp_diagnostics(config, detector, fs, f_min, f_merge, f_ref, 
                        lw_win, uw_win, ligo_w_full, umh_w_full, k_phys, idx_merge_loc,
                        rho_signed, lag_samp, i0_snr, i1_snr, geom_delay_sec_raw=geom_delay_sec_raw, geom_delay_sec_eff=geom_delay_sec_eff, t_peak_abs=t_peak_ligo_abs,
                        t_event_sec=t_event_sec, lag_meas_sec=lag_meas_sec, anchor_t_peak_abs=anchor_t_peak_ligo_abs,
                        BINARY_IOTA_DEG=BINARY_IOTA_DEG, pol_psi_deg=pol_psi_deg, F_plus=F_plus, F_cross=F_cross, sign_pred_gen=sign_pred_gen, 
                        global_pol=global_pol_eff, sign_corr=sign_corr, detector_polarity_flip_applied=detector_polarity_flip_applied, lc_win=lc_win, uc_win=uc_win, 
                        ligo_xcorr_delay_sec=ligo_xcorr_delay_sec, ligo_xcorr_strength=ligo_xcorr_strength, 
                        umh_xcorr_delay_sec=umh_xcorr_delay_sec, umh_xcorr_strength=umh_xcorr_strength, 
                        ligo_cond_full = ligo_cond_full, umh_cond_full = umh_cond_full, dtype=dtype,
                        anchor_dt_env2_ligo = anchor_dt_env2_ligo, anchor_dt_env3_ligo = anchor_dt_env3_ligo, 
                        anchor_dt_env2_umh  = anchor_dt_env2_umh,  anchor_dt_env3_umh  = anchor_dt_env3_umh)

            dr["diag"] = diag; summary[detector] = diag
            dr["resid_cond"] = resid_cond; dr["resid_w"] = resid_w;
        else: diag = dr["diag"]; summary[detector] = diag


        if config.get("GENERATE_VISUALS", True):
            # fs to use
            fs_base = float(fs)
            print(f"Generate Visuals: fs_base:{fs_base}")
            fs_spec = float(config.get("SPEC_FS", fs_base*1))
            t_full  = np.arange(N_ligo) / fs_base

            # merge time in detector frame
            t_merge_det = float(idx_merge_loc / fs_base)

            # choose display window (in seconds)
            #i0_Dsp = anchor_i0; i1_Dsp = anchor_i1
            i0_Dsp = i0; i1_Dsp = i1
            rsd_before_sec = int(round(float(config.get("VIS_WIN_ADJ_BEFORE_SEC", -0.020)) * fs_base))
            rsd_after_sec  = int(round(float(config.get("VIS_WIN_ADJ_AFTER_SEC",  +0.010)) * fs_base))
            i0_Dsp += rsd_before_sec; i1_Dsp += rsd_after_sec
            i0_Dsp = max(0, i0_Dsp); i1_Dsp = min(N_ligo, i1_Dsp)

            t0_Dsp = (i0_Dsp / fs_base); t1_Dsp = (i1_Dsp / fs_base)
            print(f"Zoom window (detector frame): t0={t0_Dsp:.6f}s, t1={t1_Dsp:.6f}s, t_merge_det={t_merge_det:.6f}s, "
                  f"t_merge_obs={t_merge_obs:.6f}s, tau_global={tau:.6f}s, geom_delay_eff={geom_delay_sec_eff:.6f}s")
            if i1_Dsp <= i0_Dsp + 4:
                raise RuntimeError(f"Zoom window collapsed: t0={(i0_Dsp/fs_base):.6f}s, t1={(i1_Dsp/fs_base):.6f}s, "
                    f"t_merge_obs={t_merge_obs:.6f}s, tau_global={tau:.6f}s, geom_delay_eff={geom_delay_sec_eff:.6f}s")
            
            pad_frac = float(config.get("YLIM_PAD_FRAC", 0.65))  # 65% padding

            t_win         = t_full[i0_Dsp:i1_Dsp]
            ligo_cond_win = ligo_cond_full[i0_Dsp:i1_Dsp]
            ligo_win      = ligo_w_full[i0_Dsp:i1_Dsp]
            umh_cond_win  = umh_cond_full[i0_Dsp:i1_Dsp]
            umh_win       = umh_w_full[i0_Dsp:i1_Dsp]
            res_win       = resid_w[i0_Dsp:i1_Dsp]


            # -----------------------------------------------------------
            # Whitened overlay (Aligned)
            # -----------------------------------------------------------
            t = np.arange(N_ligo) / fs_base
            plt.figure(figsize=(10, 4))
            plt.plot(t, ligo_w_full, label="LIGO whitened", alpha=0.9)
            plt.plot(t, umh_w_full, label="UMH whitened (aligned)", alpha=0.7)
            plt.xlabel("Time [s]")
            plt.ylabel("Whitened strain [arb]")

            plt.axvline(t_merge_det, color="k", ls="--", lw=0.5, alpha=0.3, label=r"$t_{\rm merge}$")

            # Dynamically scale Y-axis to the central 98% of data within chirp window
            y = np.concatenate((ligo_win, umh_win))
            yl, yu = np.percentile(y, [1, 99])
            span = yu-yl
            if span < 1e-6: yu, yl = 5e-3, -5e-3  # fallback small symmetric range
            pad = pad_frac * span
            plt.ylim(yl-pad, yu+pad)
            plt.xlim(i0_Dsp/fs_base, i1_Dsp/fs_base)
            plt.title(f"{title}: Whitened Overlay — {detector}")
            plt.legend(loc="lower left"); plt.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_Overlay.png", dpi=dpi)
            plt.close()

    
            # -----------------------------------------------------------
            # Dual-pane spectrogram (Aligned) on whitened data (LIGO[Left] vs UMH[Right])
            # -----------------------------------------------------------
            win = tukey(len(ligo_win), alpha=0.1)
            ligo_win_t = ligo_win * win
            umh_win_t  = umh_win  * win

            # --- STFT parameters tuned for this fs_spec ---
            win_sec  = float(config.get("SPEC_WIN_SEC", 0.032))  # 32 ms default
            n_target = int(round(fs_spec * win_sec))
            nperseg  = 1 << int(math.ceil(math.log2(max(512, n_target))))
            nperseg  = int(min(4096, max(512, nperseg)))
            noverlap = int(0.92 * nperseg)
            window   = "hann"

            SPEC_FS      = float(config.get("SPEC_FS", fs_base)) # ~16384
            SPEC_WIN_SEC = float(config.get("SPEC_WIN_SEC", 0.036))
            SPEC_OVERLAP = float(config.get("SPEC_OVERLAP_FRAC", 0.82))

            NPER  = next_pow2(int(SPEC_WIN_SEC * SPEC_FS))       # → 4096
            NPER  = max(256, min(NPER, 4096))                    # stays 4096
            NOVER = int(SPEC_OVERLAP * NPER)                     # → 3072

            if config.get("ADD_VISUAL_NOISE", False): p_lo, p_hi = 0.3, 99.7
            else: p_lo, p_hi = 95.3, 99.7

            f_L, t_L, S_L_db, vmin_L, vmax_L = build_spectrogram(ligo_win_t, fs_base, fs_spec, i0_Dsp, NPER, NOVER, p_lo, p_hi)
            f_U, t_U, S_U_db, vmin_U, vmax_U = build_spectrogram(umh_win_t,  fs_base, fs_spec, i0_Dsp, NPER, NOVER, p_lo, p_hi)

            vmin = min(vmin_L, vmin_U)
            vmax = max(vmax_L, vmax_U)

            # Flatten dB values
            #all_vals = np.concatenate([S_L_db.ravel(), S_U_db.ravel()])

            # Compute robust percentiles
            #vmin = np.percentile(all_vals, p_lo)   # Lower 2% of data → dark blue
            #vmax = np.percentile(all_vals, p_hi)   # Upper 2% → bright yellow

            all_vals = np.concatenate([S_L_db.ravel(), S_U_db.ravel()])
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size < 10: vmin, vmax = -120.0, -40.0
            else: vmin = np.nanpercentile(all_vals, p_lo); vmax = np.nanpercentile(all_vals, p_hi)
            if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax): vmin, vmax = -120.0, -40.0

            #S_L_db = gaussian_filter(S_L_db, sigma=(0.25, 0.12))
            #S_U_db = gaussian_filter(S_U_db, sigma=(0.25, 0.12))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

            # Left: LIGO
            #im1 = ax1.pcolormesh(t_L, f_L, S_L_db, shading="auto", vmin=vmin_L, vmax=vmax_L, cmap=ligo_cmap())
            extent = [t_L[0], t_L[-1], f_L[0], f_L[-1]] 
            im1 = ax1.imshow(S_L_db, extent=extent, aspect='auto', origin='lower', interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=ligo_cmap())

            ax1.set_title(f"LIGO ({detector})")
            ax1.set_ylim(float(band_lo) - 10.0, float(band_hi) + 50.0)

            # Right: UMH
            #im2 = ax2.pcolormesh(t_U, f_U, S_U_db, shading="auto", vmin=vmin_U, vmax=vmax_U, cmap=ligo_cmap())
            extent = [t_U[0], t_U[-1], f_U[0], f_U[-1]] 
            im2 = ax2.imshow(S_U_db, extent=extent, aspect='auto', origin='lower', interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=ligo_cmap())

            ax2.set_title(f"UMH ({detector})")
            ax2.set_ylim(float(band_lo) - 10.0, float(band_hi) + 50.0)
            ax1.set_xlim(t_L.min(), t_L.max())
            ax2.set_xlim(t_U.min(), t_U.max())

            # --- Optional contour overlay: emphasize chirp ridge on UMH panel ---
            try:
                SPECTRO_CONTOUR_STRENGTH = float(config.get("SPECTRO_CONTOUR_STRENGTH", 0.8))
                levels = np.linspace(vmin + SPECTRO_CONTOUR_STRENGTH * (vmax - vmin), vmax, 6)
                ax1.contour(t_U, f_U, S_U_db, levels=levels, colors="w", linewidths=0.8, alpha=0.75)
                ax2.contour(t_U, f_U, S_U_db, levels=levels, colors="w", linewidths=0.8, alpha=0.75)
            except Exception: pass

            # --- Overlay: model f_GW(t) track from generator (if available) ---
            try:
                if(t_track_obs is not None and f_track_obs is not None):
                    t_track_det = t_track_obs + (start_idx / fs_base) + float(tau) + geom_delay_sec_raw
                    #print(f"[INFO] Start:{(start_idx / fs_base)}, tau:{tau}, geom_delay_sec_raw:{geom_delay_sec_raw}, geom_delay_sec_eff:{geom_delay_sec_eff}")

                    t0_win = i0_Dsp / fs_base; t1_win = i1_Dsp / fs_base
                    mask = (t_track_det >= t0_win) & (t_track_det <= t1_win)

                    if np.any(mask): ax1.plot(t_track_det[mask], f_track_obs[mask], color="w", lw=1.0, alpha=0.9, label=r"$f_{\rm GW}(t)$")
                    if np.any(mask): ax2.plot(t_track_det[mask], f_track_obs[mask], color="w", lw=1.0, alpha=0.9, label=r"$f_{\rm GW}(t)$")
            except Exception as e: print(f"Generate Visuals: [{detector}] freq_track overlay skipped: {e}")

            # --- Vertical line: merger time (from generator) mapped into detector frame ---
            for ax in (ax1, ax2): ax.axvline(t_merge_det, color="k", ls="--", lw=0.5, alpha=0.3, label=r"$t_{\rm merge}$")

            # Only show legend if we actually plotted overlays
            for ax in (ax1, ax2):
                handles, labels = ax.get_legend_handles_labels()
                if handles: ax.legend(loc="upper right", frameon=True); #break  # one legend is enough

            # Shared colorbar (use imU or imL, doesn't matter)
            cbar = fig.colorbar(im2, location='right', fraction=0.046, pad=0.04)
            cbar.set_label("Whitened Power [dB]")
                 
            fig.supxlabel("Time [s]", y=0.05) 
            fig.supylabel("Frequency [Hz]")
            fig.suptitle(f"{title}: Dual Spectrogram — {detector}", fontsize=20, y=0.97) #fontsize=20, y=0.98
            fig.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_Spectrogram_Dual.png", dpi=dpi)
            plt.close(fig)


            print(f"Generate Visuals: [{detector}] RMS (window) LIGO   =", stable_rms(ligo_cond_win)) #np.sqrt(np.mean(ligo_cond_win**2)))
            print(f"Generate Visuals: [{detector}] RMS (window) UMH    =", stable_rms(umh_cond_win)) #np.sqrt(np.mean(umh_cond_win**2)))
            print(f"Generate Visuals: [{detector}] PEAK (window) LIGO  =", np.max(np.abs(ligo_cond_win)))
            print(f"Generate Visuals: [{detector}] PEAK (window) UMH   =", np.max(np.abs(umh_cond_win)))


            # -----------------------------------------------------------
            # ASD (Window) comparison (aligned, same conditioning, no whitening)
            # -----------------------------------------------------------          
            freqs_F, asd_FL = compute_asd(ligo_cond_full, fs_base, nperseg=NPER_PSD, noverlap=NOVER_PSD)
            _,     asd_FU =   compute_asd(umh_cond_full,  fs_base, nperseg=NPER_PSD, noverlap=NOVER_PSD)
            plt.figure()
            plt.loglog(freqs_F, asd_FL, label="LIGO")
            plt.loglog(freqs_F, asd_FU, label="UMH")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("ASD [strain/√Hz]")
                
            # ASD x-axis limits
            asd_lo = max(10.0, 0.8 * f_min)
            asd_hi = min(Fn_umh, 2.0 * f_merge)

            plt.xlim(asd_lo, asd_hi)
            print(f"Generate Visuals: ASD: Lo:{asd_lo}, Hi:{asd_hi}")

            for ax in (ax1, ax2): ax.set_ylim(float(band_lo) - 10.0, float(band_hi) + 50.0)

            plt.title(f"{title}: Amplitude Spectral Density Window Comparison\n({detector})")
            plt.legend(loc="lower left", frameon=True); plt.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_ASD.png", dpi=dpi)
            plt.close()


            # -----------------------------------------------------------
            # ASD (Full) comparison (aligned, same conditioning, no whitening)
            # -----------------------------------------------------------
            plt.figure()
            freqs_F, asd_FL = compute_asd(ligo_cond_full, fs_base, nperseg=NPER_PSD, noverlap=NOVER_PSD)
            _,     asd_FU =   compute_asd(umh_cond_full,  fs_base, nperseg=NPER_PSD, noverlap=NOVER_PSD)
            plt.figure()
            plt.loglog(freqs_F, asd_FL, label="LIGO")
            plt.loglog(freqs_F, asd_FU, label="UMH")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("ASD [strain/√Hz]")

            plt.title(f"{title}: Amplitude Spectral Density Full Comparison\n({detector})")
            plt.legend(loc="lower left", frameon=True); plt.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_ASD_Full.png", dpi=dpi)
            plt.close()


            # -----------------------------------------------------------
            # Residual whitened overlay  (time-series, zoomed on SNR window)
            # -----------------------------------------------------------
            # choose zoom window (in seconds)
            rsd_before_sec = float(config.get("VIS_RSD_WIN_BEFORE_SEC", 0.12))
            rsd_after_sec  = float(config.get("VIS_RSD_WIN_AFTER_SEC",  0.06))
            t_lo = t_merge_det - rsd_before_sec   # 120 ms before merger
            t_hi = t_merge_det + rsd_after_sec   # 40 ms after merger
            # build time axis for whitened signals
            t_full = np.arange(N_ligo) / fs_base
            # find indices
            mask = (t_full >= t_lo) & (t_full <= t_hi)

            t_win_z    = t_full[mask]
            ligo_win_z = ligo_w_full[mask]
            umh_win_z  = umh_w_full[mask]
            res_win_z  = resid_w[mask]

            plt.figure(figsize=(10, 4))
            plt.plot(t_win_z, ligo_win_z, label="LIGO whitened", alpha=0.85)
            plt.plot(t_win_z, umh_win_z,  label="UMH whitened (aligned)", alpha=0.8)
            plt.plot(t_win_z, res_win_z,  label="Residual (LIGO – UMH)", alpha=0.9)

            plt.figure(figsize=(10, 4))
            plt.plot(t_win_z, ligo_win_z, label="LIGO whitened", alpha=0.85)
            plt.plot(t_win_z, umh_win_z,  label="UMH whitened (aligned)", alpha=0.8)
            plt.plot(t_win_z, res_win_z,  label="Residual (LIGO – UMH)", alpha=0.9)
            
            plt.axvline(t_merge_det, color="k", ls="--", lw=0.5, alpha=0.3, label=r"$t_{\rm merge}$")

            plt.xlabel("Time [s]")
            plt.ylabel("Whitened strain [arb]")
            plt.title(f"{title}: Whitened Residual - {detector}")
            
            # Nice symmetric y-limits around 0 based on central 98% of the residual
            y = np.concatenate((ligo_win_z, umh_win_z, res_win_z))
            yl, yu = np.percentile(y, [1, 99])
            span = yu-yl
            if span < 1e-6: yu, yl = 5e-3, -5e-3  # fallback small symmetric range
            pad = pad_frac * span
            plt.ylim(yl-pad, yu+pad)

            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_Residual_Overlay.png", dpi=dpi)
            plt.close()


            # -----------------------------------------------------------
            # Residual spectrogram (whitened)
            # -----------------------------------------------------------
            f_R, t_R, S_R_db, vmin_R, vmax_R = build_spectrogram(res_win, fs_base, fs_spec, i0_Dsp, NPER, NOVER, p_lo, p_hi)

            vmin_res = min(vmin, vmin_R); vmax_res = max(vmax, vmax_R)
            vals = [vmin, vmin_R]

            vals = [z for z in vals if np.isfinite(z)]
            vmin_res = min(vals) if vals else -120.0
            vals = [vmax, vmax_R]
            vals = [z for z in vals if np.isfinite(z)]
            vmax_res = max(vals) if vals else -40.0
            if not (np.isfinite(vmin_res) and np.isfinite(vmax_res) and vmin_res < vmax_res): vmin_res, vmax_res = -120.0, -40.0

            plt.figure(figsize=(10, 6))
            extent = [t_R[0], t_R[-1], f_R[0], f_R[-1]]
            plt.imshow(S_R_db, extent=extent, aspect='auto', origin='lower',
                interpolation='bicubic', cmap=ligo_cmap(), vmin=vmin_res, vmax=vmax_res)

            # Frequency range
            plt.ylim(float(band_lo) - 10.0, float(band_hi) + 50.0)

            # --- Overlay: model f_GW(t) track (for "no residual chirp" check) ---
            try:
                # Map intrinsic track into this detector frame
                t_track_det = t_track_obs + (anchor_start_idx / fs_base) + float(tau) + geom_delay_sec_raw
                mask = (t_track_det >= t_R[0]) & (t_track_det <= t_R[-1])
                if np.any(mask): plt.plot(t_track_det[mask], f_track_obs[mask], color='white', lw=1.0, alpha=0.9, label=r'$f_{\rm GW}(t)$')
            except Exception as e: print(f"Generate Visuals: [{detector}] residual freq_track overlay skipped: {e}")

            # overlays: f_GW(t) and t_merge
            plt.axvline(t_merge_det, color='k', ls='--', lw=0.5, alpha=0.3, label=r'$t_{\rm merge}$')

            # --- Optional contour overlay: emphasize chirp ridge on UMH panel ---
            try:
                SPECTRO_CONTOUR_STRENGTH = float(config.get("SPECTRO_CONTOUR_STRENGTH", 0.8))
                levels = np.linspace(vmin_res + SPECTRO_CONTOUR_STRENGTH * (vmax_res - vmin_res), vmax_res, 6)
                plt.contour(t_U, f_U, S_U_db, levels=levels, colors="w", linewidths=0.8, alpha=0.75)
            except Exception: pass

            plt.title(f"{title}: Residual Spectrogram - {detector}")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            cbar = plt.colorbar(label="Whitened Power [dB]")

            # Only show legend if we actually plotted something
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles: plt.legend(loc='upper right', frameon=True)

            plt.tight_layout()
            plt.savefig(f"{file_path}_CMP_{detector}_Residual_Spectrogram.png", dpi=dpi)
            plt.close()


        if config.get("GENERATE_NPZ", True):
            # --- Save residuals ---
            np.savez(f"{file_path}_CMP_{detector}_Residual_phys.npz", resid_cond.astype(dtype))
            np.savez(f"{file_path}_CMP_{detector}_Residual_white.npz", resid_w.astype(dtype))
            
        print(f"================ Completed: SNR and Visual: {detector} ================")
    # End: Process each detector using the Global Anchored Alignment.


    # --- Global summary JSON ---
    print()

    # --- Global Values for all detectors used ---
    summary["GLOBAL"] = {}

    summary["GLOBAL"]["M1_solar_src"]    = M1_solar_src
    summary["GLOBAL"]["M2_solar_src"]    = M2_solar_src
    summary["GLOBAL"]["distance_Mpc"]    = distance_Mpc
    summary["GLOBAL"]["ra_deg"]          = ra_deg
    summary["GLOBAL"]["dec_deg"]         = dec_deg
    summary["GLOBAL"]["pol_psi_deg"]     = pol_psi_deg
    summary["GLOBAL"]["BINARY_IOTA_DEG"] = BINARY_IOTA_DEG

    summary["GLOBAL"]["PHYSICS_STRICT"]  = PHYSICS_STRICT
    summary["GLOBAL"]["band"]            = [band_lo, band_hi]
    summary["GLOBAL"]["umh_provenance"]  = {"distance_Mpc":float(umh.get("distance_Mpc")),
                            "UMH_z_tension":float(umh.get("UMH_z_tension", np.nan)),
                            "z_GR":float(umh.get("z_GR", np.nan))}

    summary["GLOBAL"]["psd"] = {"NPER_PSD": int(config.get("NPER_PSD", config.get("NPER", 4096))), 
        "NOVER_PSD": int(config.get("NOVER_PSD", config.get("NOVER", 2048))), 
        "band": {"band_lo": float(band_lo), "band_hi": float(band_hi)}}

    # --- NETWORK SNR for any detectors used ---
    summary["NETWORK"] = {"Anchor": anchor_detector, "Polarity": global_pol, "Network_Polarity_Flip_Applied": network_polarity_flip_applied, "Polarity_Effective": global_pol_eff}
    if all(det in summary for det in det_fnd):
        rho_list = []
        for det in det_fnd:
            val = summary[det].get("rho_peak_abs", None)
            if isinstance(val, (int, float)) and np.isfinite(val): rho_list.append(float(val))
        if rho_list: rho_net = float(np.sqrt(np.sum(np.square(rho_list))))
        else: rho_net = None
        summary["NETWORK"]["rho_net"] = rho_net

    with open(f"{file_path}_CMP_Summary.json", "w") as f: json.dump(summary, f, indent=2, sort_keys=False)
    print(f"[done] Summary written to '{file_path}_CMP_Summary.json'")

    # --- Save PSD map for use by the generator ---
    # This will contain arrays like Hanford_freqs, Hanford_psd, Livingston_freqs, Livingston_psd, etc.
    if config.get("GENERATE_NPZ", True) and config.get("GENERATE_PSD_MAP", True):
        psd_npz_dict = {}
        for det, data in psd_map_dict.items():
            psd_npz_dict[f"{det}_freqs"] = data["freqs"]
            psd_npz_dict[f"{det}_psd"]   = data["psd"]

        np.savez(f"{file_path}_CMP_PSD_Map.npz", **psd_npz_dict)
        print(f"[done] PSD map written to '{file_path}_CMP_PSD_Map.npz'")


    print(f"✅ Finished Test: {title} Completed (see summary for match quality).")

    return summary
    


def str2bool(s):
    if isinstance(s, bool): return s
    s = str(s).lower()
    if s in ("1", "true", "yes", "y", "on"): return True
    if s in ("0", "false", "no", "n", "off"): return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{s}' as bool.")

if __name__ == "__main__":
    overrides = {}

    parser = argparse.ArgumentParser(description="UMH LIGO Compiler – compare UMH chirp to LIGO data.")
    # Optional positional JSON overrides file
    parser.add_argument( "config_path", nargs="?", help="Optional JSON overrides file (positional).")
    # Optional named JSON overrides as well (--overrides)
    parser.add_argument("--overrides", dest="config_path_named", help="Optional JSON overrides file (named).")
    # Explicit override knobs
    parser.add_argument("--profile", help="Build in Profile to Use")

    parser.add_argument("--OUTPUT_FOLDER", type=str, help="Override output folder for plots/metadata.")
    parser.add_argument("--GENERATE_VISUALS", type=str2bool, help="Override whether to generate plots/visuals.")
    parser.add_argument("--GENERATE_NPZ", type=str2bool, help="Override whether to generate NPZ files.")

    args = parser.parse_args()

    cfg_path = args.config_path_named or args.config_path
    if cfg_path is not None:
        with open(cfg_path, "r") as f: file_cfg = json.load(f)
        overrides.update(file_cfg)

    if args.profile is not None:          overrides["profile"]          = args.profile

    if args.OUTPUT_FOLDER is not None:    
        overrides["OUTPUT_FOLDER"] = args.OUTPUT_FOLDER
        overrides["INPUT_FOLDER"]  = os.path.join(args.OUTPUT_FOLDER, "UMH_vs_LIGO")
    if args.GENERATE_VISUALS is not None: overrides["GENERATE_VISUALS"] = args.GENERATE_VISUALS
    if args.GENERATE_NPZ is not None:     overrides["GENERATE_NPZ"]     = args.GENERATE_NPZ

    run_ligo_compiler_test(overrides)