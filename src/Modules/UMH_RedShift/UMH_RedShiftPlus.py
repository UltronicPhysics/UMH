"""
UMH_RedShiftPlus.py (UMH RedShift Test)

RedShift under Ultronic Medium

Author: Andrew Dodge
Date: July 2025

Implements the low-z calibration and Pantheon+ redshift/time-dilation
analysis described in:
  A. Dodge, "Pantheon+ and Redshift Validation of the Ultronic Medium Hypothesis (UMH)", 2025.

Description:
  Tests whether a RedShift under Ultronic Medium can occur without Universe expansion.
  Performs low-z calibration of the UMH redshift law and Pantheon+ redshift/time-dilation tests
  under the non-expansion UMH framework.
"""

import numpy as np
import os
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing matplotlib
import sys
import json
from math import isfinite

import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.linalg import cho_factor, cho_solve

from scipy.stats import skew, kurtosis



def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.

        "LIGHT_SPEED": 299792.458,  # speed of light in km/s

        "USE_HUBER": True,

        "VPEC": 200, # km/s

        "H0": 70, # Hubble constant in km/s/Mpc

        "PANTHEON_DATA_COLUMNS":["CID","IDSURVEY","zHD","zHDERR","zCMB","zCMBERR","zHEL","zHELERR","m_b_corr","m_b_corr_err_DIAG","MU_SH0ES","MU_SH0ES_ERR_DIAG","CEPH_DIST","IS_CALIBRATOR","USED_IN_SH0ES_HF","c","cERR","x1","x1ERR","mB","mBERR","x0","x0ERR","COV_x1_c","COV_x1_x0","COV_c_x0","RA","DEC","HOST_RA","HOST_DEC","HOST_ANGSEP","VPEC","VPECERR","MWEBV","HOST_LOGMASS","HOST_LOGMASS_ERR","PKMJD","PKMJDERR","NDOF","FITCHI2","FITPROB","m_b_corr_err_RAW","m_b_corr_err_VPEC","biasCor_m_b","biasCorErr_m_b","biasCor_m_b_COVSCALE","biasCor_m_b_COVADD"],
        "PANTHEON_DATA_FILE":os.path.join(base, "Output", "PantheonData", "PantheonPlus_SH0ES.dat"),
        
        "PANTHEON_DATA_BIAS_FILE":os.path.join(base, "Output", "PantheonData", "PantheonPlus_SH0ES_STAT_SYS.cov"),

        "GENERATE_UMH_SIMULATION_CALIBRATION": True,

        "DPI":300, #PNG Resolution.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


# -------------------------------------------------------------
# Low-z calibration helpers (calibrators only;)
# -------------------------------------------------------------
def solve_wls(A, y, w):
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    yw = y * sw
    # normal equations
    ATA = Aw.T @ Aw
    ATy = Aw.T @ yw
    theta = np.linalg.pinv(ATA) @ ATy
    # covariance (scaled by chi2/dof)
    res = y - A @ theta
    chi2 = float(np.sum(w * res**2))
    dof = max(len(y) - A.shape[1], 1)
    s2 = chi2 / dof
    cov = np.linalg.pinv(ATA) * s2
    return theta, cov, res, chi2, dof


def fit_alpha_huber(d, L, w_meas, huber_c=2.0, force_through_zero=False, max_iter=50):
    """
    Low-z calibration:
      model: d = c0 + c1 * L             (if force_through_zero: c0=0)
      weights: w_meas                    (distance uncertainties)
      robust: Huber IRLS with parameter huber_c
    Returns: a_hat (a*), b0 (c0), sigma_a, info
    """
    d = np.asarray(d, float)
    L = np.asarray(L, float)

    # Design matrix: d = c0 + c1 * L
    if force_through_zero:
        A = L[:, None]                          # shape (N,1)
        p = 1
    else:
        A = np.vstack([np.ones_like(L), L]).T   # shape (N,2)
        p = 2

    w = w_meas.copy()

    theta, cov, res, chi2, dof = solve_wls(A, d, w)

    # Huber IRLS
    for _ in range(max_iter):
        # robust scale (MAD)
        s = max(1.4826 * np.median(np.abs(res - np.median(res))), 1e-9)
        u = res / s
        w_rob = np.where(np.abs(u) <= huber_c, 1.0, huber_c / np.abs(u))
        w_new = w_meas * w_rob

        theta_new, cov_new, res_new, chi2_new, dof_new = solve_wls(A, d, w_new)

        if np.allclose(theta_new, theta, rtol=1e-8, atol=1e-10):
            theta, cov, res, chi2, dof = theta_new, cov_new, res_new, chi2_new, dof_new
            break

        theta, cov, res, chi2, dof = theta_new, cov_new, res_new, chi2_new, dof_new
        w = w_new

    # Map to a* = 1/c1 and its uncertainty
    if force_through_zero:
        c0 = 0.0
        c1 = float(theta[0])
        var_c1 = float(cov[0, 0])
    else:
        c0 = float(theta[0])
        c1 = float(theta[1])
        var_c1 = float(cov[1, 1])

    a_hat = 1.0 / c1
    sigma_a = np.sqrt(var_c1) / (c1 * c1)

    info = {
        "c0": c0,
        "c1": c1,
        "sigma_c1": np.sqrt(var_c1),
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2 / dof if dof > 0 else np.nan,
        "huber_c": huber_c,
        "force_through_zero": bool(force_through_zero),
        "N": int(d.size),
    }

    if not force_through_zero and cov is not None and cov.shape == (2, 2):
        info["cov_c00"] = float(cov[0, 0])
        info["cov_c01"] = float(cov[0, 1])
        info["cov_c11"] = float(cov[1, 1])

    return a_hat, c0, sigma_a, info


def build_calibrator_vectors(df, c_kms,
                             zmax=0.10,
                             dmax=120.0,
                             sigma_ceph_default=0.20,   # Mpc (fallback)
                             vpec_kms=250.0):
    """
    Build calibrator vectors (Pantheon+ SH0ES calibrators).
    Returns: d_cal, sig_d, z_cal, sig_L
    """
    m_cal = (df["IS_CALIBRATOR"].astype(int) == 1)

    # Prefer SH0ES distance modulus; fallback to CEPH_DIST
    mu     = df.loc[m_cal, "MU_SH0ES"].to_numpy(float)
    mu_err = df.loc[m_cal, "MU_SH0ES_ERR_DIAG"].to_numpy(float)
    d_from_mu     = np.where(np.isfinite(mu), 10**((mu - 25.0)/5.0), np.nan)  # Mpc
    sig_d_from_mu = (np.log(10.0)/5.0) * d_from_mu * mu_err

    d_ceph     = df.loc[m_cal, "CEPH_DIST"].to_numpy(float)
    sig_d_ceph = np.full_like(d_ceph, sigma_ceph_default, dtype=float)

    use_mu = np.isfinite(d_from_mu)
    d_cal  = np.where(use_mu, d_from_mu, d_ceph)
    sig_d  = np.where(use_mu, sig_d_from_mu, sig_d_ceph)

    # Redshift        
    z_cal  = df.loc[m_cal, "zHD"].to_numpy(float)  # flow-corrected
    z_err  = df.loc[m_cal, "zHDERR"].to_numpy(float)

    # UMH-native L = ln(1+z); include PV floor
    L_cal = np.log1p(z_cal)
    sigma_z_floor = vpec_kms / c_kms
    #sig_L = np.sqrt((z_err/(1.0+z_cal))**2 + (sigma_z_floor/(1.0+z_cal))**2)
    sig_L = np.sqrt((z_err/(1.0+z_cal))**2 + (sigma_z_floor)**2)


    # Clean mask
    m_fin = (
        np.isfinite(d_cal) & np.isfinite(sig_d) &
        np.isfinite(z_cal) & np.isfinite(sig_L) &
        (z_cal > 0) & (z_cal <= zmax) &
        (d_cal > 0) & (d_cal <= dmax)
    )
    return d_cal[m_fin], sig_d[m_fin], z_cal[m_fin], sig_L[m_fin], z_err[m_fin]


def fit_alpha_weighted(d_Mpc, ln1pz, z, sigma_L,
    zmax=0.10, clip_sigma=3.0, force_through_zero=False):
    """
    Weighted fit of ln(1+z) ≈ α d on z ≤ zmax using y-errors sigma_L.
    Returns α, b (intercept), σα, diagnostics.
    """
    m = (z <= float(zmax))
    x = np.asarray(d_Mpc[m], float)
    y = np.asarray(ln1pz[m], float)
    s = np.asarray(sigma_L[m], float)
    if x.size < 8: raise ValueError(f"Too few low-z points (z<= {zmax}); got N={x.size}.")

    w = 1.0/np.maximum(s*s, 1e-30)

    # initial weighted LS (optionally with intercept)
    if force_through_zero:
        # α = (x^T W y)/(x^T W x)
        XtWy = np.dot(x*w, y)
        XtWx = np.dot(x*w, x)
        a_ls = XtWy/max(XtWx, 1e-30)
        b_ls = 0.0
        resid = y - a_ls*x
    else:
        # Solve [x,1] with weights
        X = np.column_stack([x, np.ones_like(x)])
        WX = X * np.sqrt(w)[:,None]
        Wy = y * np.sqrt(w)
        (a_ls, b_ls), *_ = np.linalg.lstsq(WX, Wy, rcond=None)
        resid = y - (a_ls*x + b_ls)

    # robust clip on weighted residuals
    if clip_sigma and clip_sigma > 0:
        sr = np.std(resid)
        if sr > 0:
            keep = np.abs(resid) <= clip_sigma*sr
            x, y, w = x[keep], y[keep], w[keep]

    # final weighted solution
    if force_through_zero:
        XtWy = np.dot(x*w, y)
        XtWx = np.dot(x*w, x)
        a = float(XtWy/max(XtWx, 1e-30)); b = 0.0
        # var(α) = 1/(x^T W x) * χ²_red  (use weighted residual variance)
        resid = y - a*x
        dof = max(x.size-1, 1)
        chi2 = float(np.dot(resid*w, resid))
        chi2_red = chi2/dof
        var_a = chi2_red/max(XtWx, 1e-30)
    else:
        X = np.column_stack([x, np.ones_like(x)])
        WX = X * np.sqrt(w)[:,None]
        Wy = y * np.sqrt(w)
        beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
        a, b = map(float, beta)
        resid = y - (a*x + b)
        dof = max(x.size-2, 1)
        chi2 = float(np.dot(resid*w, resid))
        chi2_red = chi2/dof
        XtWX = X.T @ (w[:,None]*X)
        cov = chi2_red * np.linalg.inv(XtWX)
        var_a = cov[0,0]

    # R² and RMSE (unweighted summaries on the kept low-z subset)
    yhat = a*x + b
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/max(ss_tot, 1e-30)
    rmse = float(np.sqrt(ss_res/max(x.size-(1 if force_through_zero else 2),1)))

    chi2_dof = chi2 / dof

    info = {
        "N_fit": int(x.size), "zmax": float(zmax),
        "chi2": float(chi2), "dof": int(dof), "chi2_dof": float(chi2_dof),
        "R2_unweighted": float(r2), "rmse_unweighted": rmse,
        "clip_sigma": float(clip_sigma), "forced_through_zero": bool(force_through_zero)
    }
    return float(a), float(b), float(np.sqrt(max(var_a,0.0))), info


# -------------------------------------------------------------
# UMH redshift law, inversion, and μ(z) under non-expansion
# -------------------------------------------------------------
def z_of_d_umh(d, a, s=0.0, b=0.0, c_=0.0, d0=1.0):
    """
    ln(1+z) = (a d + b d^2 + c ln(1 + d/d0)) / (1 + s d)
    """
    d = float(d)
    num = a*d + b*(d**2) + c_*np.log1p(d/d0)
    den = 1.0 + s*d
    if abs(den) < 1e-12:
        return np.inf
    x = num/den
    if x > 700:  # avoid overflow
        return np.inf
    if x < -50:
        return 0.0
    return np.expm1(x)


def d_of_z_umh(z_target, a, s=0.0, b=0.0, c_=0.0, d0=1.0, d_init=1.0, d_max=1e9, max_doublings=100):
    """
    Robust scalar inversion of z(d) for general (a,b,c,s).
    """
    zt = float(z_target)
    if zt <= 0.0: return 0.0
    L = float(np.log1p(zt))

    # Analytic inversion for pure a (s=b=c=0)
    if abs(s) < 1e-12 and b == 0.0 and c_ == 0.0:
        if a <= 0.0: raise RuntimeError("UMH: a must be > 0 for s=b=c=0.")
        return L / a

    # Guard: finite z ceiling for some parameter combos
    if b <= 0.0 and s > 0.0:
        z_inf = np.exp(a/s) - 1.0
        if zt >= 0.999*z_inf: raise RuntimeError(f"UMH: requested z={zt:.3g} exceeds model's max z≈{z_inf:.3g}.")

    def f(d): return z_of_d_umh(d, a=a, s=s, b=b, c_=c_, d0=d0) - zt

    if abs(s) < 1e-12:
        hi_candidates = [max(10.0, d_init)]
        if a > 0.0: hi_candidates.append(L / a)
        if b > 0.0: hi_candidates.append(np.sqrt(L / b))
        if c_ > 0.0: hi_candidates.append(d0*np.expm1(L / c_))
        hi = float(max(hi_candidates)); lo = 0.0
        n = 0
        while f(hi) <= 0.0 and hi < d_max and n < max_doublings: hi *= 2.0; n += 1
        if f(hi) <= 0.0: raise RuntimeError("UMH: could not bracket d(z) with s≈0.")
        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)

    # General s != 0
    lo, hi = 0.0, max(1.0, float(d_init))
    d_pole = (-1.0/s) if (s < 0.0) else None

    fhi = f(hi); n = 0
    while fhi <= 0.0 and hi < d_max and n < max_doublings:
        hi *= 2.0
        if d_pole is not None and hi >= 0.99*d_pole:
            hi = 0.99*d_pole
            break
        fhi = f(hi); n += 1
    if d_pole is not None and hi >= 0.99*d_pole and f(hi) <= 0.0:
        raise RuntimeError("UMH: could not bracket d(z) without crossing the s<0 pole.")
    if fhi <= 0.0: raise RuntimeError("UMH: could not bracket d(z).")

    return brentq(f, lo, hi, xtol=1e-10, maxiter=200)

def mu_umh_of_z_nonexp(z_array, a, s=0.0, b=0.0, c_=0.0, d0=1.0, delta=1.0, kappa=1.0, T_of_z=None):
    """
    Distance modulus under UMH non-expansion:
      - invert z -> d using UMH law
      - D_L = (kappa * d) * (1+z)^((1+delta)/2) / sqrt(T(z))
      - μ = 5*log10(D_L) + 25
    """
    if T_of_z is None: T_of_z = lambda z: np.ones_like(np.asarray(z, float), float)

    z_array = np.asarray(z_array, float)

    # Vectorized inversion
    if abs(s) < 1e-12 and b == 0.0 and c_ == 0.0:
        if a <= 0.0: raise RuntimeError("UMH: a must be > 0 for the pure (a) law.")
        d_vals = np.log1p(z_array) / a
    else: d_vals = np.array([d_of_z_umh(zi, a=a, s=s, b=b, c_=c_, d0=d0) for zi in z_array])

    Tvals = np.asarray(T_of_z(z_array))
    if Tvals.ndim == 0: Tvals = np.full_like(z_array, Tvals, dtype=float)

    D_L = (kappa * d_vals) * (1.0 + z_array)**((1.0 + delta)/2.0) / np.sqrt(Tvals)
    return 5.0*np.log10(D_L) + 25.0

def chi2_and_M_best(data_vec, model_mu, C):
    """
    Profile M (absolute magnitude) analytically and return (chi2, M_best).
    """
    one = np.ones_like(data_vec)
    cf  = cho_factor(C, overwrite_a=False, check_finite=False)
    Cinvd   = cho_solve(cf, data_vec - model_mu, check_finite=False)
    Cinvone = cho_solve(cf, one,             check_finite=False)
    M_best  = (one @ Cinvd) / (one @ Cinvone)
    Delta   = data_vec - model_mu - M_best
    chi2    = Delta @ cho_solve(cf, Delta, check_finite=False)
    return chi2, M_best

def make_Texp(beta1, beta2=0.0):
    """T(z) = exp[- τ(L) ], τ(L)=β1*L + β2*L^2,  L=ln(1+z)."""
    def T_of_z(z):
        L = np.log1p(np.asarray(z, float))
        tau = beta1*L + beta2*(L**2)
        return np.exp(-tau)
    return T_of_z



def mu0_umh_nonexp(z, a, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0):
    z = np.asarray(z, float)
    if abs(s) < 1e-12 and b == 0.0 and c_ == 0.0: d_vals = np.log1p(z) / a
    else: d_vals = np.array([d_of_z_umh(zi, a=a, s=s, b=b, c_=c_, d0=d0) for zi in z])
    return 5.0*np.log10(kappa*d_vals) + 2.5*np.log10(1.0+z)


LOG10E = 1.0/np.log(10.0)

def fit_M_gamma_beta2(z, mb_corr, C, a, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0):
    """
    GLS fit of [M, gamma(=δ+β1), beta2].  Removes the δ–β1 collinearity.
    """
    z  = np.asarray(z, float)
    L  = np.log1p(z)

    # Basis: μ = μ0 + [1,  A,     B2] · [M, gamma, beta2]
    A  = 2.5*np.log10(1.0+z)        # multiplies gamma
    B2 = 2.5*LOG10E * (L**2)        # multiplies beta2
    X  = np.column_stack([np.ones_like(z), A, B2])

    mu0 = mu0_umh_nonexp(z, a=a, s=s, b=b, c_=c_, d0=d0, kappa=kappa)
    rhs = np.asarray(mb_corr, float) - mu0

    # Symmetrize C and add tiny jitter for stability
    C = 0.5*(C + C.T) + 1e-12*np.eye(C.shape[0])
    cf = cho_factor(C, check_finite=False)

    CinvX   = cho_solve(cf, X,   check_finite=False)
    CinvRhs = cho_solve(cf, rhs, check_finite=False)
    XtCinvX = X.T @ CinvX
    XtCinvR = X.T @ CinvRhs

    pars = np.linalg.solve(XtCinvX, XtCinvR)   # [M, gamma, beta2]

    mu   = mu0 + X @ pars
    res  = rhs - X @ pars
    chi2 = res @ cho_solve(cf, res, check_finite=False)
    dof  = len(z) - 3

    cov  = np.linalg.inv(XtCinvX)
    cov *= (chi2 / dof)   # optional, conservative

    # safe sqrt for numerical noise
    def s(x): return float(np.sqrt(x)) if x >= 0 else 0.0

    out = dict(
        M=float(pars[0]), gamma=float(pars[1]), beta2=float(pars[2]),
        M_err=s(cov[0,0]), gamma_err=s(cov[1,1]), beta2_err=s(cov[2,2]),
        chi2=float(chi2), dof=int(dof),
        mu_model=mu, mu0=mu0, A=A, B2=B2)

    return out

def fit_M_beta_given_delta(z, mb_corr, C, delta_fixed, a, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0):
    z = np.asarray(z, float)
    L = np.log1p(z)

    # μ0 with δ fixed
    mu0 = mu0_umh_nonexp(z, a=a, s=s, b=b, c_=c_, d0=d0, kappa=kappa) + 2.5*np.log10(1.0+z)*delta_fixed

    # Design matrix for [M, β1, β2]
    B1 = 2.5*LOG10E * L
    B2 = 2.5*LOG10E * (L**2)
    X  = np.column_stack([np.ones_like(z), B1, B2])
    y  = np.asarray(mb_corr, float)

    C = 0.5*(C + C.T) + 1e-12*np.eye(C.shape[0])
    cf = cho_factor(C, check_finite=False)

    CinvX = cho_solve(cf, X, check_finite=False)
    CinvR = cho_solve(cf, y - mu0, check_finite=False)
    XtCinvX = X.T @ CinvX
    XtCinvR = X.T @ CinvR
    pars = np.linalg.solve(XtCinvX, XtCinvR)      # [M, β1, β2]

    mu  = mu0 + X @ pars
    res = y - mu
    chi2 = float(res @ cho_solve(cf, res, check_finite=False))
    dof  = len(z) - 3
    
    cov  = np.linalg.inv(XtCinvX)
    cov *= (chi2 / dof)   # optional, conservative

    return dict(M=float(pars[0]),
        beta1=float(pars[1]), beta2=float(pars[2]),
        M_err=float(np.sqrt(max(cov[0,0],0.0))),
        beta1_err=float(np.sqrt(max(cov[1,1],0.0))),
        beta2_err=float(np.sqrt(max(cov[2,2],0.0))),
        chi2=chi2, dof=dof, mu_model=mu)


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides: config.update(config_overrides)
    
    #c = 299792.458  # speed of light in km/s
    c_kms=config["LIGHT_SPEED"]

    vpec_kms=config["VPEC"]
    
    #H0 = 70  # Hubble constant in km/s/Mpc
    H0=config["H0"]

    columns=config["PANTHEON_DATA_COLUMNS"]
    panfile=config["PANTHEON_DATA_FILE"]

    panfile_bias=config["PANTHEON_DATA_BIAS_FILE"]

    GENERATE_UMH_SIMULATION_CALIBRATION=config["GENERATE_UMH_SIMULATION_CALIBRATION"]

    USE_HUBER=config.get("USE_HUBER",True)

    dpi=config["DPI"]

    outdir = config["OUTPUT_FOLDER"]
    
    file_root="UMH_RedShift"

    title="UMH RedShift"
    file_hdr="UMH_RedShift"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")

    # Load Pantheon+
    df = pd.read_csv(panfile, comment="#")
    print(f"[load] Pantheon+ rows: {len(df)}")
    print(df.head(3))

    # ===== Low-z calibrators: estimate a =====
    d_cal, sig_d, z_cal, sig_L, z_err = build_calibrator_vectors(df, c_kms,
                                                          zmax=0.10, dmax=120.0,
                                                          sigma_ceph_default=0.20,
                                                          vpec_kms=vpec_kms)
    L_cal = np.log1p(z_cal)

    if(USE_HUBER):
        # weight by distance errors (conservative); can also use 1/sig_L^2
        L_cal = np.log1p(z_cal)                       # z_cal from low-z selection (z<=0.10)
        w_meas = 1.0/np.maximum(sig_d, 1e-9)**2       # weight by distance errors

        a_hat, b0, sigma_a, info = fit_alpha_huber(d_cal, L_cal, w_meas=w_meas, huber_c=2.0, force_through_zero=False) # robust IRLS, same c≈2 as Pantheon+
    else:
        # use σ_L-weighted fit (with robust clip built-in)
        def wfit_alpha(d, z, sigL):
            ln1pz = np.log1p(z)
            return fit_alpha_weighted(d, ln1pz, z, sigL, zmax=0.10, clip_sigma=3.0, force_through_zero=False)
        a_hat, b0, sigma_a, info = wfit_alpha(d_cal, z_cal, sig_L)

    print(f"[calib] a = {a_hat:.6e} ± {sigma_a:.2e}  1/Mpc  (N={d_cal.size})")


    # ===== Diagnostic plots in the (d,z) and L-spaces =====
    # Plot: z(d) for calibrators with UMH vs Hubble
    d_grid = np.linspace(0, max(d_cal)*1.05 if len(d_cal) else 150.0, 400)
    z_umh  = np.expm1(a_hat * d_grid)       # pure a-law for display
    
    H0_ref = c_kms * a_hat
    H0_err = c_kms * sigma_a
    z_hub  = (H0_ref/c_kms) * d_grid  # = a_hat * d_grid

    
    # ========= UMH vs Hubble plot suite (Pantheon+ calibrators) =========
    cap = "(green/orange curves are diagnostics with β fixed; preferred model is δ=1 with profiled β)"


    plt.figure(figsize=(9,6))

    sigma_z = np.sqrt(z_err**2 + (vpec_kms/c_kms)**2)
    plt.errorbar(d_cal, z_cal, xerr=sig_d, yerr=sigma_z, fmt="o", ms=4, alpha=0.7, label="Calibrators (Pantheon+)")

    plt.plot(d_grid, z_umh, lw=2, label=fr"UMH: $z=\exp(a d)-1$, $a={a_hat:.3e}\,$Mpc^{{-1}}$")
    plt.plot(d_grid, z_hub, lw=2, ls="--", label=f"Hubble: z≈(H0/c)d, H0={H0_ref:.1f} km/s/Mpc")
    plt.xlabel("Comoving distance d [Mpc]"); plt.ylabel("Redshift z")
    plt.title(f"{title}: tension-driven z(d) vs linear Hubble")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    plt.tight_layout(); 
    plt.savefig(f"{file_path}_Calibrators_z_vs_Distance.png", dpi=dpi); plt.close()

    # Plot: L ≡ ln(1+z) vs d with ~95% CI for L = a d
    L_umh  = a_hat * d_grid
    band_L = 1.96 * sigma_a * d_grid


    # --- Preferred: plot in fit space d vs L (polished) ---
    # Pull fit params & provenance
    c0 = float(info.get("c0", 0.0))
    c1 = float(info.get("c1", 1.0 / a_hat))
    N = int(info.get("N", len(L_cal)))
    chi2_dof = float(info.get("chi2_dof", np.nan))
    # Grid and fitted line
    L_grid = np.linspace(np.min(L_cal), np.max(L_cal), 400)
    d_fit  = c0 + c1 * L_grid
    # Use covariance if available; otherwise fall back to slope-only uncertainty
    cov00 = float(info.get("cov_c00", 0.0))
    cov01 = float(info.get("cov_c01", 0.0))
    cov11 = float(info.get("cov_c11", info.get("sigma_c1", 0.0)**2))
    sigma_d = np.sqrt(np.maximum(cov00 + 2*cov01*L_grid + cov11*L_grid**2, 0.0))
    band_d  = 1.96 * sigma_d  # ≈95% CI

    # --- Plot in fit space: d vs L (with-intercept), using OO API throughout ---
    fig, ax = plt.subplots(figsize=(9, 6))
    # Fit and CI FIRST → legend order: Fit → 95% CI → Calibrators
    line_fit, = ax.plot(L_grid, d_fit, lw=2.2, label=r"Fit: $d=c_0+c_1 L$")
    band = ax.fill_between(L_grid, d_fit - band_d, d_fit + band_d, alpha=0.15, label="≈95% CI")
    # Calibrators last
    ax.errorbar(L_cal, d_cal, xerr=sig_L, yerr=sig_d, fmt="o", ms=4, alpha=0.75, label="Calibrators (Pantheon+)")
    # Titles (split across lines)
    fig.suptitle(rf"{title}: low-$z$ calibration", y=0.98)
    ax.set_title(rf"$\alpha={a_hat:.3e}\pm{sigma_a:.1e}$ Mpc$^{{-1}}$  ↔  "
        rf"$H_0={H0_ref:.1f}\pm{H0_err:.1f}$ km s$^{{-1}}$ Mpc$^{{-1}}$"
        rf"\n(c0={c0:+.3f} Mpc; N={N}; $\chi^2$/dof={chi2_dof:.3f})",
        fontsize=10)
    # Labels, grid, legend
    ax.set_xlabel(r"$L \equiv \ln(1+z)$")
    ax.set_ylabel(r"Distance $d$ (Mpc)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    fig.savefig(f"{file_path}_ln1pz_vs_Distance.png", dpi=dpi)
    plt.close(fig)


    # ===== SN-only Hubble diagram under UMH (non-expansion) =====
    mask_sn = (df["IS_CALIBRATOR"].astype(int) == 0)
    z_sn    = df.loc[mask_sn, "zHD"].to_numpy(float)
    mb_corr = df.loc[mask_sn, "m_b_corr"].to_numpy(float)

    # Load full STAT+SYS covariance, subset to SN mask
    if os.path.exists(panfile_bias):
        with open(panfile_bias, "rt") as f:
            N = int(f.readline().strip())
            vals = np.loadtxt(f)
        C_full = vals.reshape((N, N))
        idx = np.flatnonzero(mask_sn.to_numpy() if hasattr(mask_sn, "to_numpy") else mask_sn)
        Csel = C_full[np.ix_(idx, idx)]
        # jitter for numerical stability
        Csel = 0.5*(Csel+Csel.T) + 1e-12*np.eye(Csel.shape[0])
    else:
        # fall back to diagonal from m_b_corr_err_DIAG
        sig = df.loc[mask_sn, "m_b_corr_err_DIAG"].to_numpy(float)
        Csel = np.diag(sig**2)


    # --- Fit identifiable combo gamma=δ+β1 (and β2)
    res_g = fit_M_gamma_beta2(z_sn, mb_corr, Csel,
        a=a_hat, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0)
    print(f"[γ,β2]  γ = {res_g['gamma']:.3f} ± {res_g['gamma_err']:.3f}, "
          f"β2 = {res_g['beta2']:.3f} ± {res_g['beta2_err']:.3f}, "
          f"M = {res_g['M']:.3f}, χ²/dof = {res_g['chi2']/res_g['dof']:.3f}")

    # Two interpretations to plot:
    # (a) 'δ free, β=0'   ->  δ = γ, β1=0, β2=0

    mu_delta_only = res_g['mu0'] + res_g['A']*res_g['gamma']
    chi2_do, M_do = chi2_and_M_best(mb_corr, mu_delta_only, Csel)
    mu_delta_only = mu_delta_only + M_do


    # (b) 'δ=1, β profiled' -> β1 = γ - 1, β2 = fitted
    beta1 = res_g['gamma'] - 1.0
    L     = np.log1p(z_sn)
    B1    = 2.5/np.log(10.0) * L
    mu_beta = res_g['mu0'] + res_g['A']*1.0 + B1*beta1 + res_g['B2']*res_g['beta2'] + res_g['M']

    # --- Plot
    sort = np.argsort(z_sn)
    plt.figure(figsize=(9,6))
    plt.scatter(z_sn, mb_corr, s=9, alpha=0.8, label="Pantheon+ SNe (SN-only)")
    plt.plot(z_sn[sort], mu_delta_only[sort], lw=2, label="UMH (δ free, β=0)")
    plt.plot(z_sn[sort], mu_beta[sort],      lw=2, ls="--", label="UMH (δ=1, β profiled)")
    plt.xlabel("Redshift z"); plt.ylabel("Distance Modulus μ")
    plt.title("UMH RedShift: Pantheon+ Hubble diagram — δ vs profiled β (no expansion)")
    plt.figtext(0.5, 0.01, cap, ha="center", fontsize=9)

    plt.grid(True, alpha=0.3); plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0)); plt.tight_layout()
    out_png = f"{file_path}_Hubble_UMH_delta_vs_betas.png"
    plt.savefig(out_png, dpi=dpi); plt.close()


    # UMH μ(z): start with simplest a-law (s=b=c=0), delta=1, T=1
    mu_umh = mu_umh_of_z_nonexp(z_sn, a=a_hat, s=0.0, b=0.0, c_=0.0, d0=1.0, delta=1.0, kappa=1.0, T_of_z=None)
    chi2_1, M1 = chi2_and_M_best(mb_corr, mu_umh, Csel)
    dof_1 = max(len(z_sn)-1,1)
    print(f"[UMH μ(z)] delta=1.0:  chi2/dof = {chi2_1/dof_1:.3f},  profiled M = {M1:.3f}")

    # Hubble diagram plot (UMH only, with profiled M applied)
    idx = np.argsort(z_sn)
    plt.figure(figsize=(9,6))
    plt.scatter(z_sn, mb_corr, s=9, alpha=0.8, label="Pantheon+ SNe (SN-only)")
    plt.plot(z_sn[idx], (mu_umh + M1)[idx], lw=2.0, color="tab:green", label="UMH (non-expansion)")
    plt.xlabel("Redshift z"); plt.ylabel("Distance Modulus μ")
    plt.title(f"{title}: Pantheon+ Hubble diagram, with time dilation (δ=1)")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0)); plt.tight_layout(); 
    plt.savefig(f"{file_path}_Hubble_UMH_delta1.png", dpi=dpi); plt.close()

    # ===== Time-dilation scan: vary delta in μ(z) and compute chi2 =====
    deltas = np.linspace(0.6, 1.4, 33)  # wide-ish scan around 1
    chi2s  = []
    Ms     = []
    for dlt in deltas:
        mu_try = mu_umh_of_z_nonexp(z_sn, a=a_hat, s=0.0, b=0.0, c_=0.0, d0=1.0, delta=dlt, kappa=1.0, T_of_z=None)
        chi2, Mbest = chi2_and_M_best(mb_corr, mu_try, Csel)
        chi2s.append(chi2); Ms.append(Mbest)
    chi2s = np.asarray(chi2s); Ms = np.asarray(Ms)
    jbest = int(np.argmin(chi2s))

    res_beta = fit_M_beta_given_delta(z_sn, mb_corr, Csel, delta_fixed=1.0, a=a_hat)
    print(f"[δ=1] β1={res_beta['beta1']:.3f}±{res_beta['beta1_err']:.3f}, "
          f"β2={res_beta['beta2']:.3f}±{res_beta['beta2_err']:.3f}, "
          f"χ²/dof={res_beta['chi2']/res_beta['dof']:.3f}")

    # χ²/dof for δ=1 with β profiled
    chi2dof_prof = res_beta["chi2"] / res_beta["dof"]

    plt.figure(figsize=(8,5))
    # existing scan (β=0)
    plt.plot(deltas, chi2s/float(dof_1), lw=2, label=r"β fixed (0)")
    # mark δ = 1
    plt.axvline(1.0, color="k", ls=":", lw=1)
    # add the Option-1 result as a dot at δ=1
    plt.scatter([1.0], [chi2dof_prof], s=70, zorder=3, label=r"δ=1, β profiled")
    plt.annotate(fr"{chi2dof_prof:.3f}", xy=(1.0, chi2dof_prof),
                 xytext=(1.02, chi2dof_prof+0.01), arrowprops=dict(arrowstyle="-", lw=0.8))
    plt.xlabel(r"Time-dilation exponent $\delta$ in $D_L \propto (1+z)^{(1+\delta)/2}$")
    plt.ylabel(r"$\chi^2/\mathrm{dof}$")
    plt.title(f"{title}: δ-scan (β=0) with δ=1, β profiled overlay")
    plt.legend(loc="upper right"); plt.tight_layout(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{file_path}_delta_scan_beta_profiled_vs_fixed.png", dpi=dpi); plt.close()


    plt.figure(figsize=(8,5))
    # δ-scan curve (β=0)
    plt.plot(deltas, chi2s/float(dof_1), lw=2, label=r"β fixed (0)")
    # vertical marker at best δ
    plt.axvline(deltas[jbest], color="k", ls="--", lw=1, label=rf"best $\delta$ = {deltas[jbest]:.2f}")
    plt.xlabel(r"Time-dilation exponent $\delta$ in $D_L \propto (1+z)^{(1+\delta)/2}$")
    plt.ylabel(r"$\chi^2/\mathrm{dof}$")
    plt.title(f"{title}: χ² vs δ (diagnostic; β fixed to 0)")
    plt.grid(True, alpha=0.3); plt.legend(loc="upper right"); plt.tight_layout();
    plt.savefig(f"{file_path}_delta_scan_chi2.png", dpi=dpi); plt.close()

    print(f"[delta-scan] best δ ≈ {deltas[jbest]:.3f} with χ2/dof ≈ {chi2s[jbest]/dof_1:.3f}")


    # Pick δ values to illustrate
    # δ=1 is the UMH/physical time-dilation expectation.
    deltas_to_plot = [(1.0, "UMH expectation (δ=1)")]

    # Show the δ that best fits when β=0:
    delta_best_scan = float(deltas[jbest])
    deltas_to_plot.append((delta_best_scan, rf"best δ with β=0 ({delta_best_scan:.2f})"))

    # Solved for γ=δ+β1 to show the δ-equivalent with β1=0:
    delta_equiv = float(res_g["gamma"])
    deltas_to_plot.append((delta_equiv, rf"δ-equivalent from γ (β1=0): {delta_equiv:.2f}"))

    # --- build μ for (a) δ free, β=0  (use best δ from β=0 scan)
    delta_best_scan = float(deltas[int(jbest)])
    A = 2.5*np.log10(1.0 + z_sn)
    mu0 = mu0_umh_nonexp(z_sn, a=a_hat)                    # δ=0, T=1
    mu_delta = mu0 + A*delta_best_scan
    _, M_do = chi2_and_M_best(mb_corr, mu_delta, Csel)     # profile M
    mu_delta += M_do

    # --- (b) δ=1, β profiled (already solved)
    mu_beta = res_beta["mu_model"]

    # --- residual vectors
    res_a = mb_corr - mu_delta
    res_b = mb_corr - mu_beta

    # running median (windowed) for the profiled-β case
    order = np.argsort(z_sn)
    zs, rs = z_sn[order], (mb_corr - res_beta["mu_model"])[order]

    s_rs = pd.Series(rs)
    med = s_rs.rolling(window=75, center=True, min_periods=30).median().to_numpy()
    zmid = zs

    plt.figure(figsize=(10,5.8))
    plt.scatter(z_sn, mb_corr - (mu0 + A*delta_best_scan + M_do), s=10, alpha=0.55, label=r"δ free, β=0 (best δ)")
    plt.scatter(z_sn, rs, s=10, alpha=0.55, label=r"δ=1, β profiled")
    plt.plot(zmid, med, lw=2, label="running median (δ=1, β profiled)")
    plt.axhline(0, color="k", lw=1)
    plt.ylim(-0.6, 0.6)                    # symmetric limits read cleaner
    plt.xlabel("Redshift z"); plt.ylabel(r"Residual $\mu_{\rm data}-\mu_{\rm model}$ (mag)")
    plt.title(f"{title}: Hubble-diagram Residuals vs z")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_mu_vs_z.png", dpi=dpi); plt.close()


    # unweighted mean/std (M profiling makes the mean ~0)
    mu_res = float(np.mean(res_b)); sigma_res = float(np.std(res_b, ddof=1))
    sk = float(skew(res_b)); ku = float(kurtosis(res_b, fisher=True))
    N = len(res_b)

    plt.figure(figsize=(9,5.4))
    n, bins, _ = plt.hist(res_b, bins=40, density=True, alpha=0.45, label=f"Residuals (δ=1, β profiled), N={N}")
    x = np.linspace(bins[0], bins[-1], 400)
    plt.plot(x, (1/(sigma_res*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu_res)/sigma_res)**2),
             lw=2, label=fr"Normal fit: $\mu={mu_res:.3f}$, $\sigma={sigma_res:.3f}$")
    plt.xlabel(r"Residual $\mu_{\rm data}-\mu_{\rm model}$ (mag)")
    plt.ylabel("Density")
    plt.title(f"{title}: Residual Distribution (skew={sk:.2f}, kurtosis={ku:.2f})")
    plt.grid(True, alpha=0.3); plt.legend(loc="upper left"); plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_Hist_beta_profiled.png", dpi=dpi); plt.close()



    def bin_equal_count(x, y, nbins=16):
        x = np.asarray(x); y = np.asarray(y)
        order = np.argsort(x); x, y = x[order], y[order]
        edges = np.linspace(0, len(x), nbins+1, dtype=int)
        xc, med, ylo, yhi, n = [], [], [], [], []
        for i in range(nbins):
            sl = slice(edges[i], edges[i+1])
            if edges[i+1]-edges[i] < 3: continue
            xs, ys = x[sl], y[sl]
            xc.append(xs.mean())
            med.append(np.median(ys))
            q16, q84 = np.percentile(ys, [16, 84])
            ylo.append(np.median(ys)-q16)
            yhi.append(q84-np.median(ys))
            n.append(len(ys))
        return map(np.asarray, (xc, med, ylo, yhi, n))

    xc, med, ylo, yhi, nbin = bin_equal_count(z_sn, res_b, nbins=14)

    plt.figure(figsize=(10,5.8))
    plt.errorbar(xc, med, yerr=[ylo, yhi], fmt="o", ms=5, capsize=2, label=r"median $\pm$68% (δ=1, β profiled)")
    ax = plt.gca()
    for x, n in zip(xc, nbin):
        ax.text(x, 0.005, f"{n}", ha="center", va="bottom", fontsize=8, color="0.4", alpha=0.6)   # smaller, grey, semi-transparent

    plt.axhline(0, color="k", lw=1)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Redshift z"); plt.ylabel(r"Residual $\mu_{\rm data}-\mu_{\rm model}$ (mag)")
    plt.title(f"{title}: Binned Residuals vs z (equal-N bins)")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_Binned_beta_profiled.png", dpi=dpi); plt.close()

    
    zmax = float(np.nanmax(z_sn))
    z = np.linspace(0.0, zmax, 400)

    plt.figure(figsize=(8,5))
    for dlt, lbl in deltas_to_plot:
        F = (1.0 + z)**(0.5*float(dlt))
        plt.plot(z, F, lw=2, label=lbl or rf"δ = {dlt:.2f}")

    F_GR = (1.0 + z)**0.5               # GR expectation inside μ
    plt.plot(z, F_GR, ls="--", lw=2, color="k", label=r"GR / SN Ia: $(1+z)^{1/2}$")
    plt.text(0.05, 0.02, "δ=1 overlaps GR (dashed)", transform=plt.gca().transAxes, fontsize=9)

    ax = plt.gca()
    ax.axvline(zmax, ls=":", lw=1, color="k", alpha=0.4)
    ax.text(zmax, ax.get_ylim()[0] + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            rf"$z_\mathrm{{max}}\approx{zmax:.2f}$",
            ha="left", va="bottom", fontsize=9, color="k", alpha=0.7, rotation=0)

    plt.xlabel("Redshift z")
    plt.ylabel(r"Time-dilation factor in $\mu$: $(1+z)^{\delta/2}$")
    plt.title(r"UMH RedShift: $\mu$ time-dilation factor vs redshift")
    plt.figtext(0.5, 0.001, cap, ha="center", fontsize=9)
    plt.grid(True, alpha=0.3); plt.legend(loc="upper left");
    plt.savefig(f"{file_path}_mu_timedilation_factor.png", dpi=dpi); plt.close()

    
    plt.figure(figsize=(8,5))
    for dlt, lbl in deltas_to_plot:
        S = (1.0 + z)**(float(dlt))
        plt.plot(z, S, lw=2, label=lbl or rf"δ = {dlt:.2f}")

    # --- GR / real-life stretch: S(z) = 1 + z  (dashed black)
    S_GR = 1.0 + z
    plt.plot(z, S_GR, ls="--", lw=2, color="k", label="GR / SN Ia: S(z)=1+z")
    plt.text(0.05, 0.02, "δ=1 overlaps GR (dashed)", transform=plt.gca().transAxes, fontsize=9)

    ax = plt.gca()
    ax.axvline(zmax, ls=":", lw=1, color="k", alpha=0.4)
    ax.text(zmax, ax.get_ylim()[0] + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            rf"$z_\mathrm{{max}}\approx{zmax:.2f}$",
            ha="left", va="bottom", fontsize=9, color="k", alpha=0.7, rotation=0)

    plt.xlabel("Redshift z")
    plt.ylabel(r"Physical time-stretch $S(z)=(1+z)^{\delta}$")

    plt.title("UMH RedShift: physical time-stretch")
    plt.figtext(0.5, 0.01, cap, ha="center", fontsize=9)

    plt.tight_layout(); plt.grid(True, alpha=0.3); plt.legend(loc="upper left")
    
    plt.savefig(f"{file_path}_Time_Stretch.png", dpi=dpi); plt.close()

    # ========= end plot suite =========


    if GENERATE_UMH_SIMULATION_CALIBRATION:  # Used by UMH_Simulation_Sphere.py
        MPC_IN_M = 3.085677581e22

        # ---------- helpers ----------
        def distance_modulus_to_DL_Mpc(mu):
            """Luminosity distance D_L [Mpc] from distance modulus mu = m - M."""
            return 10.0**((np.asarray(mu) - 25.0)/5.0)

        def build_redshift_dataset(z, d_Mpc=None, mb=None, M=None, use_comoving=True,
            z_err=None, d_err=None, vpec_kms=250.0, c_kms=299792.458):
            """
            Return arrays for calibration:
              - z, ln1pz, sigma_L (error on ln(1+z)), d_Mpc (+ optional d_err)
            If d_Mpc is None, must have (mb, M) to compute D_L and then d.
            Uncertainty model:
              sigma_L^2 = (z_err/(1+z))^2 + (vpec/c)^2    (peculiar-velocity floor)
            """
            z = np.asarray(z, float)

            # distance
            if d_Mpc is None:
                if mb is None or M is None: raise ValueError("Provide d_Mpc or (mb and M).")
                mu = np.asarray(mb, float) - float(M)
                DL = distance_modulus_to_DL_Mpc(mu)  # Mpc
                d_Mpc = DL/(1.0+z) if use_comoving else DL
            else: d_Mpc = np.asarray(d_Mpc, float)

            # optional errors
            if z_err is None: z_err = np.zeros_like(z)
            else: z_err = np.asarray(z_err, float)

            if d_err is None: d_err = np.zeros_like(d_Mpc)
            else: d_err = np.asarray(d_err, float)

            ok = np.isfinite(z) & np.isfinite(d_Mpc) & (z > 0.0) & (d_Mpc > 0.0)
            z, d_Mpc, z_err, d_err = z[ok], d_Mpc[ok], z_err[ok], d_err[ok]

            ln1pz = np.log1p(z)
            sigma_L = np.sqrt((z_err/np.maximum(1.0+z, 1e-30))**2 + (vpec_kms/c_kms)**2)

            return z, d_Mpc, d_err, ln1pz, sigma_L


        def save_redshift_calibration(file_path, H0, z, z_err, d_Mpc, d_err, ln1pz, sigma_L,
            alpha, alpha_err, intercept, info, res_beta):

            csv_path  = f"{file_path}_Calibration_Data.csv"
            json_path = f"{file_path}_Calibration_Fit.json"

            header = "z,z_err,d_Mpc,d_err,ln1pz,sigma_L"
            data_mat = np.vstack([z, z_err, d_Mpc, d_err, ln1pz, sigma_L]).T
            np.savetxt(csv_path, data_mat, delimiter=",", header=header, comments="")

            payload = {
                "model": "UMH non-expansion",
                "delta_fixed": 1.0,

                "alpha_1_per_Mpc":  float(alpha),
                "alpha_err_1_per_Mpc": float(alpha_err),
                "alpha_1_per_m":    float(alpha / MPC_IN_M),
                "intercept":        float(intercept),
                "H0":               float(H0),
                "H0_km_s_Mpc":      float(c_kms * alpha),
                "meta":             info,
                "units": {
                    "z": "dimensionless",
                    "z_err": "dimensionless",
                    "d_Mpc": "Mpc", "d_err": "Mpc",
                    "ln1pz": "dimensionless",
                    "sigma_L": "dimensionless"
                },
                "peculiar_velocity_floor_kms": float(vpec_kms),

                # profiled M and betas (with errors)
                "M_best": float(res_beta["M"]),
                "M_err": float(res_beta.get("M_err", np.nan)),
                "beta1": float(res_beta["beta1"]),
                "beta1_err": float(res_beta["beta1_err"]),
                "beta2": float(res_beta["beta2"]),
                "beta2_err": float(res_beta["beta2_err"]),

                # goodness of fit
                "chi2": float(res_beta["chi2"]),
                "dof": int(res_beta["dof"]),
                "chi2_dof": float(res_beta["chi2"] / res_beta["dof"]),
            }

            payload["data"] = {
                "z":      z.astype(np.float32).tolist(),
                "z_err":  z_err.astype(np.float32).tolist(),
                "d_Mpc":  d_Mpc.astype(np.float32).tolist(),
                "d_err":  d_err.astype(np.float32).tolist(),
                "ln1pz":  ln1pz.astype(np.float32).tolist(),
                "sigma_L":sigma_L.astype(np.float32).tolist(),
                "dtype":  "float32"
            }

            with open(json_path, "w") as f: json.dump(payload, f, indent=2)

            return csv_path, json_path


        # ---------- use the calibrator arrays already built ----------
        # (these come from build_calibrator_vectors)
        z      = z_cal.astype(float)         # redshift
        d_Mpc  = d_cal.astype(float)         # distance (Mpc)
        d_err  = sig_d.astype(float)         # distance uncertainty (Mpc)
        ln1pz  = np.log1p(z)                 # L ≡ ln(1+z)

        sigma_L = np.sqrt((z_err/np.maximum(1.0+z,1e-30))**2 + (vpec_kms/c_kms)**2)

        # ---------- fit α on low-z ----------
        alpha, alpha_err = float(a_hat), float(sigma_a)


        # ---------- save CSV + JSON ----------
        csv_path, json_path = save_redshift_calibration(
            file_path, H0,
            z       = z,
            z_err   = z_err.astype(float),
            d_Mpc   = d_Mpc,
            d_err   = d_err,
            ln1pz   = ln1pz,
            sigma_L = sigma_L,
            alpha = alpha,
            alpha_err     = alpha_err,
            intercept     = b0,
            info          = info,
            res_beta = res_beta)

        if(USE_HUBER):
            print(f"[CAL] α = {a_hat:.6e} ± {sigma_a:.2e} 1/Mpc "
                  f"({a_hat/MPC_IN_M:.3e} 1/m) | H0 = {c_kms*a_hat:.2f} km/s/Mpc "
                  f"| N={info['N']}  χ²/dof={info['chi2_dof']:.3f}  "
                  f"c0={info['c0']:+.3f} Mpc  huber_c={info['huber_c']:.2f}  "
                  f"through_origin={info['force_through_zero']}")
        else:
            print(f"[CAL] α = {alpha:.6e} ± {alpha_err:.2e} 1/Mpc  "
                  f"({alpha/MPC_IN_M:.3e} 1/m) | H0 = {c_kms*alpha:.1f} km/s/Mpc  "
                  f"| N={info['N_fit']}  χ²/dof={info['chi2_dof']:.3f}  R²={info['R2_unweighted']:.3f}")
        print("Saved:", csv_path, json_path)


    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)