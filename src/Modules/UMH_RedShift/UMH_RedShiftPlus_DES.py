"""
UMH_RedShiftPlus_DES.py (UMH DES RedShift Test)

RedShift under Ultronic Medium using DES-SN5YR/Dovekie

Author: Andrew Dodge
Date: July 2025

Implements the DES-SN5YR/Dovekie redshift/time-dilation validation of the
Ultronic Medium Hypothesis (UMH).

Description:
  Tests the same non-expansion UMH redshift/time-dilation formulation against
  the DES-SN5YR/Dovekie Hubble diagram. The cosmological redshift scale alpha
  is fixed from the Pantheon+/Cepheid low-z calibration. DES is then used as
  an independent transmission-coefficient recovery/transfer test.
"""

import numpy as np
import os
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing matplotlib
import sys
import json

import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from scipy.stats import skew, kurtosis



def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.

        "LIGHT_SPEED": 299792.458,  # speed of light in km/s

        "USE_HUBER": True,

        "VPEC": 200, # km/s

        "H0": 70, # Hubble constant in km/s/Mpc

        # Independent low-z calibration carried into the DES validation.
        "UMH_ALPHA_FIXED": 2.481805e-04,      # 1/Mpc, Pantheon+/Cepheid calibrated
        "UMH_ALPHA_SIGMA_FIXED": 8.0e-06,     # 1/Mpc, rounded published uncertainty

        # Which transmission coefficients define the fixed-coefficient DES diagnostic.
        # "des_recovered" = recover beta1,beta2 from DES, then hold them fixed.
        # "pantheon_fixed" = transfer the Pantheon+ values directly into DES.
        "BETA_SOURCE": "pantheon_fixed",
        "PANTHEON_BETA1_FIXED": 0.432,
        "PANTHEON_BETA2_FIXED": -0.270,

        "DES_SN5YR_DATA_COLUMNS": ['CID', 'IDSURVEY', 'zHD', 'zHEL', 'MU', 'MUERR', 'MUERR_VPEC', 'MUERR_SYS', 'PROBIA_BEAMS'],
        "DES_SN5YR_DATA_FILE":os.path.join(base, "Output", "PantheonData", "DES-Dovekie_HD.csv"),

        "DES_SN5YR_DATA_BIAS_FILE":os.path.join(base, "Output", "PantheonData", "DES_SN5YR_STAT_SYS.npz"),
        "DES_SN5YR_DATA_BIAS_SO_FILE":os.path.join(base, "Output", "PantheonData", "DES_SN5YR_STAT_ONLY.npz"),

        "GENERATE_UMH_SIMULATION_CALIBRATION": False, # DES uses external Pantheon+/Cepheid alpha calibration.

        "DPI":300, #PNG Resolution.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def load_des_covariance_from_npz(npz_path):
    """
    DES-Dovekie provides inverse covariance matrices in .npz format. This loads and unpacks the inverse covariance and returns it directly.
    """
    data = np.load(npz_path)
    print(f"[cov] Loaded {npz_path}")
    print(f"[cov] NPZ keys: {list(data.keys())}")
    if "nsn" not in data.files or "cov" not in data.files:
        raise RuntimeError(f"Expected keys 'nsn' and 'cov' in {npz_path}, but found {list(data.files)}")
    nsn = int(np.asarray(data["nsn"]).item()); cov_arr = np.asarray(data["cov"], dtype=float)
    print(f"[cov] nsn={nsn}"); print(f"[cov] raw cov shape={cov_arr.shape}, ndim={cov_arr.ndim}")
    if cov_arr.ndim == 2:
        if cov_arr.shape != (nsn, nsn): raise RuntimeError(f"cov key is 2D but shape is {cov_arr.shape}; expected {(nsn, nsn)}")
        Cinv = cov_arr
    elif cov_arr.ndim == 1:
        if cov_arr.size == nsn * nsn: Cinv = cov_arr.reshape((nsn, nsn))
        elif cov_arr.size == nsn * (nsn + 1) // 2:
            Cinv = np.zeros((nsn, nsn), dtype=float)
            # DES stores the packed triangular matrix in upper-triangle order.
            tri = np.triu_indices(nsn)
            Cinv[tri] = cov_arr
            Cinv = Cinv + Cinv.T - np.diag(np.diag(Cinv))
            diag = np.diag(Cinv)
            print(f"[cov] unpacked diag range: {np.min(diag):.3e} to {np.max(diag):.3e}")
            if np.any(diag <= 0):
                raise RuntimeError("Unpacked DES inverse covariance has non-positive diagonal entries. "
                                   "This means the triangular unpacking order is still wrong; check the DES likelihood loader.")
        else:
            raise RuntimeError(f"cov key is 1D with size {cov_arr.size}; expected either {nsn * nsn} full matrix values or "
                               f"{nsn * (nsn + 1) // 2} packed-triangle values")
    else: raise RuntimeError(f"cov key has unsupported ndim={cov_arr.ndim}, shape={cov_arr.shape}")
    print(f"[cov] Using 'cov' as inverse covariance with shape {Cinv.shape}")
    Cinv = 0.5 * (Cinv + Cinv.T)
    print(f"[cov] final Cinv diag range: {np.min(np.diag(Cinv)):.3e} to {np.max(np.diag(Cinv)):.3e}")
    print(f"[cov] Cinv finite: {np.isfinite(Cinv).all()}")
    # Existing code expects covariance C, not inverse covariance Cinv.
    #C = np.linalg.inv(Cinv)
    #C = 0.5 * (C + C.T)
    #C += 1e-12 * np.eye(C.shape[0])
    #return C
    # DES provides inverse covariance / precision matrix.
    # Return Cinv directly and use direct quadratic forms downstream.
    Cinv = 0.5 * (Cinv + Cinv.T)
    Cinv += 1e-14 * np.eye(Cinv.shape[0])
    return Cinv


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

def chi2_and_M_best(data_vec, model_mu, Cinv):
    """
    Profile M analytically using the DES inverse covariance / precision matrix.
    Returns (chi2, M_best).
    """
    one = np.ones_like(data_vec)
    r = np.asarray(data_vec, float) - np.asarray(model_mu, float)
    Cinv_r = Cinv @ r
    Cinv_one = Cinv @ one
    M_best = float((one @ Cinv_r) / (one @ Cinv_one))
    Delta = r - M_best
    chi2 = float(Delta @ (Cinv @ Delta))
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

def fit_M_gamma_beta2(z, mb_corr, Cinv, a, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0):
    """
    GLS fit of [M, gamma(=delta+beta1), beta2] using the DES precision matrix.
    Removes the delta-beta1 collinearity.
    """
    z = np.asarray(z, float)
    L = np.log1p(z)

    A = 2.5*np.log10(1.0+z)
    B2 = 2.5*LOG10E * (L**2)
    X = np.column_stack([np.ones_like(z), A, B2])

    mu0 = mu0_umh_nonexp(z, a=a, s=s, b=b, c_=c_, d0=d0, kappa=kappa)
    rhs = np.asarray(mb_corr, float) - mu0

    Cinv = 0.5*(Cinv + Cinv.T)
    XtCinvX = X.T @ Cinv @ X
    XtCinvR = X.T @ Cinv @ rhs
    pars = np.linalg.solve(XtCinvX, XtCinvR)

    mu = mu0 + X @ pars
    res = np.asarray(mb_corr, float) - mu
    chi2 = float(res @ Cinv @ res)
    dof = len(z) - 3

    cov = np.linalg.inv(XtCinvX)
    cov *= (chi2 / dof)

    def s_err(x): return float(np.sqrt(x)) if x >= 0 else 0.0

    return dict(
        M=float(pars[0]), gamma=float(pars[1]), beta2=float(pars[2]),
        M_err=s_err(cov[0,0]), gamma_err=s_err(cov[1,1]), beta2_err=s_err(cov[2,2]),
        chi2=chi2, dof=int(dof), mu_model=mu, mu0=mu0, A=A, B2=B2)


def fit_M_beta_given_delta(z, mb_corr, Cinv, delta_fixed, a,
        s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0,
        huber_c=2.0, max_iter=30):
    """
    Robust precision-matrix GLS recovery of [M, beta1, beta2] at fixed delta.

    Used only for the DES transmission-calibration/recovery diagnostic.
    The recovered coefficients are then held fixed before any conditional
    final-likelihood evaluation.
    """
    z = np.asarray(z, float)
    L = np.log1p(z)

    mu0 = mu0_umh_nonexp(
        z, a=a, s=s, b=b, c_=c_, d0=d0, kappa=kappa
    ) + 2.5*np.log10(1.0+z)*delta_fixed

    B1 = 2.5*LOG10E * L
    B2 = 2.5*LOG10E * (L**2)
    X = np.column_stack([np.ones_like(z), B1, B2])
    y = np.asarray(mb_corr, float) - mu0

    Cinv = 0.5*(Cinv + Cinv.T)
    Lp = np.linalg.cholesky(Cinv)
    K = Lp.T

    X0 = K @ X
    y0 = K @ y

    w = np.ones_like(y0)
    pars = None
    cov_unscaled = None

    for _ in range(max_iter):
        sw = np.sqrt(w)
        Xw = X0 * sw[:, None]
        yw = y0 * sw

        ATA = Xw.T @ Xw
        ATy = Xw.T @ yw
        try:
            cov_unscaled = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            cov_unscaled = np.linalg.pinv(ATA)

        pars_new = cov_unscaled @ ATy
        r_white = y0 - X0 @ pars_new

        scale = max(
            1.4826 * np.median(np.abs(r_white - np.median(r_white))),
            1e-6
        )
        u = r_white / scale
        w_new = np.where(
            np.abs(u) <= huber_c,
            1.0,
            huber_c / np.abs(u)
        )

        if pars is not None and np.allclose(
            pars_new, pars, rtol=1e-7, atol=1e-9
        ):
            pars = pars_new
            w = w_new
            break

        pars = pars_new
        w = w_new

    mu = mu0 + X @ pars
    res = np.asarray(mb_corr, float) - mu
    chi2 = float(res @ Cinv @ res)
    dof = len(z) - 3

    cov = cov_unscaled * (chi2 / dof)

    return dict(
        M=float(pars[0]),
        beta1=float(pars[1]),
        beta2=float(pars[2]),
        M_err=float(np.sqrt(max(cov[0,0], 0.0))),
        beta1_err=float(np.sqrt(max(cov[1,1], 0.0))),
        beta2_err=float(np.sqrt(max(cov[2,2], 0.0))),
        chi2=chi2,
        dof=dof,
        mu_model=mu,
        recovery_method="robust_precision_huber_irls",
        huber_c=float(huber_c)
    )


def marginal_precision_from_full_precision(Cinv, keep_mask):
    """
    Return the marginal precision matrix for a subset selected from a full
    precision matrix. A plain submatrix of a precision matrix is conditional,
    not marginal, so use the Schur complement when points are removed.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.all():
        return Cinv.copy()

    keep = np.flatnonzero(keep_mask)
    drop = np.flatnonzero(~keep_mask)

    Paa = Cinv[np.ix_(keep, keep)]
    Pab = Cinv[np.ix_(keep, drop)]
    Pba = Cinv[np.ix_(drop, keep)]
    Pbb = Cinv[np.ix_(drop, drop)]

    correction = Pab @ np.linalg.solve(Pbb, Pba)
    Pmarg = Paa - correction
    return 0.5 * (Pmarg + Pmarg.T)


def dL_lcdm_flat_grid(z_grid, H0=70.0, Omega_m=0.333, c_kms=299792.458):
    """
    Flat ΛCDM luminosity distance in Mpc on a sorted redshift grid. Integrates from z=0, matching the Pantheon+ comparison code.
    """
    z_grid = np.asarray(z_grid, float)
    if np.any(np.diff(z_grid) < 0): raise ValueError("z_grid must be sorted in ascending order.")
    # Prepend z=0 so the cumulative integral starts from the origin.
    z_aug = np.concatenate(([0.0], z_grid))
    Ez = np.sqrt(Omega_m * (1.0 + z_aug)**3 + (1.0 - Omega_m))
    inv_E = 1.0 / Ez
    integ = np.zeros_like(z_aug)
    dz = np.diff(z_aug)
    integ[1:] = np.cumsum(0.5 * dz * (inv_E[1:] + inv_E[:-1]))
    dC_aug = (c_kms / H0) * integ; dL_aug = (1.0 + z_aug) * dC_aug
    return dL_aug[1:]


def dL_umh_nonexp_grid(z_grid, a, beta1, beta2, delta=1.0):
    """
    UMH luminosity distance in Mpc for the preferred non-expansion model.
    """
    z_grid = np.asarray(z_grid, float); d_vals = np.log1p(z_grid) / a; Tvals = make_Texp(beta1, beta2)(z_grid)
    return d_vals * (1.0 + z_grid)**((1.0 + delta) / 2.0) / np.sqrt(Tvals)


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    c_kms = config["LIGHT_SPEED"]
    dpi = config["DPI"]

    desfile = config["DES_SN5YR_DATA_FILE"]
    des_cov_file = config["DES_SN5YR_DATA_BIAS_FILE"]

    outdir = config["OUTPUT_FOLDER"]
    file_root = "UMH_RedShift"
    title = "UMH RedShift DES-SN5YR"
    file_hdr = "UMH_RedShift_DES"

    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir = os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path = os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")

    # ------------------------------------------------------------------
    # DES data and precision matrix
    # ------------------------------------------------------------------
    df = pd.read_csv(desfile)
    print(f"Loaded {len(df)} DES-Dovekie supernovae...")
    print(df.head(3))

    required = {"CID", "IDSURVEY", "zHD", "MU", "MUERR"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"DES-Dovekie HD file is missing required columns: {missing}")

    z_sn = df["zHD"].to_numpy(float)
    mb_corr = df["MU"].to_numpy(float)
    err_plot = np.clip(df["MUERR"].to_numpy(float), 0, 1.0)

    Cinv = load_des_covariance_from_npz(des_cov_file)
    if Cinv.shape != (len(z_sn), len(z_sn)):
        raise RuntimeError(
            f"Precision/data length mismatch: Cinv is {Cinv.shape}, "
            f"but DES HD has N={len(z_sn)}."
        )

    # ------------------------------------------------------------------
    # Alpha is NOT recalibrated from DES.  It is carried in from the
    # independent Pantheon+/Cepheid low-z calibration used in the paper.
    # ------------------------------------------------------------------
    a_hat = float(config["UMH_ALPHA_FIXED"])
    sigma_a = float(config.get("UMH_ALPHA_SIGMA_FIXED", np.nan))
    H0_ref = c_kms * a_hat
    H0_err = c_kms * sigma_a if np.isfinite(sigma_a) else np.nan

    print(f"[fixed alpha] alpha = {a_hat:.6e} 1/Mpc "
          f"(sigma={sigma_a:.2e}); H0 reference = {H0_ref:.2f} km/s/Mpc")

    cap = "(DES diagnostic: beta=0 versus delta=1; public reproduction uses Pantheon+ beta values transferred unchanged)"

    # ------------------------------------------------------------------
    # Identifiable gamma diagnostic: delta + beta1
    # ------------------------------------------------------------------
    res_g = fit_M_gamma_beta2(z_sn, mb_corr, Cinv,
        a=a_hat, s=0.0, b=0.0, c_=0.0, d0=1.0, kappa=1.0)

    print(f"[gamma,beta2] gamma = {res_g['gamma']:.3f} ± {res_g['gamma_err']:.3f}, "
          f"beta2 = {res_g['beta2']:.3f} ± {res_g['beta2_err']:.3f}, "
          f"basis intercept = {res_g['M']:.3f}, chi2/dof = {res_g['chi2']/res_g['dof']:.3f}")

    # Diagnostic A: beta=0, allow delta-equivalent gamma to carry curvature.
    mu_delta_only = res_g['mu0'] + res_g['A'] * res_g['gamma']
    chi2_do, M_do = chi2_and_M_best(mb_corr, mu_delta_only, Cinv)
    mu_delta_only = mu_delta_only + M_do

    # Diagnostic B: delta=1, recover beta1 and beta2 from DES.
    res_beta = fit_M_beta_given_delta(
        z_sn, mb_corr, Cinv, delta_fixed=1.0, a=a_hat)
    beta1_des = float(res_beta["beta1"])
    beta2_des = float(res_beta["beta2"])

    print(f"[DES robust transmission recovery] beta1={beta1_des:.4f}±{res_beta['beta1_err']:.4f}, "
          f"beta2={beta2_des:.4f}±{res_beta['beta2_err']:.4f}, "
          f"chi2/dof={res_beta['chi2']/res_beta['dof']:.4f}")

    # ------------------------------------------------------------------
    # Fixed-coefficient final diagnostic.
    # Select either coefficients recovered from DES or the Pantheon+
    # coefficients transferred into DES without refitting.
    # ------------------------------------------------------------------
    beta_source = str(config.get("BETA_SOURCE", "pantheon_fixed")).strip().lower()

    if beta_source == "des_recovered":
        beta1_pref, beta2_pref = beta1_des, beta2_des
        beta_source_label = "DES robust-recovered"
    elif beta_source == "pantheon_fixed":
        beta1_pref = float(config["PANTHEON_BETA1_FIXED"])
        beta2_pref = float(config["PANTHEON_BETA2_FIXED"])
        beta_source_label = "Pantheon+ transferred (publication reproduction)"
    else:
        raise ValueError("BETA_SOURCE must be 'des_recovered' or 'pantheon_fixed'.")

    T_pref = make_Texp(beta1_pref, beta2_pref)
    mu_pref_noM = mu_umh_of_z_nonexp(
        z_sn, a=a_hat, delta=1.0, T_of_z=T_pref)
    chi2_pref, M_pref = chi2_and_M_best(mb_corr, mu_pref_noM, Cinv)
    mu_pref = mu_pref_noM + M_pref
    dof_pref = max(len(z_sn) - 1, 1)

    print(f"[conditional final DES diagnostic: {beta_source_label}] "
          f"beta1={beta1_pref:.4f}, beta2={beta2_pref:.4f}; "
          f"M={M_pref:.5f}; chi2={chi2_pref:.2f}; "
          f"dof={dof_pref}; chi2/dof={chi2_pref/dof_pref:.4f}")

    # Always report the direct Pantheon+ -> DES transfer as an independent check.
    beta1_pan = float(config["PANTHEON_BETA1_FIXED"])
    beta2_pan = float(config["PANTHEON_BETA2_FIXED"])
    T_pan = make_Texp(beta1_pan, beta2_pan)
    mu_pan_noM = mu_umh_of_z_nonexp(
        z_sn, a=a_hat, delta=1.0, T_of_z=T_pan)
    chi2_pan, M_pan = chi2_and_M_best(mb_corr, mu_pan_noM, Cinv)

    print(f"[Pantheon+ -> DES fixed-beta transfer] "
          f"beta1={beta1_pan:.4f}, beta2={beta2_pan:.4f}; "
          f"M={M_pan:.5f}; chi2={chi2_pan:.2f}; "
          f"chi2/dof={chi2_pan/dof_pref:.4f}")

    # ------------------------------------------------------------------
    # Hubble diagram: beta=0 diagnostic vs delta=1 recovered-beta model.
    # ------------------------------------------------------------------
    sort = np.argsort(z_sn)
    plt.figure(figsize=(9,6))
    plt.scatter(z_sn, mb_corr, s=9, alpha=0.65, label="DES-SN5YR/Dovekie")
    plt.plot(z_sn[sort], mu_delta_only[sort], lw=2,
             label="UMH diagnostic (delta free-equivalent, beta=0)")
    plt.plot(z_sn[sort], res_beta["mu_model"][sort], lw=2, ls="--",
             label="UMH diagnostic (delta=1, beta recovered from DES)")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance Modulus mu")
    plt.title("UMH RedShift DES: beta=0 vs recovered transmission (no expansion)")
    plt.figtext(0.5, 0.01, cap, ha="center", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Hubble_delta_vs_betas.png", dpi=dpi)
    plt.close()

    # Conditional final fixed-coefficient display.
    plt.figure(figsize=(9,6))
    plt.scatter(z_sn, mb_corr, s=9, alpha=0.65, label="DES-SN5YR/Dovekie")
    plt.plot(z_sn[sort], mu_pref[sort], lw=2,
             label=f"UMH delta=1 ({beta_source_label} beta)")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance Modulus mu")
    plt.title(f"{title}: fixed-coefficient conditional final diagnostic")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Hubble_fixed_beta.png", dpi=dpi)
    plt.close()

    # ------------------------------------------------------------------
    # Delta scan with beta=0.
    # ------------------------------------------------------------------
    deltas = np.linspace(0.6, 1.4, 33)
    chi2s = []
    Ms = []
    for dlt in deltas:
        mu_try = mu_umh_of_z_nonexp(
            z_sn, a=a_hat, delta=dlt, T_of_z=None)
        chi2, Mbest = chi2_and_M_best(mb_corr, mu_try, Cinv)
        chi2s.append(chi2)
        Ms.append(Mbest)

    chi2s = np.asarray(chi2s)
    Ms = np.asarray(Ms)
    jbest = int(np.argmin(chi2s))
    dof_delta = max(len(z_sn) - 1, 1)

    print(f"[delta scan, beta=0] best delta ≈ {deltas[jbest]:.3f}; "
          f"chi2/dof ≈ {chi2s[jbest]/dof_delta:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(deltas, chi2s/float(dof_delta), lw=2, label="beta=0")
    plt.axvline(deltas[jbest], ls="--", lw=1,
                label=f"best delta={deltas[jbest]:.2f}")
    plt.scatter([1.0], [chi2_pref/dof_pref], s=70, zorder=3,
                label=f"delta=1, fixed beta ({beta_source_label})")
    plt.xlabel("Time-dilation exponent delta")
    plt.ylabel("chi2/dof")
    plt.title(f"{title}: delta scan and fixed-beta comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_delta_scan.png", dpi=dpi)
    plt.close()

    # ------------------------------------------------------------------
    # Residual diagnostics for the selected fixed-coefficient model.
    # ------------------------------------------------------------------
    residuals = mb_corr - mu_pref

    plt.figure(figsize=(10,5.8))
    plt.scatter(z_sn, residuals, s=10, alpha=0.55,
                label=f"delta=1, fixed beta ({beta_source_label})")
    order = np.argsort(z_sn)
    zs = z_sn[order]
    rs = residuals[order]
    med = pd.Series(rs).rolling(
        window=75, center=True, min_periods=30).median().to_numpy()
    plt.plot(zs, med, lw=2, label="running median")
    plt.axhline(0, lw=1)
    plt.xlabel("Redshift z")
    plt.ylabel("Residual mu_data - mu_model (mag)")
    plt.title(f"{title}: residuals vs z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_mu_vs_z.png", dpi=dpi)
    plt.close()

    mu_res = float(np.mean(residuals))
    sigma_res = float(np.std(residuals, ddof=1))
    sk = float(skew(residuals))
    ku = float(kurtosis(residuals, fisher=True))

    plt.figure(figsize=(9,5.4))
    n, bins, _ = plt.hist(
        residuals, bins=40, density=True, alpha=0.45,
        label=f"Residuals, N={len(residuals)}")
    x = np.linspace(bins[0], bins[-1], 400)
    plt.plot(
        x,
        (1/(sigma_res*np.sqrt(2*np.pi))) *
        np.exp(-0.5*((x-mu_res)/sigma_res)**2),
        lw=2,
        label=f"Normal fit: mean={mu_res:.3f}, sigma={sigma_res:.3f}")
    plt.xlabel("Residual mu_data - mu_model (mag)")
    plt.ylabel("Density")
    plt.title(f"{title}: residual distribution (skew={sk:.2f}, kurtosis={ku:.2f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_Hist.png", dpi=dpi)
    plt.close()

    # ------------------------------------------------------------------
    # Precision-whitened residual check.
    #
    # DES supplies the inverse covariance / precision matrix P = C^{-1}.
    # If P = L L^T (Cholesky), then w = L^T r satisfies
    #     w^T w = r^T P r = chi^2.
    # These whitened residuals, rather than the raw magnitude residuals,
    # are the appropriate quantities to compare with an N(0,1)-like shape.
    # ------------------------------------------------------------------
    P = 0.5 * (Cinv + Cinv.T)
    Lp = np.linalg.cholesky(P)
    whitened_residuals = Lp.T @ residuals

    white_mean = float(np.mean(whitened_residuals))
    white_std = float(np.std(whitened_residuals, ddof=1))
    white_skew = float(skew(whitened_residuals))
    white_kurt = float(kurtosis(whitened_residuals, fisher=True))
    white_norm2 = float(whitened_residuals @ whitened_residuals)
    white_chi2_diff = float(white_norm2 - chi2_pref)

    print(
        f"[whitened residuals] mean={white_mean:.4f}, "
        f"std={white_std:.4f}, skew={white_skew:.4f}, "
        f"excess_kurtosis={white_kurt:.4f}"
    )
    print(
        f"[whitened residuals] w^T w={white_norm2:.6f}, "
        f"chi2={chi2_pref:.6f}, difference={white_chi2_diff:.3e}"
    )

    plt.figure(figsize=(9,5.4))
    n_w, bins_w, _ = plt.hist(
        whitened_residuals,
        bins=40,
        density=True,
        alpha=0.45,
        label=f"Whitened residuals, N={len(whitened_residuals)}"
    )
    x_w = np.linspace(bins_w[0], bins_w[-1], 400)
    standard_normal = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * x_w**2)
    plt.plot(
        x_w,
        standard_normal,
        lw=2,
        label="Standard normal N(0,1)"
    )
    plt.axvline(0.0, lw=1, ls="--")
    plt.xlabel("Precision-whitened residual")
    plt.ylabel("Density")
    plt.title(
        f"{title}: whitened residuals "
        f"(mean={white_mean:.3f}, σ={white_std:.3f}, "
        f"skew={white_skew:.2f}, kurt={white_kurt:.2f})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_WhitenedResiduals.png", dpi=dpi)
    plt.close()

    # ------------------------------------------------------------------
    # High-z validation.  Because DES supplies a precision matrix, use the
    # marginal precision (Schur complement), not a plain precision submatrix.
    # ------------------------------------------------------------------
    print("\n[DES high-z validation] fixed alpha, beta1, beta2, delta=1; profile only M")
    highz_rows = []
    for zcut in (0.10, 0.15, 0.20):
        keep = z_sn > zcut
        z_cut = z_sn[keep]
        mb_cut = mb_corr[keep]
        Cinv_cut = marginal_precision_from_full_precision(Cinv, keep)

        mu_cut = mu_umh_of_z_nonexp(
            z_cut, a=a_hat, delta=1.0, T_of_z=T_pref)
        chi2_cut, M_cut = chi2_and_M_best(mb_cut, mu_cut, Cinv_cut)
        dof_cut = max(len(z_cut) - 1, 1)
        chi2_dof_cut = chi2_cut / dof_cut
        highz_rows.append(
            (zcut, len(z_cut), chi2_cut, dof_cut, chi2_dof_cut, M_cut))
        print(f" z > {zcut:.2f}: N={len(z_cut)}, chi2={chi2_cut:.2f}, "
              f"dof={dof_cut}, chi2/dof={chi2_dof_cut:.4f}, M={M_cut:.4f}")

    highz_path = f"{file_path}_HighZ_Validation.csv"
    np.savetxt(
        highz_path,
        np.array(highz_rows, dtype=float),
        delimiter=",",
        header="z_cut,N,chi2,dof,chi2_dof,M_profiled",
        comments="")
    print(f"[DES high-z validation] saved: {highz_path}")

    # ------------------------------------------------------------------
    # Time-dilation guide plots. These display the model-implied scaling;
    # they are not an independent light-curve stretch fit.
    # ------------------------------------------------------------------
    zmax = float(np.nanmax(z_sn))
    z_plot = np.linspace(0.0, zmax, 400)
    delta_best_scan = float(deltas[jbest])
    delta_equiv = float(res_g["gamma"])

    deltas_to_plot = [
        (1.0, "UMH expectation (delta=1)"),
        (delta_best_scan, f"best delta with beta=0 ({delta_best_scan:.2f})"),
        (delta_equiv, f"delta-equivalent from gamma ({delta_equiv:.2f})")
    ]

    plt.figure(figsize=(8,5))
    for dlt, lbl in deltas_to_plot:
        S = (1.0 + z_plot)**float(dlt)
        plt.plot(z_plot, S, lw=2, label=lbl)
    plt.plot(z_plot, 1.0 + z_plot, ls="--", lw=2,
             label="standard guide: S(z)=1+z")
    plt.xlabel("Redshift z")
    plt.ylabel("Model time-stretch S(z)")
    plt.title(f"{title}: model-implied time-stretch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Time_Stretch.png", dpi=dpi)
    plt.close()

    # ------------------------------------------------------------------
    # UMH vs flat LCDM fractional luminosity-distance difference using
    # the DES comparison Omega_m value.
    # ------------------------------------------------------------------
    z_cmp = np.linspace(0.001, np.nanmax(z_sn), 1000)
    dL_umh = dL_umh_nonexp_grid(
        z_cmp, a=a_hat, beta1=beta1_pref, beta2=beta2_pref, delta=1.0)
    dL_lcdm = dL_lcdm_flat_grid(
        z_cmp, H0=c_kms*a_hat, Omega_m=0.329, c_kms=c_kms)
    frac_diff = (dL_umh - dL_lcdm) / dL_lcdm
    delta_mu = 5.0 * np.log10(dL_umh / dL_lcdm)

    frac_path = f"{file_path}_UMH_LCDM_Fractional_Difference.csv"
    np.savetxt(
        frac_path,
        np.column_stack([z_cmp, dL_umh, dL_lcdm, frac_diff, delta_mu]),
        delimiter=",",
        header="z,dL_UMH_Mpc,dL_LCDM_Mpc,frac_diff,delta_mu_mag",
        comments="")

    print(f"[UMH vs LCDM] saved: {frac_path}")

    # ------------------------------------------------------------------
    # Save a compact machine-readable summary.
    # ------------------------------------------------------------------
    summary_path = f"{file_path}_Summary.json"
    summary = {
        "dataset": "DES-SN5YR/Dovekie",
        "N": int(len(z_sn)),
        "alpha_1_per_Mpc": a_hat,
        "alpha_sigma_1_per_Mpc": sigma_a,
        "alpha_source": "Pantheon+/Cepheid low-z calibration",
        "beta_source_selected": beta_source,
        "publication_reproduction_default": bool(beta_source == "pantheon_fixed"),
        "des_beta_recovery_method": res_beta.get("recovery_method", "unknown"),
        "des_beta_recovery_huber_c": res_beta.get("huber_c", None),
        "beta1_des_recovered": beta1_des,
        "beta2_des_recovered": beta2_des,
        "beta1_pantheon_fixed": beta1_pan,
        "beta2_pantheon_fixed": beta2_pan,
        "beta1_selected": beta1_pref,
        "beta2_selected": beta2_pref,
        "chi2_selected": chi2_pref,
        "dof_selected": dof_pref,
        "M_selected": M_pref,
        "chi2_pantheon_transfer": chi2_pan,
        "M_pantheon_transfer": M_pan,
        "delta_beta0_best": float(deltas[jbest]),
        "chi2_delta_beta0_best": float(chi2s[jbest]),
        "whitened_residual_mean": white_mean,
        "whitened_residual_std": white_std,
        "whitened_residual_skew": white_skew,
        "whitened_residual_excess_kurtosis": white_kurt,
        "whitened_residual_norm2": white_norm2,
        "whitened_residual_chi2_difference": white_chi2_diff
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", summary_path)
    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)