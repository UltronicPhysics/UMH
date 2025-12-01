"""
UMH_vs_PantheonPlus.py  (UMH vs Pantheon Test)

Pantheon+ Hubble-diagram comparison of UMH (non-expansion) vs flat ΛCDM.

Author: Andrew Dodge
Date: July 2025

Implements the analysis described in:
  A. Dodge, "Pantheon+ and Redshift Validation of the Ultronic Medium Hypothesis (UMH)",
  July 2025, and Appendix A.2.7–A.2.8 of "The Ultronic Medium Hypothesis (UMH)".

Key features:
  - Uses Pantheon+ SN-only sample (N=1624) with full STAT+SYS covariance.
  - One-time low-z calibration of α from Cepheid-anchored calibrators.
  - UMH non-expansion μ(z) with theory-fixed (α, β1, β2), profiling only M.
  - Flat ΛCDM reference fit with free Ωm and profiled M.
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
from matplotlib.patches import Patch

from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.linalg import cho_factor, cho_solve

from mpmath import gammainc, gamma
from scipy.stats import binned_statistic

from scipy.optimize import brentq  # for inverting z(d) -> d(z)


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
       
        "LIGHT_SPEED": 299792.458,  # speed of light in km/s

        "H0": 70, # Hubble constant in km/s/Mpc

        "PANTHEON_DATA_COLUMNS":["CID","IDSURVEY","zHD","zHDERR","zCMB","zCMBERR","zHEL","zHELERR","m_b_corr","m_b_corr_err_DIAG","MU_SH0ES","MU_SH0ES_ERR_DIAG","CEPH_DIST","IS_CALIBRATOR","USED_IN_SH0ES_HF","c","cERR","x1","x1ERR","mB","mBERR","x0","x0ERR","COV_x1_c","COV_x1_x0","COV_c_x0","RA","DEC","HOST_RA","HOST_DEC","HOST_ANGSEP","VPEC","VPECERR","MWEBV","HOST_LOGMASS","HOST_LOGMASS_ERR","PKMJD","PKMJDERR","NDOF","FITCHI2","FITPROB","m_b_corr_err_RAW","m_b_corr_err_VPEC","biasCor_m_b","biasCorErr_m_b","biasCor_m_b_COVSCALE","biasCor_m_b_COVADD"],
        "PANTHEON_DATA_FILE":os.path.join(base, "Output", "PantheonData", "PantheonPlus_SH0ES.dat"),
        
        "PANTHEON_DATA_BIAS_FILE":os.path.join(base, "Output", "PantheonData", "PantheonPlus_SH0ES_STAT_SYS.cov"),

        "DPI":300, #PNG Resolution.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }



def run(config_overrides=None):
    config = get_default_config()
    if config_overrides: config.update(config_overrides)
    
    #c = 299792.458  # speed of light in km/s
    c=config["LIGHT_SPEED"]
    
    #H0 = 70  # Hubble constant in km/s/Mpc
    H0=config["H0"]

    columns=config["PANTHEON_DATA_COLUMNS"]
    panfile=config["PANTHEON_DATA_FILE"]

    panfile_bias=config["PANTHEON_DATA_BIAS_FILE"]

    dpi=config["DPI"]

    outdir = config["OUTPUT_FOLDER"]

    file_root="UMH_vs_Pantheon"

    title="UMH vs Pantheon+"
    file_hdr="UMH_vs_PantheonPlus"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")

    # UMH Settings
    # low-z UMH calibration from calibrators (SH0ES/ceph) ----
    DO_LOWZ_CALIB = True
    ZMAX_CALIB    = 0.10

    DO_PREWHITEN = True

    DO_CALC_BETAS_FROM_DATA = True
    
    # UMH redshift law: ln(1+z) = (a d + b d^2 + c ln(1 + d/d0)) / (1 + s d)
    # Used in Mpc, d0=1 Mpc so the log is dimensionless.
    d0_ref = 1.0  # Mpc
    delta_td = 1.0

    # --- Constants for ΛCDM ---
    Omega_m = 0.3
    Omega_L = 1.0 - Omega_m

    # --- Load the table (Pantheon+ .dat is comma-separated and has its own header) ---
    df = pd.read_csv(panfile, comment='#')  # don't pass header=None/names=...

    print(f"Loaded {len(df)} supernovae...")
    print(df.head(3))

    # --- SN-only mask (exclude Cepheid calibrators) ---
    mask = df["IS_CALIBRATOR"].astype(int) == 0
    idx  = np.flatnonzero(mask)

    # --- Extract cosmology columns ---
    z = df.loc[mask, "zHD"].to_numpy()          # use zHD column from Pantheon+
    len_z = len(z)
    mb_corr  = df.loc[mask, "m_b_corr"].to_numpy()     # corrected apparent magnitude (data vector)
    mb_corr_err_diag = df.loc[mask, "m_b_corr_err_DIAG"].to_numpy()

    ids_subset = df.loc[mask, "IDSURVEY"].to_numpy()
    ids_subset = np.where(np.isfinite(ids_subset), ids_subset, -1).astype(int)

    # --- Load the STAT+SYS covariance (first line N, then N^2 row-major floats) ---
    with open(panfile_bias, "rt") as f:
        N = int(f.readline().strip())
        vals = np.loadtxt(f)
    C = vals.reshape((N, N))

    # --- Subset covariance with the SAME indices as the .dat mask ---
    Csel = C[np.ix_(idx, idx)]
    Csel = 0.5 * (Csel + Csel.T)  # enforce symmetry
    err_plot = np.sqrt(np.diag(Csel))  # for plotting only

    # Add tiny jitter for numerical stability in Cholesky/solves
    Csel += 1e-12 * np.eye(Csel.shape[0])

    # ----------------- UMH (non-expansion) MODEL + χ² WITH PROFILED M -----------------

    
    def z_of_d_umh(d, a, s=0.0, b=0.0, c_=0.0, d0=1.0):
        d = float(d)
        num = a*d + b*(d**2) + c_*np.log1p(d/d0)
        den = 1.0 + s*d
        if abs(den) < 1e-12: return np.inf  # avoid division-by-zero near the pole
        x = num/den
        if x > 700.0: return np.inf
        if x < -50.0: return 0.0
        return np.expm1(x)


    # --- Invert z(d) -> d(z) robustly ------------------------------------------
    def d_of_z_umh(z_target, a, s=0.0, b=0.0, c_=0.0, d0=d0_ref, d_init=1.0, d_max=1e9, max_doublings=100):
        zt = float(z_target)
        if zt <= 0.0: return 0.0

        # target L = ln(1+z), scalar
        L = float(np.log1p(zt))

        # ----- Analytic inversion for s=0, c=0 -----
        if np.isclose(s, 0.0, atol=1e-12) and np.isclose(c_, 0.0, atol=1e-15):
            if np.isclose(b, 0.0, atol=0.0):
                if a <= 0.0: raise RuntimeError("UMH: a must be > 0 for s=0,b=0.")
                return L / a
            else:
                disc = a*a + 4.0*b*L          # scalar
                if disc < 0.0: raise RuntimeError("UMH: no real d solves a d + b d^2 = ln(1+z).")
                sq = float(np.sqrt(disc))
                # two roots; pick the positive one that best satisfies the equation
                r1 = (-a + sq) / (2.0*b)
                r2 = (-a - sq) / (2.0*b)
                cand = [r for r in (r1, r2) if r > 0.0]
                if not cand: raise RuntimeError("UMH: positive root not found for given (a,b).")
                return min(cand, key=lambda d: abs(a*d + b*d*d - L))

        # ----- Guard when b<=0 and s>0: z has a finite ceiling -----
        if b <= 0.0 and s > 0.0:
            z_inf = np.exp(a / s) - 1.0
            if zt >= 0.999 * z_inf:
                raise RuntimeError(
                    f"UMH: requested z={zt:.3g} exceeds model's max z≈{z_inf:.3g} (a/s={a/s:.3f}). "
                    "Increase a/s or allow b>0.")

        # function for root-finder: scalar
        def f(d): return z_of_d_umh(d, a=a, s=s, b=b, c_=c_, d0=d0) - zt

        # ----- Special handling when s ~ 0 (but c_ != 0) -----
        if np.isclose(s, 0.0, atol=1e-12):
            hi_candidates = [max(10.0, d_init)]
            if a > 0.0: hi_candidates.append(L / a)
            if b > 0.0: hi_candidates.append(np.sqrt(L / b))
            if c_ > 0.0: hi_candidates.append(d0 * np.expm1(L / c_))
            hi = float(max(hi_candidates))
            lo = 0.0
            n = 0
            while f(hi) <= 0.0 and hi < d_max and n < max_doublings:
                hi *= 2.0
                n += 1
            if f(hi) <= 0.0: raise RuntimeError("UMH: could not bracket d(z) with s≈0; increase a or b (or c), or raise d_max.")

            return brentq(f, lo, hi, xtol=1e-10, maxiter=200)

        # ----- General case (s not ~ 0) -----
        lo, hi = 0.0, max(1.0, float(d_init))
        d_pole = (-1.0/s) if (s < 0.0) else None
        def f(d): return z_of_d_umh(d, a=a, s=s, b=b, c_=c_, d0=d0) - zt

        # grow hi safely but never cross the pole
        fhi = f(hi); n = 0
        while fhi <= 0.0 and hi < d_max and n < max_doublings:
            hi *= 2.0
            if d_pole is not None and hi >= 0.99*d_pole:
                hi = 0.99*d_pole
                break
            fhi = f(hi); n += 1

        if d_pole is not None and hi >= 0.99*d_pole and f(hi) <= 0.0:
            raise RuntimeError("UMH: could not bracket d(z) without crossing the s<0 pole.")
        if fhi <= 0.0:
            raise RuntimeError("UMH: could not bracket d(z); try larger a/s or enable b>0.")

        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)


    # --- Distance modulus for UMH non-expansion family ---------------------------
    def mu_umh_of_z_nonexp(z_array, a, s=0.0, b=0.0, c_=0.0, d0=1.0, delta=1.0, kappa=1.0, T_of_z=None):
        """
        z_array : array of redshifts
        a,s,b,c_,d0 : UMH redshift-law parameters (ln(1+z) = (a d + b d^2 + c ln(1+d/d0)) / (1 + s d))
        delta   : observed time-dilation exponent (keep delta=1 to match SN data)
        kappa   : optional scale converting UMH depth d -> metric distance r = kappa*d (Mpc)
        T_of_z  : optional transmission function (defaults to 1)
        """
        # make the default transmission vectorized
        if T_of_z is None: T_of_z = lambda zz: np.ones_like(np.asarray(zz), dtype=float)
        
        z_array = np.asarray(z_array, float)

        # Analytic inversion for the pure "one-parameter" law: s=0, b=0, c=0.
        if np.isclose(s, 0.0, atol=1e-12) and b == 0.0 and c_ == 0.0:
            if a <= 0.0: raise RuntimeError("UMH: a must be > 0 for the pure s=0, b=c=0 case.")
            d_vals = np.log1p(z_array) / a
        else:
            # General case: invert with the robust bracket+brent used elsewhere
            d_vals = np.array([d_of_z_umh(zi, a=a, s=s, b=b, c_=c_, d0=d0) for zi in z_array])

        # --- vectorized transmission + luminosity distance ---
        Tvals = np.asarray(T_of_z(z_array))
        if Tvals.ndim == 0: Tvals = np.full_like(z_array, Tvals, dtype=float)   # allow scalar-returning T_of_z

        D_L = (kappa * d_vals) * (1.0 + z_array)**((1.0 + delta)/2.0) / np.sqrt(Tvals)

        return 5.0*np.log10(D_L) + 25.0


    def chi2_and_M_best(data_vec, model_mu, C):
        """
        data_vec: m_b_corr (observed corrected magnitudes)
        model_mu: model distance modulus at the same z (no M)
        C: subselected STAT+SYS covariance (same ordering)
        """
        one = np.ones_like(data_vec)
        cf = cho_factor(C, overwrite_a=False, check_finite=False)
        Cinvd   = cho_solve(cf, data_vec - model_mu, check_finite=False)
        Cinvone = cho_solve(cf, one,       check_finite=False)
        M_best  = (one @ Cinvd) / (one @ Cinvone)     # analytic profiled M
        Delta   = data_vec - model_mu - M_best
        chi2    = Delta @ cho_solve(cf, Delta, check_finite=False)
        return chi2, M_best


    def make_Texp_umh_theory(beta1, beta2=0.0):
        """T(z) = exp[- τ(L) ], τ(L)=β1*L + β2*L^2,  with L=ln(1+z)."""
        def T_of_z(z):
            L = np.log1p(np.asarray(z, float))
            tau = beta1*L + beta2*(L**2)
            return np.exp(-tau)
        return T_of_z

    # --- UMH microphysics reporter (choose a length; Planck by default) ---
    def print_umh_microphysics(a, beta1, beta2, ell_int_mpc=None, label="Planck"):
        # defaults: Planck length
        if ell_int_mpc is None: ell_int_mpc = 1.616255e-35 / 3.085677581491367e22  # ℓ_P in Mpc

        eps_red = a * ell_int_mpc                # per-interaction redshift increment
        p_loss  = beta1 * eps_red                # per-interaction loss prob
        dpdL    = 2.0 * beta2 * eps_red          # L-derivative of p_loss (micro-evolution)
        alpha_eff = a * beta1                    # dτ/dd
        mfp = np.inf if alpha_eff <= 0 else 1.0/alpha_eff

        print(f"[UMH microphysics @ {label}] ℓ_int = {ell_int_mpc:.3e} Mpc")
        print(f"  ε_red per interaction  = {eps_red:.3e}")
        print(f"  p_loss per interaction = {p_loss:.3e}")
        print(f"  d p_loss / dL          = {dpdL:.3e}")
        print(f"  dτ/dd = a*β1 = {alpha_eff:.3e} 1/Mpc  → mean free path ≈ {mfp:.1f} Mpc")


    def lowz_alpha_from_calibrators(
        df,
        z_col="zHD",
        mu_col="MU_SH0ES",
        mu_err_col="MU_SH0ES_ERR_DIAG",
        ceph_col="CEPH_DIST",
        zmax=0.10,
        drop_ceph_if_mu=True,
        ceph_rel_sigma=0.08,     # ~0.16 mag → ~7.4–8% distance error
        ceph_abs_floor=0.2,      # ≥0.2 Mpc absolute floor for CEPH-only distances
        use_robust=True,
        huber_c=1.345,
        force_through_zero=False # if True, enforce d = c1 * L (c0=0)
    ):
        """
        Return (alpha, sigma_alpha, summary_dict).

        - Builds a calibrator sample at z <= zmax.
        - If MU_SH0ES is present, uses distance d from μ with propagated σ_d.
        - If only CEPH_DIST is present, uses it with an inflated uncertainty:
            σ_d = max(ceph_rel_sigma * d, ceph_abs_floor).
        - If drop_ceph_if_mu is True, rows with both μ and CEPH keep μ and drop cepheid.
        - Optionally runs a robust Huber IRLS fit to reduce outlier leverage.
        - Model: d = c0 + c1 * L, with L=ln(1+z); alpha = 1/c1.

        Notes:
          Robust step rescales residuals by MAD and downweights |u|>c, u=r/s.
          When force_through_zero=True, c0 is fixed to 0 and alpha = 1/c1
          with closed-form WLS solution.
        """
        # --------------- selection ---------------
        mask = (df.get("IS_CALIBRATOR", 0).astype(int) == 1)
        mask &= np.isfinite(df[z_col])
        mask &= (df[z_col] <= zmax)

        z_all = df.loc[mask, z_col].to_numpy()
        L_all = np.log1p(z_all)

        has_mu   = mask & np.isfinite(df[mu_col])
        has_ceph = mask & np.isfinite(df[ceph_col])

        # distance from μ (if present)
        use_mu = has_mu.copy()
        d_mu   = 10.0 ** ((df.loc[use_mu, mu_col].to_numpy() - 25.0)/5.0)
        if mu_err_col in df.columns and np.any(np.isfinite(df.loc[use_mu, mu_err_col])):
            sig_mu = df.loc[use_mu, mu_err_col].to_numpy()
            # propagate: σ_d = d * (ln10/5) * σ_μ
            sig_d_mu = d_mu * (np.log(10.0)/5.0) * sig_mu
            # guard against zeros
            sig_d_mu = np.maximum(sig_d_mu, 1e-3)
        else: sig_d_mu = np.maximum(0.08 * d_mu, 1e-3) # if no μ errors available, assign a conservative 8% distance error
            
        # Cepheid-only distances
        if drop_ceph_if_mu: use_ceph = has_ceph & (~has_mu)
        else: use_ceph = has_ceph

        d_ceph = df.loc[use_ceph, ceph_col].to_numpy()
        # inflate CEPH-only uncertainties (systematics + inhomogeneity)
        sig_d_ceph = np.maximum(ceph_rel_sigma * d_ceph, ceph_abs_floor)

        # assemble vectors
        d   = np.concatenate([d_mu, d_ceph])
        sig = np.concatenate([sig_d_mu, sig_d_ceph])
        L   = np.concatenate([np.log1p(df.loc[use_mu,  z_col].to_numpy()),
                              np.log1p(df.loc[use_ceph, z_col].to_numpy())])

        n_mu, n_ceph = d_mu.size, d_ceph.size
        if d.size < 5:
            print("[low-z α] Not enough calibrators after cuts."); 
            return None

        # --------------- design matrix ---------------
        if force_through_zero:
            # d = c1 * L  (no intercept)
            A = L[:, None]                   # shape (N,1)
        else:
            # d = c0 + c1 * L
            A = np.vstack([np.ones_like(L), L]).T  # shape (N,2)

        if d.size <= A.shape[1]:
            print("[low-z α] Not enough calibrators after cuts.")
            return None

        w_meas = 1.0 / np.maximum(sig, 1e-6)**2

        # --------------- solver: WLS or robust IRLS ---------------
        def solve_WLS(A, y, w):
            sw = np.sqrt(w)                   # (N,)
            Aw = A * sw[:, None]              # (N×k)
            yw = y * sw                       # (N,)

            ATA = Aw.T @ Aw                   # (k×k)
            ATy = Aw.T @ yw                   # (k,)

            try: cov = np.linalg.inv(ATA)
            except np.linalg.LinAlgError: cov = np.linalg.pinv(ATA)

            theta = cov @ ATy
            res = y - A @ theta
            dof = max(len(y) - A.shape[1], 1)
            chi2 = np.sum(w * res**2)         # still χ² with *w*, not *sw*
            cov_scaled = cov * (chi2 / dof)
            return theta, cov_scaled, chi2, res

        if not use_robust: theta, cov, chi2, res = solve_WLS(A, d, w_meas)
        else:
            # Huber IRLS
            theta = None
            w = w_meas.copy()
            for _ in range(50):
                theta_old = None if theta is None else theta.copy()
                theta, cov, chi2, res = solve_WLS(A, d, w)
                # robust weights
                # scale via MAD (consistent with Gaussian)
                s = 1.4826 * np.median(np.abs(res - np.median(res)))
                s = max(s, 1e-6)
                u = res / s
                # Huber psi → weights
                w_rob = np.where(np.abs(u) <= huber_c, 1.0, huber_c/np.abs(u))
                w = w_meas * w_rob
                if theta_old is not None and np.allclose(theta, theta_old, rtol=1e-6, atol=1e-9):
                    break

        # --------------- unpack & convert to alpha ---------------
        if force_through_zero: c1 = float(theta[0]); var_c1 = float(cov[0,0])
        else:
            c0 = float(theta[0])
            c1 = float(theta[1]); var_c1 = float(cov[1,1])

        alpha = 1.0 / c1
        sig_alpha = np.sqrt(var_c1) / (c1*c1)

        # --------------- logging ---------------
        print(f"[low-z α] Using {d.size} calibrators (z≤{zmax}): N_μ={n_mu}, N_CEPH={n_ceph}, "
              f"{'robust' if use_robust else 'WLS'}, "
              f"{'no-intercept' if force_through_zero else 'with-intercept'}")
        if not force_through_zero: print(f"[low-z α] c0={c0:.3f} Mpc, c1={c1:.6e} 1,  χ²={chi2:.2f}")
        else: print(f"[low-z α] c1={c1:.6e} 1,  χ²={chi2:.2f}")
        print(f"[low-z α] α (=a) = {alpha:.6e} ± {sig_alpha:.6e} 1/Mpc")

        summary = dict( n_total=int(d.size), n_mu=int(n_mu), n_ceph=int(n_ceph),
            robust=bool(use_robust), force_zero=bool(force_through_zero),
            c1=c1, alpha=alpha, sigma_alpha=sig_alpha, chi2=chi2)

        if not force_through_zero: summary["c0"] = c0
        return alpha, sig_alpha, summary


    # ---------- UMH (non-exp) Perform Calibration on Data ----------
    if DO_LOWZ_CALIB:
        lowz_out = lowz_alpha_from_calibrators(
            df,
            zmax=ZMAX_CALIB,
            drop_ceph_if_mu=True,   # (a) drop CEPH if μ exists
            ceph_rel_sigma=0.08,    # (b) inflate CEPH-only errors
            ceph_abs_floor=0.2,
            use_robust=True,        # turn on robust Huber IRLS
            huber_c=2.0,          # a hair tighter than 1.345
            force_through_zero=False #False
        )
        if lowz_out is not None:
            A_NATIVE = lowz_out[0]
            print(f"[low-z α] Adopted A_NATIVE = {A_NATIVE:.6e} 1/Mpc for the UMH run.")
        else:
            A_NATIVE = H0 / config["LIGHT_SPEED"]        # ≈ H0/c in 1/Mpc
            print(f"[Calculated α] A_NATIVE={A_NATIVE} 1/Mpc")
            # A_NATIVE = 2.2118e-4                  # 1/Mpc from low-z calibration (ln(1+z) ≈ α d), see UMH_Compressed A.2.8
    else:
        A_NATIVE = H0 / config["LIGHT_SPEED"]        # ≈ H0/c in 1/Mpc
        print(f"[Calculated α] A_NATIVE={A_NATIVE} 1/Mpc")
        # A_NATIVE = 2.2118e-4                  # 1/Mpc from low-z calibration (ln(1+z) ≈ α d), see UMH_Compressed A.2.8

    # ---------- UMH (non-exp) Setup Defaults ----------

    mu_noatt = mu_umh_of_z_nonexp(z, a=A_NATIVE, s=0.0, b=0.0, c_=0.0, delta=delta_td, T_of_z=None)
        
    # --- optional PV floor diagnostic (one-time) --------------------
    def maybe_add_pv_floor(C, z, sigma_v=250.0):   # km/s
        c_kms = config["LIGHT_SPEED"] #299792.458
        # convert to σ_μ per-SN: σ_μ = (5/ln10) * σ_v / (c z)
        sig_mu_pv = (5.0/np.log(10.0)) * (sigma_v / (c_kms * np.maximum(z, 1e-3)))
        add = sig_mu_pv**2
        return C + np.diag(add)

    if DO_PREWHITEN:
        # whiten residuals using current Csel and a UMH baseline (mu_noatt + profiled M)
        try:
            chi2_tmp, M_tmp = chi2_and_M_best(mb_corr, mu_noatt, Csel)
            L_wh = np.linalg.cholesky(Csel)
            r_wh = np.linalg.solve(L_wh, mb_corr - (mu_noatt + M_tmp))
            mask_lowz = (z < 0.02)
            if mask_lowz.any():
                rms_lowz = float(r_wh[mask_lowz].std(ddof=1))
                print(f"[diag] whitened RMS at z<0.02 = {rms_lowz:.2f}")
                if rms_lowz > 1.2:  # PV under-fit → add a small floor
                    Csel = maybe_add_pv_floor(Csel, z, sigma_v=250.0)
                    err_plot = np.sqrt(np.diag(Csel))  # refresh plotting errors
                    print("[diag] Applied PV floor (σ_v=250 km/s) to Csel.")
            else: print("[diag] No z<0.02 calibrators → PV check skipped.")
        except Exception as e: print(f"[diag] PV-floor check skipped: {e}")
    # ---------------------------------------------------------------


    def betas_from_residuals_GLS(z, mb_corr, mu_noatt, C, huber_c=1.35, max_iter=30):
        """
        Robust GLS for (M, C1, C2) in Δμ ≈ M + C1*L + C2*L^2 with C-aware recentering.
        Returns (β1, β2) with Δμ = 1.086*(β1 L + β2 L^2).
        """
        L = np.log1p(z)

        one = np.ones_like(L)
        # Factor C once
        cf = cho_factor(C, overwrite_a=False, check_finite=False)
        # Compute C^{-1} 1  by solving C x = 1
        Cinv_one = cho_solve(cf, one, check_finite=False)
        # Now compute the weighted mean pieces
        num = float(L @ Cinv_one)        # L^T C^{-1} 1
        den = float(one @ Cinv_one)      # 1^T C^{-1} 1
        L0  = num / den                  # weighted "mean" of L under C^{-1}

        Lc = L - L0
        A = np.vstack([np.ones_like(Lc), 1.086*Lc, 1.086*(Lc**2)]).T  # design in centered basis

        # pre-whiten
        Lch = np.linalg.cholesky(C)
        y = np.linalg.solve(Lch, mb_corr - mu_noatt)
        X = np.linalg.solve(Lch, A)

        # Huber IRLS
        w = np.ones_like(y); theta = None
        for _ in range(max_iter):
            sw = np.sqrt(w); Xw = X * sw[:, None]; yw = y * sw
            ATA = Xw.T @ Xw; ATy = Xw.T @ yw
            try: cov = np.linalg.inv(ATA)
            except np.linalg.LinAlgError: cov = np.linalg.pinv(ATA)
            theta_new = cov @ ATy
            r = y - X @ theta_new
            s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
            u = r / s
            w_new = np.where(np.abs(u) <= huber_c, 1.0, huber_c/np.abs(u))
            if theta is not None and np.allclose(theta_new, theta, rtol=1e-7, atol=1e-9):
                theta = theta_new; break
            theta, w = theta_new, w_new
        M_c, C1_c, C2_c = map(float, theta)   # centered-basis coefficients

        # Convert centered-basis (Lc) → original L basis:
        # Δμ = M_c + C1_c*(L-L0) + C2_c*(L-L0)^2 = M + C1*L + C2*L^2
        C2 = C2_c
        C1 = C1_c - 2.0*C2_c*L0
        M  = M_c - C1_c*L0 + C2_c*(L0**2)

        beta1, beta2 = C1/1.086, C2/1.086
        return beta1, beta2


    def betas_jackknife_idsurvey(ids, z, mb_corr, mu_noatt, C, huber_c=1.35, min_keep=30):
        """
        Jackknife β's by leaving out one IDSURVEY code at a time.
        All inputs must be aligned 1:1 (same length/order).
        """
        assert ids.shape[0] == len(z) == len(mb_corr) == len(mu_noatt) == C.shape[0], \
            "ids/z/mb_corr/mu_noatt/C must be aligned to the same subset"

        uniq = np.unique(ids[np.isfinite(ids)])
        b1_list, b2_list = [], []
        for code in uniq:
            m = (ids != code)  # boolean mask SAME LENGTH as z etc.
            if m.sum() < min_keep:   # skip if this holdout leaves too few SNe
                continue
            b1, b2 = betas_from_residuals_GLS(
                z[m], mb_corr[m], mu_noatt[m], C[np.ix_(m, m)], huber_c=huber_c)

            b1_list.append(b1); b2_list.append(b2)

        if len(b1_list) >= 2: return float(np.median(b1_list)), float(np.median(b2_list))
        # fallback: use all data
        return betas_from_residuals_GLS(z, mb_corr, mu_noatt, C, huber_c=huber_c)


    # ---------- UMH (non-exp) Determine Data Betas ----------
    if DO_CALC_BETAS_FROM_DATA:
        for c in (1.35, 1.50, 1.80, 2.00):
            b1,b2 = betas_from_residuals_GLS(z, mb_corr, mu_noatt, Csel, huber_c=c)
            dmu = 1.086*(b1*np.log1p(z) + b2*np.log1p(z)**2)
            chi2,_ = chi2_and_M_best(mb_corr, mu_noatt + dmu, Csel)
            # GLS slope after applying betas
            A = np.vstack([z, np.ones_like(z)]).T
            cf = np.linalg.cholesky(Csel)
            X = np.linalg.solve(cf, A)
            y = np.linalg.solve(cf, mb_corr - (mu_noatt + dmu))
            m,_ = np.linalg.lstsq(X, y, rcond=None)[0]
            print(f"c={c:.2f}  β1={b1:.4f}  β2={b2:.4f}  χ²={chi2:.1f}  GLS slope={m:.3f}")

        # IMPORTANT:
        # (A_NATIVE, UMH_BETA1_THEORY, UMH_BETA2_THEORY) are calibrated once
        # and then held fixed. In the main Hubble fit, the only effectively free
        # parameter is the profiled magnitude M (k=1 for UMH).
        UMH_BETA1_THEORY, UMH_BETA2_THEORY = betas_jackknife_idsurvey(ids_subset, z, mb_corr, mu_noatt, Csel, huber_c=2.0)

        print(f"From Data using Theory: β1={UMH_BETA1_THEORY:.4f}, β2={UMH_BETA2_THEORY:.4f}")
    else: 
        UMH_BETA1_THEORY=0.430923; UMH_BETA2_THEORY=-0.275549
        print(f"From Theory: β1={UMH_BETA1_THEORY:.4f}, β2={UMH_BETA2_THEORY:.4f}")

    print_umh_microphysics(a=float(A_NATIVE),
                       beta1=float(UMH_BETA1_THEORY),
                       beta2=float(UMH_BETA2_THEORY),
                       ell_int_mpc=None,  # Planck
                       label="Planck")



    def fit_umh_theory_fixed():
        """
        Use theory-fixed (a, β1, β2), profile M, and return an 'inc'-shaped dict
        This matches the "one-knob" UMH configuration described in the paper.
        """
        a = float(A_NATIVE)
        beta1, beta2 = float(UMH_BETA1_THEORY), float(UMH_BETA2_THEORY)
        # If β's are both zero, pass T_of_z=None to avoid useless calls
        Tcall = None if (beta1 == 0.0 and beta2 == 0.0) else make_Texp_umh_theory(beta1, beta2)

        mu = mu_umh_of_z_nonexp(z, a=a, s=0.0, b=0.0, c_=0.0, delta=delta_td, T_of_z=Tcall)
        chi2, M = chi2_and_M_best(mb_corr, mu, Csel)

        # Model comparison: k_umh=1 (only M profiled), k_lcdm=2 (Ωm + M).
        print(f"[UMH theory-fixed] a={a:.9g}, β1={beta1:g}, β2={beta2:g} → χ² = {chi2:.1f}  (k = 1, counting profiled M)")

        # k = 1 because only M is effectively free/profled; (a, β's) are fixed by theory here
        return dict(name='umh_theory', a=a, b=0.0, s=0.0, tau0=None, chi2=chi2, M=M, mu=mu, k=1, beta1=beta1, beta2=beta2)



    # ---------- UMH (non-exp) Run fit_umh_theory_fixed ----------
    inc = fit_umh_theory_fixed()
    a_best, b_best, tau0_best = inc['a'], inc['b'], inc['tau0']
    tau_lo_best = inc.get('tau_lo'); tau_hi_best = inc.get('tau_hi')

    mu_umh_best, chi2_umh_best, M_umh_best, k_umh = inc['mu'], inc['chi2'], inc['M'], inc['k']

    # One clean, truthful best-fit print
    print(f"{title}: UMH(non-exp) best-fit  a={a_best}; M={M_umh_best:.5f}; χ²={chi2_umh_best:.1f} (N={len_z})")
    # --------------------------------------------------------------------------


    # k_umh p-value (χ² tail; if χ² << dof this will be near 1 due to conservative covariances)
    dof = len_z - k_umh
    p_large = float(gammainc(dof/2, chi2_umh_best/2, np.inf) / gamma(dof/2))
    print(f"p(χ²≥obs) ≈ {p_large:.3f}")

    # ----------------- ΛCDM (fit Ωm the same way) -----------------
    def H_LCDM(z, Om): return H0 * np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))

    def d_L_LCDM(z, Om):
        integ, _ = quad(lambda zp: c / H_LCDM(zp, Om), 0.0, z)
        return (1.0 + z) * integ

    def mu_LCDM_param(z_array, Om):
        return np.array([5.0 * np.log10(d_L_LCDM(zi, Om)) + 25.0 for zi in z_array])

    def chi2_for_Om(Om):
        mu = mu_LCDM_param(z, Om)
        return chi2_and_M_best(mb_corr, mu, Csel)[0]

    def slope_gls(x, r, C):
        """
        GLS fit of r = m*x + b with covariance C.
        Returns: m, b, m_err, b_err
        """
        A  = np.vstack([x, np.ones_like(x)]).T
        cf = cho_factor(C, overwrite_a=False, check_finite=False)
        # GLS normal equations
        ATA = A.T @ cho_solve(cf, A, check_finite=False)
        ATy = A.T @ cho_solve(cf, r, check_finite=False)
        theta = np.linalg.solve(ATA, ATy)   # [m, b]
        cov   = np.linalg.inv(ATA)
        m, b  = theta
        m_err = float(np.sqrt(cov[0, 0]))
        b_err = float(np.sqrt(cov[1, 1]))
        return m, b, m_err, b_err


    # Residuals & basic plots for UMH (best-fit)
    residuals_umh_best = mb_corr - (mu_umh_best + M_umh_best)

    plt.figure(figsize=(10,6))
    plt.errorbar(z, mb_corr, yerr=err_plot, fmt='.', label="Pantheon+ (m_b_corr)", alpha=0.5)
    parts = [f"a*={a_best:.3g}"]
    parts.append(f"β₁={inc.get('beta1',0):.3g}")
    if abs(inc.get('beta2',0.0)) > 0: parts.append(f"β₂={inc['beta2']:.3g}")
    lbl = "UMH non-exp (" + ", ".join(parts) + ")"
    plt.plot(z, mu_umh_best + M_umh_best, 'r-', label=lbl)

    #plt.plot(z, mu_umh_best + M_umh_best, 'r-', label=f"UMH non-exp (a*={a_best:.3g})")
    plt.xlabel("Redshift z"); plt.ylabel("Distance Modulus μ")
    plt.title(f"{title}: UMH(non-exp)  χ²={chi2_umh_best:.1f} (N={len_z})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{file_path}_Supernovae.png", dpi=dpi); plt.close()
    print(f"{title}: Supernovae Png Saved: {file_path}_Supernovae.png")

    plt.figure(figsize=(10,4))
    plt.errorbar(z, residuals_umh_best, yerr=err_plot, fmt='.')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Redshift z"); plt.ylabel("m_b_corr − (μ_UMH(non-exp) + M)")
    plt.title(f"{title}: Residuals (Pantheon+ − UMH non-exp)")
    plt.grid(True); plt.tight_layout()

    # GLS trend line on residuals vs z
    m_umh, b_umh, em_umh, eb_umh = slope_gls(z, residuals_umh_best, Csel)
    zz = np.linspace(float(z.min()), float(z.max()), 300)
    plt.plot(zz, m_umh*zz + b_umh, '-', lw=2, label=f'GLS trend: m={m_umh:.3f}±{em_umh:.3f}')
    plt.legend()

    plt.savefig(f"{file_path}_Residuals.png", dpi=dpi); plt.close()
    print(f"{title}: Residuals Png Saved: {file_path}_Residuals.png")

    # Residual stats
    print(f"{title}: Residuals stats: min={np.nanmin(residuals_umh_best):.3f}, "
          f"max={np.nanmax(residuals_umh_best):.3f}, "
          f"nan={np.isnan(residuals_umh_best).sum()}, inf={np.isinf(residuals_umh_best).sum()}")

    # Binned residuals (skip empty bins)
    bins = np.linspace(z.min(), z.max(), 20)
    stat, edges, _ = binned_statistic(z, residuals_umh_best, statistic='mean', bins=bins)
    cnt,  _,    _ = binned_statistic(z, residuals_umh_best, statistic='count', bins=edges)
    std,  _,    _ = binned_statistic(z, residuals_umh_best, statistic='std',   bins=edges)

    bin_centers = 0.5 * (edges[1:] + edges[:-1])

    # error on the mean; make empty-bin errors NaN so we can mask them
    err = np.where(cnt > 0, std / np.sqrt(cnt), np.nan)

    valid = (cnt > 0) & np.isfinite(stat) & np.isfinite(err)

    plt.figure(figsize=(8, 3))
    plt.errorbar(bin_centers[valid], stat[valid], yerr=err[valid], fmt='o-', capsize=3)
    plt.axhline(0, ls='--', color='k')
    plt.xlabel('z'); plt.ylabel('⟨res⟩ per bin')
    plt.title(f"{title}: Binned residuals (UMH non-exp)")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Binned_Residuals.png", dpi=dpi)
    plt.close()



    res_Om  = minimize_scalar(chi2_for_Om, bounds=(0.1, 0.5), method='bounded')
    Om_best = float(res_Om.x)
    mu_lcdm_best = mu_LCDM_param(z, Om_best)
    chi2_lcdm_best, M_l_best = chi2_and_M_best(mb_corr, mu_lcdm_best, Csel)
    print(f"{title}: ΛCDM best-fit Ωm={Om_best:.3f}, M={M_l_best:.5f}, χ²={chi2_lcdm_best:.1f}")

    # --- Overlay: Pantheon+ vs best-fit UMH(non-exp) vs best-fit ΛCDM ---
    plt.figure(figsize=(10,6))
    plt.errorbar(z, mb_corr, yerr=err_plot, fmt='o', ms=3, label='Pantheon+ (m_b_corr)', alpha=0.6)

    parts = [f"a*={a_best:.3g}"]
    parts.append(f"β₁={inc.get('beta1',0):.3g}")
    if abs(inc.get('beta2',0.0)) > 0: parts.append(f"β₂={inc['beta2']:.3g}")
    lbl = "UMH non-exp (" + ", ".join(parts) + ")"
    plt.plot(z, mu_umh_best + M_umh_best, 'r-', label=lbl)
    #plt.plot(z, mu_umh_best  + M_umh_best,  'r-',  label=f'UMH non-exp (a*={a_best:.3g})')
    plt.plot(z, mu_lcdm_best + M_l_best,    'g--', label=f'ΛCDM (Ωm*={Om_best:.3f})')
    plt.xlabel('Redshift z'); plt.ylabel('Distance Modulus μ')
    plt.title(f"{title}: Best-fit UMH(non-exp) vs ΛCDM vs Pantheon+")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{file_path}_vs_LCDM.png", dpi=dpi)
    plt.close()


    # --- Residuals (best-fits) + whitened comparisons ---
    residuals_lcdm_best = mb_corr - (mu_lcdm_best + M_l_best)
    L = np.linalg.cholesky(Csel)
    w_umh   = np.linalg.solve(L, residuals_umh_best)
    w_lcdm  = np.linalg.solve(L, residuals_lcdm_best)
    frac_closer = np.mean(np.abs(w_umh) < np.abs(w_lcdm))
    print(f'Fraction with |whitened residual| smaller for UMH: {frac_closer:.3f}')

    plt.figure(figsize=(12,5))
    plt.errorbar(z, residuals_umh_best,  yerr=err_plot, fmt='o', label=f'Pantheon+ − UMH (a*={a_best:.3g})', alpha=1.0)
    plt.errorbar(z, residuals_lcdm_best, yerr=err_plot, fmt='s', label=f'Pantheon+ − ΛCDM (Ωm*={Om_best:.3f})', alpha=0.6)
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('Redshift z'); plt.ylabel('Residuals (data − model)')
    plt.title(f'{title}: Residuals (best fits)')
    plt.legend(); plt.grid(True); plt.tight_layout()


    # GLS trend lines for both models (vs z)
    m_lcdm, b_lcdm, em_lcdm, eb_lcdm = slope_gls(z, residuals_lcdm_best, Csel)

    zz = np.linspace(float(z.min()), float(z.max()), 300)
    plt.plot(zz, m_umh*zz + b_umh, '-',  lw=2, label=f'UMH GLS: m={m_umh:.3f}±{em_umh:.3f}')
    plt.plot(zz, m_lcdm*zz + b_lcdm, '--', lw=2, label=f'ΛCDM GLS: m={m_lcdm:.3f}±{em_lcdm:.3f}')
    plt.legend()

    plt.savefig(f'{file_path}_vs_LCDM_Residuals.png', dpi=dpi); plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(w_umh,  bins=40, alpha=1.0, label=f'UMH (a*={a_best:.3g})')
    plt.hist(w_lcdm, bins=40, alpha=0.6, label=f'ΛCDM (Ωm*={Om_best:.3f})')
    plt.xlabel('whitened residual'); plt.ylabel('count')
    plt.title(f'{title}: Whitened residuals (should be ~N(0,1))')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{file_path}_WhitenedResiduals.png", dpi=dpi)
    plt.close()


    # --- Optional: zoom inset on high-z residuals (z > 1.5) ---
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
    ax.errorbar(z, residuals_umh_best,  yerr=err_plot, fmt='o', label=f'Pantheon+ − UMH (a*={a_best:.3g})',  alpha=1.0)
    ax.errorbar(z, residuals_lcdm_best, yerr=err_plot, fmt='s', label=f'Pantheon+ − ΛCDM (Ωm*={Om_best:.3f})', alpha=0.6)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('Residuals (data − model)')
    ax.set_title(f"{title}: Residuals"); ax.legend(); ax.grid(True)
    mask_hi = z > 1.5
    if np.any(mask_hi):
        axins = inset_axes(ax, width="35%", height="30%", loc='lower left',
                           bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
        axins.errorbar(z[mask_hi], residuals_umh_best[mask_hi],  yerr=err_plot[mask_hi], fmt='o', alpha=1.0)
        axins.errorbar(z[mask_hi], residuals_lcdm_best[mask_hi], yerr=err_plot[mask_hi], fmt='s', alpha=0.6)
        axins.axhline(0, color='k', linestyle='--')
        axins.set_xlim(1.5, float(z[mask_hi].max()))
        axins.set_ylim(-0.6, 0.6)
        axins.set_title('Zoom: z > 1.5', fontsize=10)
    plt.savefig(f"{file_path}_vs_LCDM_Residuals_Zoom.png", dpi=dpi)
    plt.close()


    # --- Annotated overlay with survey-region shading (best fits) ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(z, mb_corr, yerr=err_plot, fmt='o', ms=3, label='Pantheon+ (m_b_corr)', alpha=0.6)
    parts = [f"a*={a_best:.3g}"]
    parts.append(f"β₁={inc.get('beta1',0):.3g}")
    if abs(inc.get('beta2',0.0)) > 0: parts.append(f"β₂={inc['beta2']:.3g}")
    lbl = "UMH non-exp (" + ", ".join(parts) + ")"
    plt.plot(z, mu_umh_best + M_umh_best, 'r-', label=lbl)
    #plt.plot(z, mu_umh_best  + M_umh_best,  'r-',  label=f'UMH non-exp (a*={a_best:.3g})')
    plt.plot(z, mu_lcdm_best + M_l_best,    'g--', label=f'ΛCDM (Ωm*={Om_best:.3f})')
    regions = [(0.01, 0.10, 'Low-z'), (0.10, 0.40, 'SDSS'), (0.40, 1.00, 'SNLS'), (1.00, 2.30, 'HST')]
    for zmin, zmax, _lab in regions: plt.axvspan(zmin, zmax, alpha=0.08, zorder=0)
    region_handles = [Patch(alpha=0.08, label=lab) for _, _, lab in regions]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles += region_handles; labels += [lab for _, _, lab in regions]
    plt.legend(handles, labels, loc='best', ncol=2)
    plt.xlabel('Redshift z'); plt.ylabel('Distance Modulus μ')
    plt.title(f"{title}: Best-fit UMH(non-exp) vs ΛCDM (with survey regions)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{file_path}_vs_LCDM_Annotated.png", dpi=dpi)
    plt.close()

    
    A = np.vstack([z, np.ones_like(z)]).T
    m_umh_ls, b_umh_ls = np.linalg.lstsq(A, residuals_umh_best, rcond=None)[0]
    m_lcdm_ls, b_lcdm_ls = np.linalg.lstsq(A, residuals_lcdm_best, rcond=None)[0]
    print(f"slope(residual vs z): UMH={m_umh_ls:.3f}, ΛCDM={m_lcdm_ls:.3f} mag per unit z")


    print(f"GLS slope: UMH={m_umh:.3f}±{em_umh:.3f} (b={b_umh:.3f}±{eb_umh:.3f}), "
          f"ΛCDM={m_lcdm:.3f}±{em_lcdm:.3f} (b={b_lcdm:.3f}±{eb_lcdm:.3f})")


    # --- Model comparison: AIC/BIC for best-fits ---
    k_lcdm = 2
    print(f"k_umh: UMH(non-exp)={int(k_umh)}, ΛCDM={int(k_lcdm)}")

    aic_umh  = chi2_umh_best  + 2 * k_umh
    bic_umh  = chi2_umh_best  + np.log(len_z) * k_umh
    aic_lcdm = chi2_lcdm_best + 2 * k_lcdm
    bic_lcdm = chi2_lcdm_best + np.log(len_z) * k_lcdm

    print(f"AIC: UMH(non-exp)={aic_umh:.1f}, ΛCDM={aic_lcdm:.1f} (Δ={aic_umh-aic_lcdm:+.1f})")
    print(f"BIC: UMH(non-exp)={bic_umh:.1f}, ΛCDM={bic_lcdm:.1f} (Δ={bic_umh-bic_lcdm:+.1f})")


    # Residuals vs x = ln(1+z), with GLS trend
    x  = np.log1p(z)
    mx, bx, emx, ebx = slope_gls(x, residuals_umh_best, Csel)

    plt.figure(figsize=(10,4))
    plt.errorbar(x, residuals_umh_best, yerr=err_plot, fmt='.')
    xx = np.linspace(float(x.min()), float(x.max()), 300)
    plt.plot(xx, mx*xx + bx, '-', lw=2, label=f'GLS trend: m={mx:.3f}±{emx:.3f}')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('x = ln(1+z)'); plt.ylabel('Residuals (data − model)')
    plt.title(f"{title}: Residuals vs ln(1+z) (UMH non-exp)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals_vs_ln1pz.png", dpi=dpi)
    plt.close()

    print(f"GLS slope in x=ln(1+z): m={mx:.3f}±{emx:.3f}  "
          f"(compare to vs z: {m_umh:.3f}±{em_umh:.3f})")


    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)