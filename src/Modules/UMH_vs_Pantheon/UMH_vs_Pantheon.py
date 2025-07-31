""" 
UMH vs Pantheon Test

Author: Andrew Dodge
Date: July 2025

Description:
Tests whether UMH can account for Pantheon Data

Parameters:
    - grid_size
    - dx
    - dt
    - frequencies to test

Output:
    - 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json

from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.

        #"LATTICE_SPACING":0.01, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.

        #"MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        #"MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
       
        "LIGHT_SPEED": 299792.458,  # speed of light in km/s

        "H0": 70, # Hubble constant in km/s/Mpc
        "BETA": 0.9, # UMH expansion exponent

        # Pantheon Explicit column list
        "PANTHEON_DATA_COLUMNS":["name", "zcmb", "zhel", "dz", "mb", "dmb", "x1", "dx1", "color", "dcolor","3rdvar", "d3rdvar", "cov_m_s", "cov_m_c", "cov_s_c", "set", "ra", "dec", "biascor"],
        "PANTHEON_DATA_FILE":os.path.join(base, "Output", "PantheonData", "lcparam_full_long.csv"),

        "DPI":300, #PNG Resolution.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }



def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    #Tu=config["MEDIUM_DENSITY"]
    #rho_u=config["MEDIUM_PRESSURE"]
    #dx=["LATTICE_SPACING"]

    c=config["LIGHT_SPEED"]

    H0=config["H0"]
    beta=config["BETA"]

    columns=config["PANTHEON_DATA_COLUMNS"]
    panfile=config["PANTHEON_DATA_FILE"]

    dpi=config["DPI"]

    outdir = config["OUTPUT_FOLDER"]

    title="UMH vs_Pantheon"
    file_hdr="UMH_vs_Pantheon"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")


    # UMH Model Parameters
    #H0 = 70  # Hubble constant in km/s/Mpc
    #beta = 0.9  # UMH expansion exponent
    #c = 299792.458  # speed of light in km/s

    #columns = ["name", "zcmb", "zhel", "dz", "mb", "dmb", "x1", "dx1", "color", "dcolor","3rdvar", "d3rdvar", "cov_m_s", "cov_m_c", "cov_s_c", "set", "ra", "dec", "biascor"]

    # Load with whitespace-based separator and assigned column names
    df = pd.read_csv(panfile, sep=',', comment='#', header=None, names=columns)

    print(f"Loaded {len(df)} supernovae...")
    print(df.head())

    # Now extract values
    z = df["zcmb"].values
    mb = df["mb"].values
    dmb = df["dmb"].values
    biascor = pd.to_numeric(df["biascor"], errors='coerce')

    # UMH prediction function
    def H_umh(z):
        return H0 * (1 + z)**beta

    def D_L_umh(z):
        integral, _ = quad(lambda z_: c / H_umh(z_), 0, z)
        return (1 + z) * integral  # in Mpc

    def chi2_for_M(M_guess):
        mu_obs = mb - M_guess
        mu_umh = np.array([5 * np.log10(D_L_umh(zi)) + 25 for zi in z])
        residuals = mu_obs - mu_umh
        chi2 = np.sum((residuals / dmb)**2)
        return chi2 / (len(z) - 2)

    res = minimize_scalar(chi2_for_M, bounds=(-19.5, -19.0), method='bounded')
    print(f"{title}: Optimal M:", res.x)
    print(f"{title}: Min χ²_red:", res.fun)

    M=res.x

    # Corrected distance modulus
    #mu_obs = mb - biascor
    #M = -19.3  # Typical SN Ia absolute magnitude
    mu_obs = mb - M


    print(f"{title}: Loaded {len(z)} supernovae")
    print(f"{title}: First few redshifts:", z[:5])
    print(f"{title}: First few magnitudes:", mb[:5])


    # Compute theoretical distance modulus
    mu_umh = np.array([5 * np.log10(D_L_umh(zi)) + 25 for zi in z])

    print(f"{title}: Sample D_L_umh (z=1):", D_L_umh(1))  # Should be ~6600 Mpc
    print(f"{title}: mu (z=1):", 5 * np.log10(D_L_umh(1)) + 25)  # Should be ~44


    #print("mu_obs :", np.isnan(mu_obs))
    #print("mu_umh:", np.isnan(mu_umh))

    valid = np.isfinite(mu_obs) & np.isfinite(mu_umh) & np.isfinite(dmb)
    z_valid = z[valid]
    mu_obs_valid = mu_obs[valid]
    mu_umh_valid = mu_umh[valid]
    dmb_valid = dmb[valid]

    residuals = mu_obs_valid - mu_umh_valid
    chi_squared = np.sum((residuals / dmb_valid)**2)
    reduced_chi2 = chi_squared / (len(residuals) - 2)

    # Residuals and Chi-squared
    #residuals = mu_obs - mu_umh
    #chi_squared = np.sum((residuals / dmb)**2)
    #reduced_chi2 = chi_squared / (len(z) - 2)


    # === Plot μ(z) comparison ===
    plt.figure(figsize=(10,6))
    plt.errorbar(z, mu_obs, yerr=dmb, fmt='.', label="Pantheon (obs)", alpha=0.5)
    plt.plot(z, mu_umh, 'r-', label=f"UMH Model (β={beta})")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance Modulus μ")
    plt.title(f"{title}: Supernovae — χ²_red = {reduced_chi2:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Pantheon_Supernovae.png", dpi=dpi)
    plt.close()


    print(f"{title}: Residuals stats:")
    print(f"{title}: min:", np.nanmin(residuals))
    print(f"{title}: max:", np.nanmax(residuals))
    print(f"{title}: nan count:", np.isnan(residuals).sum())
    print(f"{title}: inf count:", np.isinf(residuals).sum())


    # === Plot Residuals ===
    plt.figure(figsize=(10,4))
    plt.errorbar(z, residuals, yerr=dmb, fmt='.')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Redshift z")
    plt.ylabel("μ_obs - μ_UMH")
    plt.title(f"{title}: Residuals (Pantheon - UMH)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals.png", dpi=dpi)
    plt.close()


    #Overlay ACDM.

    

    # Constants
    c = 299792.458  # km/s
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_L = 0.7

    # ΛCDM Hubble parameter
    def H_LCDM(z):
        return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

    # Luminosity distance for ΛCDM
    def d_L_LCDM(z):
        integral, _ = quad(lambda z_prime: 1.0 / H_LCDM(z_prime), 0, z)
        return (1 + z) * c * integral

    # Distance modulus for ΛCDM
    def mu_LCDM(z_array):
        d_L_vals = np.array([d_L_LCDM(z) for z in z_array])
        return 5 * np.log10(d_L_vals) + 25

    def compute_aic_bic(chi_squared, num_params, num_points):
        aic = chi_squared + 2 * num_params
        bic = chi_squared + np.log(num_points) * num_params
        return aic, bic


    # Compute LCDM prediction
    mu_lcdm = mu_LCDM(z_valid)

    # --- Plot: Overlay of Pantheon, UMH, LCDM ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_valid, mu_obs_valid, yerr=dmb_valid, fmt='o', markersize=3, label='Pantheon (obs)', alpha=0.6)
    plt.plot(z_valid, mu_umh_valid, 'r-', label=f'UMH Model (β={beta})')
    plt.plot(z_valid, mu_lcdm, 'g--', label='ΛCDM Model')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus μ')
    plt.title("UMH vs ΛCDM vs Pantheon Supernovae")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_ACDM.png", dpi=dpi)
    plt.close()

    # --- Plot: Residuals ---
    plt.figure(figsize=(12, 5))
    plt.errorbar(z_valid, mu_obs_valid - mu_umh_valid, yerr=dmb_valid, fmt='o', label='Pantheon - UMH', alpha=0.6)
    plt.errorbar(z_valid, mu_obs_valid - mu_lcdm, yerr=dmb_valid, fmt='s', label='Pantheon - ΛCDM', alpha=0.6)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Redshift z')
    plt.ylabel('Residuals (μ_obs - μ_model)')
    plt.title("Residuals: Pantheon vs UMH and ΛCDM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_ACDM_Residuals.png", dpi=dpi)
    plt.close()


    chi2_umh = np.sum(((mu_obs_valid - mu_umh_valid) / dmb_valid)**2)
    chi2_lcdm = np.sum(((mu_obs_valid - mu_lcdm) / dmb_valid)**2)

    aic_umh, bic_umh = compute_aic_bic(chi2_umh, 1, len(mu_obs_valid))  # β only
    aic_lcdm, bic_lcdm = compute_aic_bic(chi2_lcdm, 2, len(mu_obs_valid))  # Ωm + ΩΛ

    print(f"{title}: AIC:", aic_umh, "BIC:", bic_umh)
    print(f"{title}: ΛCDM AIC:", aic_lcdm, "BIC:", bic_lcdm)


    # --- Residuals: Full Range ---
    #fig, ax = plt.subplots(figsize=(12, 5))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.errorbar(z_valid, mu_obs_valid - mu_umh_valid, yerr=dmb_valid, fmt='o', label='Pantheon - UMH', alpha=0.6)
    ax.errorbar(z_valid, mu_obs_valid - mu_lcdm, yerr=dmb_valid, fmt='s', label='Pantheon - ΛCDM', alpha=0.6)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residuals (μ_obs - μ_model)')
    ax.set_title("Residuals: Pantheon vs UMH and ΛCDM")
    ax.legend()
    ax.grid(True)

    # --- Zoom Inset on High-z Residuals (z > 1.5) ---
    #axins = inset_axes(ax, width="35%", height="40%", loc='upper right', borderpad=2)
    axins = inset_axes(ax, width="35%", height="30%", loc='lower left',bbox_to_anchor=(0.05, 0.05, 1, 1),bbox_transform=ax.transAxes, borderpad=2)

    # Filter high-z data
    z_high = z_valid[z_valid > 1.5]
    mu_obs_high = mu_obs_valid[z_valid > 1.5]
    mu_umh_high = mu_umh_valid[z_valid > 1.5]
    mu_lcdm_high = mu_lcdm[z_valid > 1.5]
    dmb_high = dmb_valid[z_valid > 1.5]

    # Plot zoomed-in residuals
    axins.errorbar(z_high, mu_obs_high - mu_umh_high, yerr=dmb_high, fmt='o', alpha=0.6)
    axins.errorbar(z_high, mu_obs_high - mu_lcdm_high, yerr=dmb_high, fmt='s', alpha=0.6)
    axins.axhline(0, color='k', linestyle='--')
    axins.set_xlim(1.5, max(z_high))
    axins.set_ylim(-0.6, 0.6)
    axins.set_title(f"{title}: Zoom: Residuals for $z > 1.5$", fontsize=10)

    #plt.tight_layout()
    plt.savefig(f"{file_path}_vs_ACDM_Residuals_Zoom.png", dpi=dpi)
    plt.close()


    # --- Plot: Overlay of Pantheon, UMH, LCDM with Survey Regions ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_valid, mu_obs_valid, yerr=dmb_valid, fmt='o', markersize=3, label='Pantheon (obs)', alpha=0.6)
    plt.plot(z_valid, mu_umh_valid, 'r-', label=f'UMH Model (β={beta})')
    plt.plot(z_valid, mu_lcdm, 'g--', label='ΛCDM Model')

    # Optional: Label survey data ranges (manual binning)
    bins = [(0.01, 0.1, 'Low-z'), (0.1, 0.4, 'SDSS'), (0.4, 1.0, 'SNLS'), (1.0, 2.3, 'HST')]
    for zmin, zmax, label in bins:
        plt.axvspan(zmin, zmax, alpha=0.08, label=label)

    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus μ')
    plt.title("UMH vs ΛCDM vs Pantheon Supernovae")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_ACDM_Annotated.png", dpi=dpi)
    plt.close()


    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)