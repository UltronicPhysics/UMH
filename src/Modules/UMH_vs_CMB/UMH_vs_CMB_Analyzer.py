"""
UMH_vs_CMB_Analyzer.py

Author: Andrew Dodge
Date: July 2025

Description:
Compare UMH Simulation to CMB Power Spectrum

Inputs:
- UMH_SimData.npy
- UMH_SimData_Velocity.npy

Outputs:
- Validation
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
import os
import sys
import json
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from numpy.polynomial.legendre import legval
from healpy import sphtfunc
from astropy.io import fits
from scipy.special import eval_legendre,legendre
from scipy.stats import chisquare
from scipy.ndimage import gaussian_filter1d


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d
    from ..UMH_SupportModules.UMH_HealPix import project_to_healpix_numba
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d
    from UMH_SupportModules.UMH_HealPix import project_to_healpix_numba
    


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "INPUT_SIMDATA_DIR": os.path.join(base, "Output", "UMH_SimData"),
        "PLANCK_DATA_TXT": os.path.join(base, "Output", "PlanckData", "COM_PowerSpect_CMB-tt-full_R3.01.txt"),
        "PLANCK_DATA_FITS": os.path.join(base, "Output", "PlanckData", "COM_CMB_IQU-smica_2048_R3.00_full.fits"),
        "SIZE": 4096, #4096
        "TIME_STEPS": 500,
        "LATTICE_SPACING": 1.0,
        "WAVE_SPEED": 1.0,
        "GAUSSIAN_SIGMA": 4, #8
        "LMAX":2048,

        "DTYPE":np.float64, #Precision.
        "DPI": 300,
        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def load_and_average_sim_data(sim_data_dir,title):
    files = sorted(glob.glob(os.path.join(sim_data_dir, "UMH_SimData_*.npy")))
    if not files:
        raise RuntimeError(f"❌ {title}: No UMH simulation data found.")
    accum = None
    for i, file in enumerate(files):
        print(f"{title}: Loading {file} ({i+1}/{len(files)})...")
        data = np.load(file, mmap_mode='r')
        accum = data if accum is None else accum + (data - accum) / (i + 1)
    print(f"✅ {title}: Files Load Complete.")
    return accum


def project_and_save_healpix_map(data, nside, file_path, title, gaussian_sigma=8, dpi=300, dtype=np.float64):
    smoothed_data = smooth_field_3d(data, kernel_size=1)
    filtered_data = gaussian_filter(smoothed_data, sigma=gaussian_sigma)
    Nx, Ny, Nz = filtered_data.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

    max_possible_r = min(cx, cy, cz) - 1  # Leave buffer inside grid
    r_max = int(max_possible_r * 1.2)
    r_max = min(r_max, max_possible_r)  # Clamp if 1.2 pushed it too far
    r_min = max(r_max - 2, 1)  # Avoid zero or negative radius
    #r_max = max_possible_r  # Use full radius available (not 1.2x overshoot)
    #shell_thickness = int(0.1 * r_max)  # Typically 10–16
    #r_min = max(r_max - shell_thickness, 1)

    assert r_min < r_max, f"Invalid shell range: r_min={r_min}, r_max={r_max}"

    #r_min=int(0.9 * min(cx, cy, cz)) #Sanity Test.
    #r_max=int(1.0 * min(cx, cy, cz)) #Sanity Test.
    apply_weighting=True #Set to True after test.
    flip_sign=False #Flip Test to correct for outward vs inward view from UMH vs CMB.

    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    #hp_map = project_to_healpix_numba(filtered_data, theta, phi, cx, cy, cz, r_min, r_max, apply_weighting=True, dtype=dtype)
    hp_map = project_to_healpix_numba(filtered_data, theta, phi, cx, cy, cz, r_min, r_max, apply_weighting=apply_weighting, flip_sign=flip_sign, dtype=dtype)


    hp.write_map(f"{file_path}_Projected_Strain.fits", hp_map, overwrite=True)
    hp.mollview(hp_map, title=f"{title} Projected Strain Map", unit="Strain")
    plt.savefig(f"{file_path}_Projected_Strain_Map.png", dpi=dpi)
    plt.close()
    print(f"✅ {title}: Projected Strain Map.")
    return hp_map


def compute_power_spectrum(hp_map, flip=False, lmax=None):
    if flip:
        hp_map = -hp_map
    cl = hp.anafast(hp_map)
    #if lmax is None:
    #    lmax = 3 * hp.get_nside(hp_map) - 1
    #cl = hp.anafast(hp_map, lmax=lmax)
    return cl, np.arange(len(cl))


def scale_cl_to_planck(cl_umh, ell_umh, planck_txt_path, file_path, title):
    planck_data = np.loadtxt(planck_txt_path)
    #planck_data = planck_data[planck_data[:, 0] <= 2048]
    ell_planck = planck_data[:, 0].astype(int)
    cl_planck = planck_data[:, 1]

    ell_common = np.intersect1d(ell_planck[ell_planck >= 30], ell_umh)
    #ell_common = np.intersect1d(ell_planck, ell_umh)
    if not len(ell_common):
        raise RuntimeError("❌ {title}: No overlapping ℓ range found for scaling.")

    cl_planck_common = cl_planck[np.isin(ell_planck, ell_common)]
    cl_umh_common = cl_umh[ell_common]

    scaling_factor = np.abs(np.mean(cl_planck_common) / np.mean(cl_umh_common))
    cl_umh_scaled = cl_umh * scaling_factor
    print(f"✅ {title}: Scale UMH to Planck.")

    #print(f"✅ {title}: Common ℓ values (used for scaling): {ell_common}")
    print(f"✅ {title}: Number of common ℓ values: {len(ell_common)}")

    # Optional — find and print uncommon ℓ values for both datasets:
    ell_umh_only = np.setdiff1d(ell_umh, ell_common)
    ell_planck_only = np.setdiff1d(ell_planck, ell_common)

    print(f"ℓ in UMH only: (not in Planck): Length:{len(ell_umh_only)}") # : Data:{ell_umh_only}
    print(f"ℓ in Planck only: (not in UMH): Length:{len(ell_planck_only)}") # : Data:{ell_planck_only}
    print(f"ℓ in UMH & Planck: (Common): Length:{len(ell_common)}") # : Data:{ell_common}

    if len(ell_common)>0:
        # Save to file
        with open(f"{file_path}_Common.txt", "w") as f:
            f.write(",".join(str(ell) for ell in ell_common) + "\n")

    if len(ell_umh)>0:
        # Save to file
        with open(f"{file_path}_UMH_Only.txt", "w") as f:
            f.write(",".join(str(ell) for ell in ell_umh_only) + "\n")

    if len(ell_planck_only)>0:
        # Save to file
        with open(f"{file_path}_Planck_Only.txt", "w") as f:
            f.write(",".join(str(ell) for ell in ell_planck_only) + "\n")

    # Save full Cl comparison (UMH vs Planck) for plotting
    with open(f"{file_path}_Power_Spectrum.txt", "w") as f:
        f.write("# ell    Cl_UMH_Scaled    Cl_Planck\n")
        for ell in ell_common:
            cl_u = cl_umh_scaled[ell]
            cl_p = cl_planck[ell_planck == ell][0]
            f.write(f"{ell:5d}    {cl_u:.6e}    {cl_p:.6e}\n")

    return cl_umh_scaled, ell_common, planck_data


def compute_c_theta(cl, ell, theta_rad):
    x = np.cos(theta_rad)
    c_theta = np.zeros_like(theta_rad)
    for i, xi in enumerate(x):
        coeffs = np.zeros(ell[-1] + 1)
        coeffs[ell] = (2 * ell + 1) / (4 * np.pi) * cl
        c_theta[i] = legval(xi, coeffs)
    return c_theta

# --- Function to Compute Cl and Angular Correlation C(theta) ---
def compute_cl_c_theta(t_map, ell_max=300):
    cl = hp.anafast(t_map)
    ell = np.arange(len(cl))
    
    # Clamp ell and cl to desired max
    ell = ell[:ell_max]
    cl = cl[:ell_max]

    theta_deg = np.linspace(0, 180, 361)
    theta_rad = np.deg2rad(theta_deg)

    # ✅ Call your helper function here
    c_theta = compute_c_theta(cl, ell, theta_rad)

    # Normalize
    c_theta_norm = c_theta / np.max(np.abs(c_theta))
    return theta_deg, c_theta_norm

def normalize_and_match_angular_correlation(c_umh, c_planck, theta_deg, match_theta_deg=60):
    idx_match = np.argmin(np.abs(theta_deg - match_theta_deg))
    scale_factor = np.abs(c_planck[idx_match] / c_umh[idx_match]) if c_umh[idx_match] != 0 else 1
    return c_umh * scale_factor, scale_factor


def rotate_healpix_map(hp_map, rot_angles=(180, 0, 0), degrees=True):
    nside = hp.get_nside(hp_map)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rotator = hp.Rotator(rot=rot_angles, deg=degrees)
    theta_rot, phi_rot = rotator(theta, phi)
    return hp.get_interp_val(hp_map, theta_rot, phi_rot)

def compute_bandwise_rmse(cl_umh, cl_planck, ell_common, bands):
    for band_start, band_end in bands:
        mask = (ell_common >= band_start) & (ell_common <= band_end)
        cl_umh_band = cl_umh[mask]
        cl_planck_band = cl_planck[mask]

        if len(cl_umh_band) == 0:
            print(f"[Warning] No overlapping ℓ values in band {band_start}-{band_end}")
            continue

        diff = cl_umh_band - cl_planck_band
        rmse = np.sqrt(np.mean(diff**2))
        print(f"RMSE in ℓ-band {band_start}-{band_end}: {rmse:.2e}")


def analyze_and_plot_bao(c_theta_umh, c_theta_planck, theta_deg, file_path, title, step, dpi=300):
    bao_min_deg = 0.5
    bao_max_deg = 10.0 #Was 5.0.  Broaden to not miss may miss some Planck BAO harmonics appearing at wider angles (>5deg).
    idx_range = (theta_deg >= bao_min_deg) & (theta_deg <= bao_max_deg)

    peaks_umh, _ = find_peaks(c_theta_umh[idx_range])
    peaks_planck, _ = find_peaks(c_theta_planck[idx_range])

    theta_umh_peaks = theta_deg[idx_range][peaks_umh]
    theta_planck_peaks = theta_deg[idx_range][peaks_planck]

    plt.figure(figsize=(10, 6))
    plt.plot(theta_deg, c_theta_umh, label="UMH Angular Correlation")
    plt.plot(theta_deg, c_theta_planck, label="Planck Angular Correlation", linestyle='--')
    plt.scatter(theta_umh_peaks, c_theta_umh[idx_range][peaks_umh], color='blue', label="UMH BAO Peaks")
    plt.scatter(theta_planck_peaks, c_theta_planck[idx_range][peaks_planck], color='orange', marker='x', label="Planck BAO Peaks")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Angular Separation θ [deg]")
    plt.ylabel("Normalized C(θ)")
    plt.title(f"{title} BAO Peaks in UMH vs Planck Angular Correlation {step}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_BAO_Comparison_{step}.png", dpi=dpi)
    plt.close()

    print(f"✅ {title}: BAO Analysis Complete. UMH Peaks at: {theta_umh_peaks}, Planck Peaks at: {theta_planck_peaks}")

def analyze_and_plot_cor(c_theta_umh, c_theta_planck, theta_deg, scale_factor, file_path, title, step, dpi=300):
    plt.figure()
    plt.plot(theta_deg, c_theta_planck, label="Planck SMICA", linestyle='--', color='green')

    plt.plot(theta_deg, c_theta_umh, label=f"UMH (Scaled by {scale_factor:.3e})", color='blue')

    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel("Angular Separation θ [deg]")
    plt.ylabel("Normalized C(θ)")
    plt.title(f"{title}: UMH vs Planck Angular Correlation {step}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_Angular_Correlation_{step}.png", dpi=dpi)
    plt.close()

    print(f"✅ {title}: Correlation Complete.")



def run_analysis(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    sim_data_dir =config["INPUT_SIMDATA_DIR"]

    planck_data_txt=config["PLANCK_DATA_TXT"]
    planck_data_fits=config["PLANCK_DATA_FITS"]

    nside = config["SIZE"]

    steps = config["TIME_STEPS"]

    v=config["WAVE_SPEED"]

    dx=config["LATTICE_SPACING"]

    lmax=config["LMAX"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]
    
    outdir = config["OUTPUT_FOLDER"]

    title="UMH vs CMB"
    file_hdr="UMH_vs_CMB"
  
    print(f"✅ Starting Test: {title} Valdation UMH vs CMB Spectrum.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")

    #Complete Preloading.

    strain_field = load_and_average_sim_data(sim_data_dir,title)
    hp_map = project_and_save_healpix_map(strain_field, nside, file_path,title, config["GAUSSIAN_SIGMA"],dpi=dpi,dtype=dtype)

    cl_umh, ell_umh = compute_power_spectrum(hp_map, flip=False, lmax=lmax)
    cl_umh_scaled, ell_common, planck_data = scale_cl_to_planck(cl_umh, ell_umh, planck_data_txt,file_path,title)


    print(f"\n📊 {title}: Bandwise RMSE diagnostics:")
    # Restrict to only ell_common
    cl_umh_common = cl_umh_scaled[np.isin(ell_umh, ell_common)]
    cl_planck_common = planck_data[:, 1][np.isin(planck_data[:, 0], ell_common)]

    compute_bandwise_rmse(cl_umh_common, cl_planck_common, ell_common, bands=[(2, 30), (31, 100), (101, 500), (501, 1500)])


    hp_map_rot = rotate_healpix_map(hp_map, rot_angles=(180, 0, 0))
    cl_umhRot, ell_umhRot = compute_power_spectrum(hp_map_rot, flip=False)
    cl_umhRot_scaled, ell_commonRot, planckRot_data = scale_cl_to_planck(cl_umhRot, ell_umhRot, planck_data_txt,f"{file_path}_Rot",title)

    # Plot Power Spectrum
    plt.figure()
    plt.loglog(ell_umh[1:], cl_umh_scaled[1:], label="UMH Scaled")
    plt.loglog(planck_data[:, 0], planck_data[:, 1], label="Planck TT", alpha=0.7)
    plt.xlabel("Multipole ℓ")
    plt.ylabel("Cℓ")
    plt.legend()
    plt.grid(True)
    plt.title(f"{title} UMH vs Planck Power Spectrum")
    plt.savefig(f"{file_path}_Power_Spectrum.png", dpi=dpi)
    plt.close()


    # Angular Correlation Function
    theta_deg = np.linspace(0, 180, 361)
    theta_rad = np.deg2rad(theta_deg)

    c_theta_umh = compute_c_theta(cl_umh_scaled, ell_umh, theta_rad)
    c_theta_umh /= np.max(np.abs(c_theta_umh))

    c_theta_umhRot = compute_c_theta(cl_umhRot_scaled, ell_umhRot, theta_rad)
    c_theta_umhRot /= np.max(np.abs(c_theta_umhRot))

    planck_map = hp.read_map(config["PLANCK_DATA_FITS"], field=0)

    #Planck Only.
    cl_planck_fits, ell_planck_fits = compute_power_spectrum(planck_map)
    c_theta_planck = compute_c_theta(cl_planck_fits, ell_planck_fits, theta_rad)
    c_theta_planck /= np.max(np.abs(c_theta_planck))
    #Planck Only.

    c_theta_umh_matched, scale_factor = normalize_and_match_angular_correlation(c_theta_umh, c_theta_planck, theta_deg)

    c_theta_umhRot_matched, scale_factor = normalize_and_match_angular_correlation(c_theta_umhRot, c_theta_planck, theta_deg)


    #Planck Only.
    c_theta_planck_norm = c_theta_planck / np.max(np.abs(c_theta_planck))

    c_theta_umh_norm = c_theta_umh_matched / np.max(np.abs(c_theta_umh_matched))
    
    c_theta_umhRot_norm = c_theta_umhRot_matched / np.max(np.abs(c_theta_umhRot_matched))


    # Angular Correlation Plot
    analyze_and_plot_cor(c_theta_umh, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH", dpi)
    analyze_and_plot_cor(c_theta_umh_matched, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH_Matched", dpi)
    analyze_and_plot_cor(c_theta_umh_norm, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH_Norm", dpi)

    analyze_and_plot_cor(c_theta_umhRot, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH_Rot", dpi)
    analyze_and_plot_cor(c_theta_umhRot_matched, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH_Matched", dpi)
    analyze_and_plot_cor(c_theta_umhRot_norm, c_theta_planck_norm, theta_deg, scale_factor, file_path, title,"UMH_Norm", dpi)

    # BAO Analysis and Plot
    analyze_and_plot_bao(c_theta_umh, c_theta_planck_norm, theta_deg, file_path,title,"UMH", dpi)
    analyze_and_plot_bao(c_theta_umh_matched, c_theta_planck_norm, theta_deg, file_path,title,"UMH_Matched", dpi)
    analyze_and_plot_bao(c_theta_umh_norm, c_theta_planck_norm, theta_deg, file_path,title,"UMH_Norm", dpi)

    analyze_and_plot_bao(c_theta_umhRot, c_theta_planck_norm, theta_deg, file_path,title,"UMH_Rot", dpi)
    analyze_and_plot_bao(c_theta_umhRot_matched, c_theta_planck_norm, theta_deg, file_path,title,"UMH_Matched", dpi)
    analyze_and_plot_bao(c_theta_umhRot_norm, c_theta_planck_norm, theta_deg, file_path,title,"UMH_Norm", dpi)

    plt.figure(figsize=(10, 5))
    residual = cl_umh_scaled[ell_common] - planck_data[:,1][np.isin(planck_data[:,0], ell_common)]
    plt.plot(ell_common, residual, label="Residual (UMH - Planck)")
    plt.axhline(0, color="black", lw=0.5)
    plt.xlabel("Multipole ℓ")
    plt.ylabel("Residual C_ℓ")
    plt.title(f"{title}: Residual per ℓ between UMH and Planck")
    plt.grid()
    plt.legend()
    plt.savefig(f"{file_path}_Power_Spectrum_Residuals.png", dpi=dpi)
    plt.close()

    # Difference in angular correlation
    diff_corr = c_theta_umh_norm - c_theta_umhRot_norm

    plt.figure(figsize=(10, 5))
    plt.plot(theta_deg, diff_corr, label="ΔC(θ) = UMH - Rotated UMH", color="purple")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(f"{title}: Difference in Angular Correlation (Original vs Rotated UMH)")
    plt.xlabel("θ [deg]")
    plt.ylabel("ΔC(θ)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_Angular_Correlation_Diff_UMH_vs_Rot.png", dpi=dpi)
    plt.close()

    if False:
        rotated_map = rotate_healpix_map(c_theta_umh_matched, rot_angles=(0, 180, 0))
        

        theta_deg, c_theta_umh = compute_cl_c_theta(c_theta_umh_matched) #compute_c_theta() 
        _, c_theta_rot = compute_cl_c_theta(rotated_map)

        # --- Plot ---
        plt.figure(figsize=(12, 6))
        plt.plot(theta_deg, c_theta_umh_matched, label="UMH Original", color="blue")
        plt.plot(theta_deg, rotated_map, label="UMH Rotated", color="orange")
        plt.plot(theta_deg, c_theta_planck, label="Planck SMICA", linestyle='--', color='green')
        #plt.plot(theta_deg, c_theta_planck_norm, label="Planck SMICA", linestyle='--', color="olive")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.xlabel("Angular Separation θ [degrees]")
        plt.ylabel("Normalized C(θ)")
        plt.title(f"{title}: Angular Correlation: Original UMH vs UMH Rotated vs Planck")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{file_path}_UMH_vs_UMH_Rotated_vs_Planck_AngCorr.png",dpi=dpi)
        plt.close()


    print(f"✅ Finished Test: {title} Valdation UMH vs CMB Spectrum.")


if __name__ == "__main__":
    run_analysis()
