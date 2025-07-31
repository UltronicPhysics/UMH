"""
UMH_GW_Flux.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH GW Flux Test.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""

import numpy as np
import numba
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd

from numba import njit, prange
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import linregress


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp

    

def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "INPUT_SIMDATA": os.path.join(base, "Output", "UMH_SimData", "UMH_SimData.npy"),
        "INPUT_SIMDATA_VELOCITY": os.path.join(base, "Output", "UMH_SimData", "UMH_SimData_Velocity.npy"),
        "INPUT_SIMDATA_PREV": os.path.join(base, "Output", "UMH_SimData", "UMH_SimData_Prev.npy"),
        

        "LATTICE_SIZE": 768, #768,
        "TIME_STEPS": 1200, #400

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":40, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary.
        "PML_ABORPTION":0.15, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.04
        "PML_POWER": 2,

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def find_stable_far_field_start(slope_array, threshold=-0.1, window=3):
    for i in range(len(slope_array) - window):
        if np.all(slope_array[i:i+window] < threshold):
            return i
    return 0  # fallback to 0 if not found


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    #size = config["LATTICE_SIZE"] # Memory-safe grid size.
    steps = config["TIME_STEPS"]

    dx=config["LATTICE_SPACING"]
    
    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]
    pml_power = config["PML_POWER"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH GW Flux"
    file_hdr="UMH_GW_Flux"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    # ===============================
    # ✅ Load Strain Field Snapshots
    # ===============================
    # Load two snapshots to compute time derivative (velocity)
    field_now = np.load(config["INPUT_SIMDATA"])
    field_prev = np.load(config["INPUT_SIMDATA_PREV"])

    #field_now = np.load('umh_field_now.npy')
    #field_prev = np.load('field_step_250.npy')  # Replace with previous step file

    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    print(f"{title} Calculated Wave Speed Constant C = {v}")
    v2 = v ** 2

    # Check grid parameters
    size = field_now.shape[0]

    # Safety Factor Adjust GridSize.
    safety_factor=0.5 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)
 
    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor) #0.25 0.14
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")
    #dt = 1.0  # Time step between field_prev and field_now

    # ===============================
    # ✅ Compute Field Velocity (Time Derivative)
    # ===============================
    velocity_field = (field_now - field_prev) / dt

    # ===============================
    # ✅ Compute Energy Density (GW Energy)
    # ===============================
    # Energy density = (1/2) * (velocity^2 + |gradient|^2)

    def gradient(field, axis):
        return np.gradient(field, dx, axis=axis)

    grad_sq = sum(gradient(field_now, axis)**2 for axis in range(3))
    energy_density = 0.5 * (velocity_field**2 + grad_sq)

    # ===============================
    # ✅ Radial Flux Calculation
    # ===============================
    center = np.array([size//2]*3)
    coords = np.indices(field_now.shape).reshape(3, -1).T
    radii = np.linalg.norm(coords - center, axis=1)

    # Flatten fields
    energy_flat = energy_density.flatten()

    # Define radial bins
    bins = np.linspace(1, size/2 - 1, 40)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    flux_total = []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (radii >= low) & (radii < high)
        shell_volume = (4/3) * np.pi * (high**3 - low**3)
        if np.any(mask):
            total_energy = np.sum(energy_flat[mask]) * dx**3
            flux_density = total_energy / shell_volume
            flux_total.append(flux_density)
        else:
            flux_total.append(np.nan)

    # ===============================
    # ✅ Plot GW Energy Flux vs Radius
    # ===============================
    plt.figure(figsize=(8,6))
    plt.loglog(bin_centers, flux_total, 'o-', label='GW Energy Flux')
    plt.xlabel('Radius')
    plt.ylabel('Energy Density')
    plt.title(f"{title}: Energy Flux vs Radius")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_Radius.png")
    plt.close()

    # ===============================
    # ✅ Save Data
    # ===============================
    np.save(f"{file_path}_Bin_Centers.npy", bin_centers)
    np.save(f"{file_path}_Values.npy", flux_total)

    print(f"✅ {title}: Analysis Complete. Radius, Plot saved as '{file_path}_vs_Radius.png'")


    # ===============================
    # ✅ Compute and Plot Log-Log Slope (all radii)
    # ===============================
    valid_mask = (~np.isnan(flux_total)) & (np.array(flux_total) > 0) & (bin_centers > 0)
    r_valid = bin_centers[valid_mask]
    flux_valid = np.array(flux_total)[valid_mask]

    log_r = np.log10(r_valid)
    log_flux = np.log10(flux_valid)

    # Safe smoothing window
    N_all = len(log_flux)
    window_length_all = min(7, N_all if N_all % 2 == 1 else N_all - 1)
    if window_length_all < 3:
        log_flux_smooth = log_flux
    else:
        log_flux_smooth = savgol_filter(log_flux, window_length=window_length_all, polyorder=2)
    slope_smooth = np.gradient(log_flux_smooth, log_r)

    # Reference -2 line (for plotting, anchored left)
    true_slope_line = -2 * (log_r - log_r[0]) + log_flux[0]

    plt.figure(figsize=(8, 6))
    plt.plot(r_valid, slope_smooth, 'o-', color='orange', label='Smoothed Slope: d log F / d log r')
    #plt.plot(r_valid, true_slope_line - log_flux[0] + slope_smooth[0], '--', color='black', alpha=0.5, label='Reference Slope: -2')
    plt.xlabel("Radius")
    plt.ylabel("Log-Log Slope")
    plt.title(f"{title}: Local Slope of Energy Flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Slope_vs_Radius.png")
    plt.close()

    print(f"✅ {title}: Slope vs Radius, Plot saved as '{file_path}_Slope_vs_Radius.png'")

    # ===============================
    # ✅ Far-field only: use radius bounds only
    # ===============================
    min_far_radius = 0.45 * size              # e.g., ~345 if size = 768
    max_far_radius = (size / 2) - 2           # e.g., ~382 if size = 768

    start_index_radius = np.argmax(r_valid > min_far_radius)
    end_index_radius = np.argmax(r_valid > max_far_radius)
    if end_index_radius == 0:
        end_index_radius = len(r_valid) - 5

    # Use only the radius mask for far-field
    radius_mask = np.zeros_like(r_valid, dtype=bool)
    radius_mask[start_index_radius:end_index_radius] = True
    far_mask = radius_mask

    print(f"r_valid: {r_valid}")
    print(f"start_index_radius: {start_index_radius}, end_index_radius: {end_index_radius}")
    print(f"radius_mask.sum(): {np.sum(radius_mask)}")
    print(f"Far-field candidate radii: {r_valid[start_index_radius:end_index_radius]}")

    # --- Robust Far-field Window Selection ---
    fallback_triggered = False
    if start_index_radius >= end_index_radius or radius_mask.sum() == 0:
        print("⚠️ Far-field window empty or invalid; using last 6 points as far-field region.")
        fallback_triggered = True
        start_index_radius = max(0, len(r_valid) - 6)
        end_index_radius = len(r_valid)
        radius_mask = np.zeros_like(r_valid, dtype=bool)
        radius_mask[start_index_radius:end_index_radius] = True
        far_mask = radius_mask
        print(f"New far-field candidate radii: {r_valid[start_index_radius:end_index_radius]}")

    flux_fallback = flux_valid[start_index_radius:end_index_radius]
    print(f"flux_valid fallback region: {flux_fallback}")
    print(f"Any NaN in fallback? {np.any(np.isnan(flux_fallback))}")
    print(f"Any <=0 in fallback? {np.any(flux_fallback <= 0)}")

    # If your mask is still empty, try increasing the window
    if len(flux_fallback) == 0 or np.all(np.isnan(flux_fallback)) or np.all(flux_fallback <= 0):
        print("❌ No valid flux in fallback region. Try increasing the number of fallback points or debug the flux data itself.")
        return

    # --- Final Far-Field Arrays: Only Use Fallback Mask, No Extra Filtering! ---
    r_far = r_valid[far_mask]
    flux_far = flux_valid[far_mask]
    if len(r_far) == 0 or len(flux_far) == 0:
        print(f"❌ {title}: No valid far-field points. Aborting slope computation.")
        return

    log_r_far = np.log10(r_far)
    log_flux_far = np.log10(flux_far)

    # Safe smoothing window for far-field
    N_far = len(log_flux_far)
    window_length_far = min(7, N_far if N_far % 2 == 1 else N_far - 1)
    if window_length_far < 3:
        print(f"⚠️ {title}: Too few points to smooth; skipping smoothing.")
        log_flux_far_smooth = log_flux_far  # no smoothing fallback
    else:
        log_flux_far_smooth = savgol_filter(log_flux_far, window_length=window_length_far, polyorder=2)

    slope_far = np.gradient(log_flux_far_smooth, log_r_far)
    avg_slope_far = np.mean(slope_far)

    print(f"📏 {title}: r_valid range: {r_valid[0]:.2f} to {r_valid[-1]:.2f}")
    print(f"🛡️ {title}: Using min_far_radius = {min_far_radius:.2f}, max_far_radius = {max_far_radius:.2f}")
    print(f"🧠 {title}: Far-field region: r = {r_far[0]:.2f} to {r_far[-1]:.2f}, points = {len(r_far)}")


    # Reference -2 slope line for far-field (anchored left)
    true_slope_line_far = -2 * (log_r_far - log_r_far[0]) + log_flux_far[0]

    plt.figure(figsize=(8, 6))
    plt.plot(r_far, slope_far, 'o-', label='Far-Field Slope: d log F / d log r')
    plt.plot(r_far, true_slope_line_far - log_flux_far[0] + slope_far[0], '--', color='black', alpha=0.6, label='Reference Slope: -2')

    plt.annotate(f"Avg. Slope ≈ {avg_slope_far:.3f}", 
                 xy=(r_far[len(r_far)//2], slope_far[len(slope_far)//2]),
                 xytext=(r_far[0] + 10, slope_far[len(slope_far)//2]),
                 fontsize=11, color='blue')

    plt.xlabel("Radius")
    plt.ylabel("Log-Log Slope")
    plt.title(f"{title}: Far-Field Slope of Energy Flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_FarField_Slope_vs_Radius.png")
    plt.close()

    slope, intercept, _, _, _ = linregress(np.log(r_far), np.log(flux_far))
    print(f"{title}: Best fit log-log slope in far-field: {slope:.3f}")


    print(f"✅ {title}: Far-Field Slope vs Radius, Plot saved as '{file_path}_FarField_Slope_vs_Radius.png'")


    print(f"✅ Finished Test: {title} Validation.")

if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()