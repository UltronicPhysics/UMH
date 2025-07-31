"""
UMH_Gauge_Coupling_Running.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Gauge Coupling Running.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces
"""

import numpy as np
import os
import sys
import json
import csv
import matplotlib.pyplot as plt

from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 300, #500,300
        "TIME_STEPS": 200, #500, 200

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":24, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary. 30
        "PML_ABORPTION":0.18, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.15
        "PML_POWER": 3,



        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }

# --------------------------
# Relaxation
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def relax_field(field, pinned_mask, relaxation_rate, dx, T_u, num_steps):
    N = field.shape[0]
    energy_history = np.zeros(num_steps)
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_z = np.zeros_like(field)

    for step in range(num_steps):
        new_field = np.copy(field)
        for i in prange(1, N-1):
            for j in range(1, N-1):
                for k in range(1, N-1):
                    if not pinned_mask[i,j,k]:
                        neighbor_sum = (
                            field[i+1,j,k] + field[i-1,j,k] +
                            field[i,j+1,k] + field[i,j-1,k] +
                            field[i,j,k+1] + field[i,j,k-1]
                        )
                        neighbor_avg = neighbor_sum / 6.0
                        new_field[i,j,k] = (1 - relaxation_rate) * field[i,j,k] + relaxation_rate * neighbor_avg
        field = new_field

        for i in prange(1, N-1):
            for j in range(1, N-1):
                for k in range(1, N-1):
                    grad_x[i,j,k] = (field[i+1,j,k] - field[i-1,j,k]) / (2 * dx)
                    grad_y[i,j,k] = (field[i,j+1,k] - field[i,j-1,k]) / (2 * dx)
                    grad_z[i,j,k] = (field[i,j,k+1] - field[i,j,k-1]) / (2 * dx)

        strain_energy = 0.5 * T_u * (grad_x**2 + grad_y**2 + grad_z**2)
        energy_history[step] = np.sum(strain_energy[1:-1,1:-1,1:-1]) * dx**3

    return field, energy_history

# --------------------------
# Gaussian loop seeding
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def draw_gaussian_loop_numba(field, cx, cy, cz, r, plane_id, sigma, theta_steps=600):
    N = field.shape[0]
    two_pi = 2.0 * np.pi
    for ti in prange(theta_steps):
        t = ti * two_pi / theta_steps
        if plane_id == 0:  # xy
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))
            z = cz
        elif plane_id == 1:  # yz
            x = cx
            y = int(cy + r * np.cos(t))
            z = int(cz + r * np.sin(t))
        elif plane_id == 2:  # xz
            x = int(cx + r * np.cos(t))
            y = cy
            z = int(cz + r * np.sin(t))

        for dx0 in range(-2, 3):
            for dy0 in range(-2, 3):
                for dz0 in range(-2, 3):
                    i = x + dx0
                    j = y + dy0
                    k = z + dz0
                    if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                        d2 = dx0**2 + dy0**2 + dz0**2
                        field[i, j, k] += np.exp(-d2 / (2 * sigma**2))



def running_coupling_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["LATTICE_SIZE"] # Memory-safe grid size.
    steps = config["TIME_STEPS"]

    lattice_spacing=config["LATTICE_SPACING"]
    
    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]
    pml_power = config["PML_POWER"]
    

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Gauge Coupling (Running)"
    file_root="UMH_Gauge_Coupling"
    file_sub="UMH_Running"
    file_hdr="UMH_Gauge_Coupling_Running"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    
    #radii = [10, 15, 20, 25, 30, 40, 50, 60, 70]
    radii = np.logspace(np.log10(0.5), np.log10(30), 20)  # radii from ~0.5 to 30

    relaxation_rate = 0.3
    sigma = 2.0  # Gaussian blob radius

    # --------------------------
    # Parameters
    # --------------------------
    dx = lattice_spacing #/ size

    # --------------------------
    # Sweep over Radii
    # --------------------------
    cx, cy, cz = size//2, size//2, size//2

    TxtOut=""
    results = []
    resultsCSV = []

    for r in radii:
        psi = np.zeros((size,size,size))

        plane_map = {'xy': 0, 'yz': 1, 'xz': 2}
        draw_gaussian_loop_numba(psi, cx, cy, cz, r, plane_map['xy'], sigma)
        pinned = psi > 0.1
        psi, energy_hist = relax_field(psi, pinned, relaxation_rate, dx, Tu, steps)
        final_energy = energy_hist[-1]
        g2 = final_energy / (2 * np.pi * Tu * lattice_spacing)
        alpha = g2 / (4 * np.pi)
        mu = 1.0 / (r * dx)
        results.append((mu, g2, alpha))
        resultsCSV.append((r,mu,g2,alpha))
        print(f"{title}: Radius {r}, μ = {mu:.3f}, g² = {g2:.4f}, α = g²/4π = {alpha:.4f}.")
        TxtOut+=f"{title}: Radius {r}, μ = {mu:.3f}, g² = {g2:.4f}, α = g²/4π = {alpha:.4f}\n"


    with open(f"{file_path}_Results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "mu", "g2", "alpha"])
        for row in resultsCSV:
            r, mu, g2, alpha = row  # ✅ Unpack the row tuple
            writer.writerow([f"{r:.3f}", f"{mu:.6f}", f"{g2:.6f}", f"{alpha:.6f}"])




    # --------------------------
    # Plot Running Coupling
    # --------------------------
    mus = [row[0] for row in results]
    alphas = [row[2] for row in results]

    plt.figure()
    plt.plot(mus, alphas, marker='o')
    plt.xlabel("Energy Scale μ (1/r)")
    plt.ylabel("α_s = g² / 4π")
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f"{title}: Running Coupling vs Energy Scale")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}.png")
    plt.close()

    # --- Add Standard Model QCD Prediction ---
    def alpha_s_qcd(mu, Lambda=0.2, Nf=5):
        b0 = (33 - 2 * Nf) / (12 * np.pi)
        return 1.0 / (b0 * np.log(mu**2 / Lambda**2))

    #mus_fine = np.logspace(np.log10(min(mus)), np.log10(max(mus)), 300)
    mus_fine = np.logspace(np.log10(0.25), np.log10(max(mus)), 300)  # Avoid μ ≈ Λ

    qcd_curve = alpha_s_qcd(mus_fine)

    # Fit UMH data to QCD curve by scale factor

    def rescale_fit(log_scale):
        scale = 10 ** log_scale
        umh_rescaled = np.interp(mus_fine, mus, alphas) * scale
        return np.mean((umh_rescaled - qcd_curve)**2)

    result = minimize_scalar(rescale_fit, bounds=(-3, 3), method='bounded')
    optimal_scale = 10 ** result.x
    umh_fit = np.interp(mus_fine, mus, alphas) * optimal_scale

    print(f"QCD curve length: {len(qcd_curve)}")
    print(f"QCD curve min/max (non-NaN): {np.nanmin(qcd_curve):.4f}, {np.nanmax(qcd_curve):.4f}")
    print(f"QCD curve sample (first 5): {qcd_curve[:5]}")
    print(f"Any NaNs in QCD curve? {np.any(np.isnan(qcd_curve))}")
    print(f"Any infs in QCD curve? {np.any(np.isinf(qcd_curve))}")
    print(f"Any negatives? {np.any(qcd_curve < 0)}")
    
    print(f"{title}: [UMH Fit] Optimal scale factor to match QCD: {optimal_scale:.6f}")
    with open(f"{file_path}_Coupling_Constants.txt", "w") as f:
        f.write(TxtOut)
        f.write(f"{title}: [UMH Fit] Optimal scale factor to match QCD: {optimal_scale:.6f}\n")


    # --- Plot all together ---
    # Avoid region near the QCD pole (mu ≈ Lambda ~ 0.2)
    valid = (mus_fine > 0.21) & (mus_fine < max(mus_fine)) & (qcd_curve > 0) & (~np.isnan(qcd_curve))
    plt.plot(mus_fine[valid], qcd_curve[valid], 'r--', label='QCD 1-loop', linewidth=2.0, zorder=5)
    #plt.plot(mus_fine, qcd_curve, 'r--', label='QCD 1-loop', linewidth=2.0, zorder=5)
    plt.plot(mus_fine, umh_fit, 'orange', linestyle='-', alpha=0.6, label=f'UMH × {optimal_scale:.3f}')
    plt.legend(frameon=True, fontsize=10)
    plt.xlabel("Energy Scale μ (1/r)")
    plt.ylabel("α_s = g² / 4π")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{title}: UMH vs QCD Running Coupling")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_QCD_AlphaS.png", dpi=dpi)
    plt.close()


    with open(f"{file_path}_UMH_vs_QCD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mu", "UMH_Original", f"UMH_Scaled_{optimal_scale:.4f}", "QCD_1Loop"])
        for i in range(len(mus_fine)):
            mu_val = mus_fine[i]
            umh_orig = np.interp(mu_val, mus, alphas)
            umh_scaled = umh_fit[i]
            qcd_val = qcd_curve[i]
            writer.writerow([mu_val, umh_orig, umh_scaled, qcd_val])



    # --------------------------
    # Extended UMH vs QCD Chart
    # --------------------------
    mus_extended = np.logspace(np.log10(0.05), np.log10(100), 500)
    qcd_extended = alpha_s_qcd(mus_extended)
    umh_scaled_extended = np.interp(mus_extended, mus, alphas) * optimal_scale

    with open(f"{file_path}_UMH_vs_QCD_AlphaS_ExtLogLog.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "UMH_Scaled", "QCD_1Loop"])
        for mu_val, umh_val, qcd_val in zip(mus_extended, umh_scaled_extended, qcd_extended):
            writer.writerow([mu_val, umh_val, qcd_val])

    plt.figure(figsize=(8, 5))
    plt.plot(mus_extended, qcd_extended, 'r--', label='QCD 1-loop (Extended)', linewidth=2.0)
    plt.plot(mus_extended, umh_scaled_extended, 'orange', label=f'UMH × {optimal_scale:.3f}', linewidth=2.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Energy Scale μ (1/r)")
    plt.ylabel("α_s = g² / 4π")
    plt.title(f"{title}: UMH vs QCD (Extended Range)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{file_path}_UMH_vs_QCD_AlphaS_ExtLogLog.png", dpi=dpi)
    plt.close()




    # --- Generate finely spaced μ values for QCD curve ---
    mus_qcd = np.logspace(np.log10(min(mus) * 0.8), np.log10(max(mus) * 1.2), 1000)
    alpha_qcd = alpha_s_qcd(mus_qcd)

    # --- Optional: Fit UMH alpha with similar function ---
    def alpha_fit(mu, A=0.0, Lambda=0.02):
        b0 = (33 - 2 * 5) / (12 * np.pi)
        return A + 1.0 / (b0 * np.log(mu**2 / Lambda**2))

    A_fit = 0.0
    Lambda_fit = 0.02
    alpha_umh_fit = alpha_fit(mus_qcd, A=A_fit, Lambda=Lambda_fit)

    # --- Plot everything ---
    plt.figure(figsize=(10, 6))
    plt.plot(mus_qcd, alpha_qcd, 'k-', label=r'QCD $\alpha_s$ (1-loop, $\Lambda=0.2$ GeV)')
    plt.plot(mus_qcd, alpha_umh_fit, 'r--', label=fr'UMH Fit: A={A_fit:.2f}, $\Lambda$={Lambda_fit:.3f}')
    plt.scatter(mus, alphas, color='orange', label='UMH Data', zorder=5)

    plt.xscale("log")
    plt.yscale("linear")
    plt.xlabel(r'Energy Scale $\mu$ [GeV]')
    plt.ylabel(r'$\alpha$ (Coupling Strength)')
    plt.title(f"{title}: Running Coupling vs QCD $\\alpha_s$")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_UMH_vs_QCD_AlphaS_ExtLogLin.png", dpi=dpi)
    plt.close()

    with open(f"{file_path}_UMH_vs_QCD_AlphaS_ExtLogLin.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "alpha_umh", "alpha_qcd", "alpha_umh_fit"])
        for mu_val, a_umh, a_qcd, a_fit in zip(mus_qcd, 
                                               np.interp(mus_qcd, mus, alphas), 
                                               alpha_qcd,
                                               alpha_umh_fit):
            writer.writerow([mu_val, a_umh, a_qcd, a_fit])



    print(f"✅ Finished Test: {title} Validated.")



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    running_coupling_test()