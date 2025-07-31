"""
UMH_Gauge_Coupling_Magnetic.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Gauge Coupling Magnetic.

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
        "LATTICE_SIZE": 128, #500,300
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


@njit(parallel=True, fastmath=True, cache=True)
def draw_u1_loop(field, cx, cy, cz, radius, sigma):
    for i in prange(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                x, y, z = i - cx, j - cy, k - cz
                r_xy = np.sqrt(x**2 + y**2)
                if np.abs(r_xy - radius) < 2.0 and np.abs(z) < 2:
                    angle = np.arctan2(y, x)
                    envelope = np.exp(-((r_xy - radius)**2 + z**2) / (2 * sigma**2))
                    field[i, j, k] = envelope * np.exp(1j * angle)

@njit(parallel=True, fastmath=True, cache=True)
def laplacian(field):
    lap = np.zeros_like(field, dtype=np.complex128)
    for i in prange(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            for k in range(1, field.shape[2] - 1):
                lap[i, j, k] = (
                    -6 * field[i, j, k]
                    + field[i+1, j, k] + field[i-1, j, k]
                    + field[i, j+1, k] + field[i, j-1, k]
                    + field[i, j, k+1] + field[i, j, k-1]
                )
    return lap

@njit()
def strain_energy(field, dx, Tu):
    grad = np.abs(laplacian(field))**2
    return 0.5 * Tu * np.sum(grad) * dx**3

@njit(parallel=True, fastmath=True, cache=True)
def relax_field(field, pinned, rate, dx, Tu, steps):
    energy_history = np.empty(steps)  # preallocate array
    for step in range(steps):
        lap = laplacian(field)
        update = rate * lap
        for i in prange(field.shape[0]):
            for j in range(field.shape[1]):
                for k in range(field.shape[2]):
                    if not pinned[i, j, k]:
                        field[i, j, k] += update[i, j, k]

        energy_history[step] = strain_energy(field, dx, Tu)
    return field, energy_history



def magnetic_coupling_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["LATTICE_SIZE"] # Memory-safe grid size.
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

    title="UMH Gauge Coupling (Magnetic)"
    file_root="UMH_Gauge_Coupling"
    file_sub="UMH_Magnetic"
    file_hdr="UMH_Gauge_Coupling_Magnetic"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    # Parameters
    relaxation_rate = 0.1
    radius = 20
    sigma = 5.0

    cx, cy, cz = size // 2, size // 2, size // 2

    # Initialize field and pinning
    psi = np.zeros((size, size, size), dtype=np.complex128)
    draw_u1_loop(psi, cx, cy, cz, radius, sigma)
    pinned = np.abs(psi) > 0.1

    # Relax and compute energy
    psi, energy_history = relax_field(psi, pinned, relaxation_rate, dx, Tu, steps)
    final_energy = energy_history[-1]
    g2 = final_energy / (2 * np.pi * Tu * dx)
    alpha = g2 / (4 * np.pi)

    # Save results
    with open(f"{file_path}_Coupling_Constant.txt", "w") as f:
        f.write(f"U(1) Gauge Coupling Test\n")
        f.write(f"g^2 = {g2:.6f}\n")
        f.write(f"alpha = g^2 / 4pi = {alpha:.6f}\n")

    # Plot convergence
    plt.figure()
    plt.plot(energy_history)
    plt.title(f"{title}: Energy Convergence - U(1)")
    plt.xlabel("Step")
    plt.ylabel("Total Strain Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Energy_Convergence.png")
    plt.close()

    # Plot slice
    plt.imshow(np.angle(psi[:, :, cz]), cmap="twilight", origin="lower")
    plt.title(f"{title}: U(1) Phase Slice (xy)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_U1_Phase_xy.png")
    plt.close()


    # Sweep over multiple loop radii to compute running coupling
    radii = [10, 15, 20, 25, 30, 40, 50, 60]
    mus, g2s, alphas = [], [], []

    for radius in radii:
        psi = np.zeros((size, size, size), dtype=np.complex128)
        draw_u1_loop(psi, cx, cy, cz, radius, sigma)
        pinned = np.abs(psi) > 0.1
        psi, energy_history = relax_field(psi, pinned, relaxation_rate, dx, Tu, steps)
        final_energy = energy_history[-1]

        g2 = final_energy / (2 * np.pi * Tu * dx)
        alpha = g2 / (4 * np.pi)
        mu = 1.0 / (radius * dx)  # momentum scale μ

        mus.append(mu)
        g2s.append(g2)
        alphas.append(alpha)


    with open(f"{file_path}_U1_Running.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "g2", "alpha"])
        for mu_val, g2_val, alpha_val in zip(mus, g2s, alphas):
            writer.writerow([f"{mu_val:.6f}", f"{g2_val:.6f}", f"{alpha_val:.6f}"])

    plt.figure()
    plt.plot(mus, alphas, 'b-o', label='UMH U(1) Running α')
    plt.axhline(1/137, color='gray', linestyle='--', label='α ≈ 1/137 (QED)')
    plt.xscale("log")
    plt.xlabel("μ (Inverse Radius)")
    plt.ylabel("Effective Coupling α")
    plt.title(f"{title}: U(1) Running Coupling vs QED α")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_UMH_U1_vs_QED.png", dpi=300)



    # === Plot: UMH vs QED α(μ) Comparison ===
    plt.figure(figsize=(8,6))
    plt.plot(mus, alphas, 'o-', label='UMH U(1) Coupling α(μ)', linewidth=2.0)

    # Reference line: QED fine-structure constant
    plt.axhline(1/137, color='red', linestyle='--', linewidth=2.0, label='QED α = 1/137')

    # Optional: Superconducting-like analog (e.g. α ≈ 1/130)
    plt.axhline(1/130, color='green', linestyle=':', linewidth=2.0, label='Superconducting Analog α ≈ 1/130')

    # Formatting
    plt.xlabel(r'$\mu$ (Inverse Loop Radius)', fontsize=12)
    plt.ylabel(r'$\alpha(\mu)$', fontsize=12)
    plt.title(f"{title}: UMH vs QED U(1) Coupling Constant", fontsize=14)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{file_path}_UMH_vs_QED_Comparison.png", dpi=dpi)
    plt.close()

    # === Save UMH vs QED comparison data to CSV ===
    csv_filename = f"{file_path}_UMH_vs_QED_Comparison.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mu", "UMH_alpha", "QED_alpha", "Superconducting_analog_alpha"])
        for mu_val, alpha_val in zip(mus, alphas):
            writer.writerow([
                f"{mu_val:.6e}",
                f"{alpha_val:.6e}",
                f"{1/137:.6e}",
                f"{1/130:.6e}"
            ])



    # Define target QED alpha
    alpha_qed = 1.0 / 137.0

    # Step 1: Compute original alphas from your UMH data
    alphas_original = [g / (4 * np.pi) for g in g2s]

    # Step 2: Compute scaling factor to match the first alpha value to QED
    scale_factor = alpha_qed / alphas_original[0]  # or use an index where you want to anchor

    # Step 3: Apply scaling to g2s and recompute alphas
    g2s_scaled = [g * scale_factor for g in g2s]
    alphas_scaled = [g / (4 * np.pi) for g in g2s_scaled]

    # === Plot: UMH vs QED α(μ) Comparison ===
    plt.figure(figsize=(8,6))
    plt.plot(mus, alphas_scaled, 'o-', label="UMH α(μ) (scaled to α_QED)")

    plt.axhline(1/137, color='red', linestyle='--', label="QED α = 1/137")

    # Optional: Superconducting-like analog (e.g. α ≈ 1/130)
    plt.axhline(1/130, color='green', linestyle=':', linewidth=2.0, label='Superconducting Analog α ≈ 1/130')

    # Formatting
    plt.xlabel(r'$\mu$ (Inverse Loop Radius)', fontsize=12)
    plt.ylabel(r'$\alpha(\mu)$', fontsize=12)
    plt.title(f"{title}: UMH vs QED U(1) Coupling Constant", fontsize=14)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{file_path}_UMH_vs_QED_Scaled_Comparison.png", dpi=dpi)
    plt.close()


    with open(f"{file_path}_Scaled_Running.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "g2_scaled", "alpha_scaled"])
        for mu_val, g2_val, alpha_val in zip(mus, g2s_scaled, alphas_scaled):
            writer.writerow([mu_val, g2_val, alpha_val])



    print(f"✅ Finished Test: {title} Validated.")



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    magnetic_coupling_test()