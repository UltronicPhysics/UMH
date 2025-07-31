"""
UMH_Gauge_Coupling_Strong.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Gauge Coupling Strong.

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

    T_xx = T_u * grad_x * grad_x
    T_yy = T_u * grad_y * grad_y
    T_zz = T_u * grad_z * grad_z
    T_xy = T_u * grad_x * grad_y
    return field, energy_history, T_xx, T_yy, T_zz, T_xy

# --------------------------
# Gaussian loop seeding
# --------------------------
def draw_gaussian_loop(field, size, cx, cy, cz, r, plane='xy', sigma=2.5):
    theta = np.linspace(0, 2*np.pi, 800)
    for t in theta:
        if plane == 'xy':
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))
            z = cz
        elif plane == 'yz':
            x = cx
            y = int(cy + r * np.cos(t))
            z = int(cz + r * np.sin(t))
        elif plane == 'xz':
            x = int(cx + r * np.cos(t))
            y = cy
            z = int(cz + r * np.sin(t))
        for dx0 in range(-3, 4):
            for dy0 in range(-3, 4):
                for dz0 in range(-3, 4):
                    i = x + dx0
                    j = y + dy0
                    k = z + dz0
                    if 0 <= i < size and 0 <= j < size and 0 <= k < size:
                        d2 = dx0**2 + dy0**2 + dz0**2
                        field[i,j,k] += np.exp(-d2 / (2 * sigma**2))



def strong_force_test(config_overrides=None):
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

    title="UMH Gauge Coupling (Strong)"
    file_root="UMH_Gauge_Coupling"
    file_sub="UMH_Strong"
    file_hdr="UMH_Gauge_Coupling_Strong"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    
    relaxation_rate = 0.3
    sigma = 2.5  # Gaussian blob radius

    # --------------------------
    # Parameters
    # --------------------------
    dx = lattice_spacing / size

    # --------------------------
    # Initialization
    # --------------------------
    psi1 = np.zeros((size, size, size))
    psi2 = np.zeros((size, size, size))
    psi3 = np.zeros((size, size, size))

    cx, cy, cz = size // 2, size // 2, size // 2
    r = size // 4

    draw_gaussian_loop(psi1,size, cx, cy, cz, r, 'xy', sigma=sigma)
    draw_gaussian_loop(psi2,size, cx, cy, cz, r, 'yz', sigma=sigma)
    draw_gaussian_loop(psi3,size, cx, cy, cz, r, 'xz', sigma=sigma)

    pinned1 = psi1 > 0.1
    pinned2 = psi2 > 0.1
    pinned3 = psi3 > 0.1

    # --------------------------
    # Run Relaxation
    # --------------------------
    psi1, e1, T1_xx, T1_yy, T1_zz, T1_xy = relax_field(psi1, pinned1, relaxation_rate, dx, Tu, steps)
    psi2, e2, T2_xx, T2_yy, T2_zz, T2_xy = relax_field(psi2, pinned2, relaxation_rate, dx, Tu, steps)
    psi3, e3, T3_xx, T3_yy, T3_zz, T3_xy = relax_field(psi3, pinned3, relaxation_rate, dx, Tu, steps)

    # --------------------------
    # Interaction Energy
    # --------------------------
    interaction_energy = np.sum((psi1 * psi2 * psi3)**2) * dx**3

    # --------------------------
    # Coupling Constant
    # --------------------------
    total_energy = e1[-1] + e2[-1] + e3[-1] + interaction_energy
    g2 = total_energy / (2 * np.pi * Tu * lattice_spacing)

    print(f"{title}: Estimated Coupling Constant (g^2): {g2:.6f}.")
    print(f"{title}: g^2 / (4π) = {g2 / (4*np.pi):.6f}.")

    with open(f"{file_path}_Coupling_Constants.txt", "w") as f:
        f.write(f"{title}: Estimated Coupling Constant (g^2): {g2:.6f}\n")
        f.write(f"{title}: g^2 / (4π) = {g2 / (4*np.pi):.6f}\n")


    # --------------------------
    # Visualization
    # --------------------------
    cz = size // 2

    plt.figure()
    plt.imshow(psi1[:, :, cz], cmap='viridis')
    plt.title(f"{title}: ψ1 Field Slice (XY plane)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Psi1_xy.png")
    plt.close()

    plt.figure()
    plt.imshow(psi2[:, cy, :], cmap='viridis')
    plt.title(f"{title}: ψ2 Field Slice (YZ plane)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Psi2_yz.png")
    plt.close()

    plt.figure()
    plt.imshow(psi3[:, :, cz], cmap='viridis')
    plt.title(f"{title}: ψ3 Field Slice (XZ plane)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Psi3_xz.png")
    plt.close()

    plt.figure()
    plt.plot(e1, label="ψ1", color="red")
    plt.plot(e2, label="ψ2", color="green")
    plt.plot(e3, label="ψ3", color="blue")
    plt.title(f"{title}: Strain Energy Convergence")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Energy_Convergence.png")
    plt.close()


    print(f"✅ Finished Test: {title} Validated.")



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    strong_force_test()