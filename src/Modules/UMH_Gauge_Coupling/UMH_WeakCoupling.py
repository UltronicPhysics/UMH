"""
UMH_Gauge_Coupling_Weak.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Gauge Coupling Weak.

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
        "LATTICE_SIZE": 500, #500,300
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
# Relaxation Function
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def relax_field(field, pinned_mask, relaxation_rate, dx, T_u, num_steps):
    N = field.shape[0]
    energy_history = np.zeros(num_steps)

    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_z = np.zeros_like(field)

    T_xx = np.zeros_like(field)
    T_yy = np.zeros_like(field)
    T_zz = np.zeros_like(field)
    T_xy = np.zeros_like(field)
    T_xz = np.zeros_like(field)
    T_yz = np.zeros_like(field)

    for step in range(num_steps):
        new_field = np.copy(field)
        for i in prange(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    if not pinned_mask[i, j, k]:
                        neighbor_sum = (
                            field[i+1, j, k] + field[i-1, j, k] +
                            field[i, j+1, k] + field[i, j-1, k] +
                            field[i, j, k+1] + field[i, j, k-1]
                        )
                        neighbor_avg = neighbor_sum / 6.0
                        new_field[i, j, k] = (1 - relaxation_rate) * field[i, j, k] + relaxation_rate * neighbor_avg
        field = new_field

        for i in prange(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    grad_x[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
                    grad_y[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dx)
                    grad_z[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dx)

        strain_energy = 0.5 * T_u * (grad_x**2 + grad_y**2 + grad_z**2)
        total_energy = np.sum(strain_energy[1:-1, 1:-1, 1:-1]) * dx**3
        energy_history[step] = total_energy

    T_xx = T_u * grad_x * grad_x
    T_yy = T_u * grad_y * grad_y
    T_zz = T_u * grad_z * grad_z
    T_xy = T_u * grad_x * grad_y
    T_xz = T_u * grad_x * grad_z
    T_yz = T_u * grad_y * grad_z

    return field, energy_history, T_xx, T_yy, T_zz, T_xy, T_xz, T_yz


def weak_force_test(config_overrides=None):
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

    title="UMH Gauge Coupling (Weak)"
    file_root="UMH_Gauge_Coupling"
    file_sub="UMH_Weak"
    file_hdr="UMH_Gauge_Coupling_Weak"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    relaxation_rate = 0.3
    dx = lattice_spacing / size
    
    # --------------------------
    # Initialize Lattice
    # --------------------------
    field = np.zeros((size, size, size))
    cx, cy, cz = size // 2, size // 2, size // 2
    r = size // 4
    theta = np.linspace(0, 2*np.pi, 800)

    for t in theta:
        x = int(cx + r * np.cos(t))
        y = int(cy + r * np.sin(t))
        z = cz
        if 0 <= x < size and 0 <= y < size:
            field[x, y, z] = 1.0

    for t in theta:
        x = int(cx + r * np.cos(t))
        y = cy
        z = int(cz + r * np.sin(t))
        if 0 <= x < size and 0 <= z < size:
            field[x, y, z] = 1.0

    pinned_mask = (field > 0)


    # --------------------------
    # Run Relaxation
    # --------------------------

    field, energy_history, T_xx, T_yy, T_zz, T_xy, T_xz, T_yz = relax_field(
        field, pinned_mask, relaxation_rate, dx, Tu, steps
    )

    for step, energy in enumerate(energy_history):
        if step % 20 == 0 or step == steps - 1:
            print(f"{title}: [relax_field] Step: {step+1}: Total strain energy = {energy:.6f}.")

    with open(f"{file_path}_Energy_Log.txt", "w") as f:
        for step, energy in enumerate(energy_history):
            if step % 20 == 0 or step == steps - 1:
                line = f"{title}: [relax_field] Step: {step+1}: Total strain energy = {energy:.6f}.\n"
                print(line.strip())  # Optional: also print to console
                f.write(line)


    # --------------------------
    # Coupling Constant Estimate
    # --------------------------
    print(f"\n")

    final_energy = energy_history[-1]
    coupling_constant = final_energy / (2 * np.pi * Tu * lattice_spacing)
    print(f"{title}: Estimated weak coupling constant (g^2): {coupling_constant:.6f}.")
    print(f"{title}: g^2 / (4π) = {coupling_constant / (4*np.pi):.6f}.")

    with open(f"{file_path}_Coupling_Constants.txt", "w") as f:
        f.write(f"{title}: Estimated weak coupling constant (g^2): {coupling_constant:.6f}\n")
        f.write(f"{title}: g^2 / (4π) = {coupling_constant / (4*np.pi):.6f}\n")

    # --------------------------
    # Save Tensors
    # --------------------------
    np.save(f"{file_path}_Tensor_T_xx.npy", T_xx)
    np.save(f"{file_path}_Tensor_T_yy.npy", T_yy)
    np.save(f"{file_path}_Tensor_T_zz.npy", T_zz)
    np.save(f"{file_path}_Tensor_T_xy.npy", T_xy)
    np.save(f"{file_path}_Tensor_T_xz.npy", T_xz)
    np.save(f"{file_path}_Tensor_T_yz.npy", T_yz)

    # --------------------------
    # Plots
    # --------------------------
    plt.figure()
    plt.plot(energy_history, color='orange')
    plt.title(f"{title}: Strain Energy Convergence During Relaxation")
    plt.xlabel("Step")
    plt.ylabel("Total Strain Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Energy_Convergence.png")
    plt.close()

    plt.figure()
    plt.imshow(field[:, :, cz], cmap='viridis')
    plt.title(f"{title}: Field Slice at Center Z (XY loop)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Field_Slice_xy.png")
    plt.close()

    plt.figure()
    plt.imshow(field[:, cy, :], cmap='viridis')
    plt.title(f"{title}: Field Slice at Center Y (XZ loop)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Field_Slice_xz.png")
    plt.close()

    plt.figure()
    plt.imshow(T_xx[:, :, cz], cmap='plasma')
    plt.title(f"{title}: Stress Tensor T_xx Slice at Z")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Tensor_T_xx.png")
    plt.close()

    plt.figure()
    plt.imshow(T_xy[:, :, cz], cmap='plasma')
    plt.title(f"{title}: Stress Tensor T_xy Slice at Z")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Tensor_T_xy.png")
    plt.close()

    plt.figure()
    plt.imshow(T_zz[:, :, cz], cmap='plasma')
    plt.title(f"{title}: Stress Tensor T_zz Slice at Z")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Tensor_T_zz.png")
    plt.close()


    print(f"✅ Finished Test: {title} Validated.")



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    weak_force_test()