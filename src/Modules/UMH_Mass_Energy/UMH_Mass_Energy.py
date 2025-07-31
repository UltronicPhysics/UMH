"""
UMH_Mass_Energy.py

Author: Andrew Dodge
Date: July 2025

Description:
Optimized mass-energy equivalence and stress-energy tensor validation using Numba-accelerated computation.
Verifies Einstein tensor curvature arises from strain energy.

Inputs:
- UMH_SimData.npy
- UMH_SimData_Velocity.npy

Outputs:
- Validation plots and Ricci, Einstein, T00, G_check fields as .npy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from numba import njit, prange
from scipy.ndimage import center_of_mass


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor, compute_gradient_components
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor, compute_gradient_components


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "INPUT_SIMDATA": os.path.join(base, "Output", "UMH_SimData", "UMH_SimData.npy"),
        "INPUT_SIMDATA_VELOCITY": os.path.join(base, "Output", "UMH_SimData", "UMH_SimData_Velocity.npy"),

        "NUM_STEPS": 10,

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.

        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
                
        "PML_THICKNESS":30, #was 20
        "PML_ABORPTION":0.07,

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    steps = config["NUM_STEPS"]

    dx=config["LATTICE_SPACING"]

    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]
    
    outdir = config["OUTPUT_FOLDER"]

    title="UMH Mass Energy"
    file_hdr="UMH_Mass_Energy"
  
    print(f"✅ Starting Test: {title} and Tensor Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")


    strain_now = np.load(config["INPUT_SIMDATA"])
    velocity_now = np.load(config["INPUT_SIMDATA_VELOCITY"])

    print(f"{title} Sim Data and Sim Data Velocity Loaded.")

    size = strain_now.shape[0] #Get Size of dimension.

    vmin=0 #Min Color Scale Plot.
    vmax=1 #Max Color Scale Plot.

    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    print(f"{title} Calculated Wave Speed Constant C = {v}")

    # Derived Gravity from Density and Pressure.
    G = dx ** 2 / Tu
    print(f"{title} Calculated Gravitational Constant G = {G}")

    # Then calculate λ as before
    # Nonlinear Laplacian with Cubic Strain Coupling
    lambda_coeff = dx ** 2 / G
    print(f"{title} Calculated Nonlinear Coupling Coefficient λ = {lambda_coeff}")

    # Define UMH Physical Parameters (User Adjustable)
    #Tu = 1.0       # Medium Tension (normalized or in units)
    #rho_u = 1.0    # Medium Density
    #dx = 1.0       # Grid spacing (distance between nodes)
    #dt = 1.0       # Time step (to be set by CFL if wave dynamics are used)

    #dx = dt = Tu = rho_u = 1.0

    v2 = v**2

    safety_factor=0.5

    # Safety Factor Adjust GridSize.
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    print(f"{title}: auto_cfl_clamp, Min:{min_factor}, Max:{max_factor}")
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)

    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor)   #0.14?
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")


    size = strain_now.shape[0]

    # Apply PML Absorption Mask
    absorption_mask = create_absorption_mask(strain_now.shape, pml_thickness, pml_absorption, dtype=dtype)

    # Relax strain field dynamically with PML applied
    psi = strain_now.copy()
    psi_prev = strain_now.copy()
    psi_next = np.zeros_like(psi)

    for step in range(steps):
        update_wave_27_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask)
        psi_prev, psi = psi, psi_next.copy()

    strain_now = psi.copy()
    velocity_now *= absorption_mask

    # Use update_wave_27_wPML with zero initial velocity to compute Laplacian
    # Equivalent to measuring second derivative
    lap_init = np.zeros_like(strain_now)
    zero_velocity = np.zeros_like(strain_now)
    update_wave_27_wPML(strain_now, strain_now, lap_init, v2, 1.0, dx, np.ones_like(strain_now))
    
    #nonlinear.
    nonlinear_term = lambda_coeff * strain_now ** 3
    lap_strain = (lap_init - strain_now) + nonlinear_term  # Laplacian + Nonlinear source term

    #linear.
    #lap_strain = (lap_init - strain_now)  # Captures the (dt² * v² * ∇²ϕ) term

    # Ricci Tensor Calculation using Laplacian estimated from stencil
    Ricci = compute_ricci_tensor(strain_now, lap_strain, dx)
    Ricci_scalar = np.einsum('ii...->...', Ricci)

    Einstein_tensor = Ricci.copy()
    for i in range(3):
        Einstein_tensor[i, i] -= 0.5 * Ricci_scalar

    gx, gy, gz = compute_gradient_components(strain_now, dx)
    grad_sq = gx**2 + gy**2 + gz**2
    T00 = 0.5 * grad_sq + 0.5 * velocity_now**2
    G00 = Ricci_scalar

    denom = 8 * np.pi * T00
    #G_check = np.divide(G00, denom, out=np.zeros_like(G00), where=denom != 0)
    epsilon = 1e-20  # or tune this value
    mask = denom > epsilon
    G_check = np.zeros_like(G00)
    G_check[mask] = G00[mask] / denom[mask]


    total_energy = np.sum(T00) * dx**3
    inferred_mass = total_energy / v**2
    print(f"{title} Total Integrated Energy: {total_energy:.10e}")
    print(f"{title} Inferred Mass via E=mc^2: {inferred_mass:.10e}")

    print(f"{title} Strain max:", np.max(np.abs(strain_now)))
    print(f"{title} Velocity max:", np.max(np.abs(velocity_now)))
    print(f"{title} Potential max:", np.max(0.5 * grad_sq))
    print(f"{title} Kinetic max:", np.max(0.5 * velocity_now**2))
    print(f"{title} Total energy:", total_energy)
    print(f"{title} Mass:", inferred_mass)

    np.save(f"{file_path}_Ricci_tensor.npy", Ricci)
    np.save(f"{file_path}_Ricci_scalar.npy", Ricci_scalar)
    np.save(f"{file_path}_Einstein_tensor.npy", Einstein_tensor)
    np.save(f"{file_path}_EnergyField_T00.npy", T00)
    np.save(f"{file_path}_G_check.npy", G_check)

    with open(f"{file_path}_Results.csv", "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Max strain,{np.max(np.abs(strain_now)):.3e}\n")
        f.write(f"Max velocity,{np.max(np.abs(velocity_now)):.3e}\n")
        f.write(f"Max potential energy,{np.max(0.5 * grad_sq):.3e}\n")
        f.write(f"Max kinetic energy,{np.max(0.5 * velocity_now**2):.3e}\n")
        f.write(f"Total Integrated Energy,{total_energy:.10e}\n")
        f.write(f"Inferred Mass via E=mc^2,{inferred_mass:.10e}\n")
        f.write(f"PML Applied,Thickness={pml_thickness}\n")
        f.write(f"PML Applied,Absorption Coeff={pml_absorption}\n")
        f.write(f"Relax Steps,{steps}\n")

    # Downsample to a manageable number of points (e.g., 100,000)
    num_points = 100_000
    T00_flat = (8 * np.pi * T00).flatten()
    G00_flat = G00.flatten()

    if len(T00_flat) > num_points:
        indices = np.random.choice(len(T00_flat), num_points, replace=False)
        T00_sample = T00_flat[indices]
        G00_sample = G00_flat[indices]
    else:
        T00_sample = T00_flat
        G00_sample = G00_flat

    plt.figure(figsize=(6, 6))
    #plt.scatter((8 * np.pi * T00).flatten(), G00.flatten(), alpha=0.02, color='blue', s=1)  #Takes a long time, downsample instead.
    plt.scatter(T00_sample, G00_sample, alpha=0.05, color='blue', s=1)
    plt.xlabel('8π T00')
    plt.ylabel('Ricci Scalar (G00)')
    plt.title(f"{title} Stress-Energy Validation")
    plt.tight_layout()
    plt.savefig(f"{file_path}_StressEnergy_RicciScatter.png", dpi=dpi)
    plt.close()
   

    center = np.array([size//2]*3)
    grid = np.indices((size,size,size))
    radii = np.sqrt(np.sum((grid - center[:, None, None, None])**2, axis=0)).flatten()
    bins = np.logspace(0.5, np.log10(size/2), 40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    flat_G = G_check.flatten()

    #ratio_mean = [
    #    flat_G[(radii >= low) & (radii < high)].mean() if np.any((radii >= low) & (radii < high)) else np.nan
    #    for low, high in zip(bins[:-1], bins[1:])
    #]
    residual = G00 - denom
    flat_residual = residual.flatten()
    residual_mean = [
        flat_residual[(radii >= low) & (radii < high)].mean() if np.any((radii >= low) & (radii < high)) else np.nan
        for low, high in zip(bins[:-1], bins[1:])
    ]


    plt.figure(figsize=(8,5))
    plt.semilogx(bin_centers, residual_mean, 'o-', label='Residual (G₀₀ - 8πT₀₀)')
    plt.axhline(0.0, color='r', linestyle='--', label='Perfect Match (Residual = 0)')
    plt.xlabel('Radius')
    plt.ylabel('Residual Value')
    plt.title(f"{title} Stress-Energy Tensor Residual (Radial)")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Tensor_Residual_Radial.png", dpi=dpi)
    plt.close()


    mid_slice = T00[size // 2]
    vmin = np.percentile(mid_slice, 1)     # or 0 for strict min
    vmax = np.percentile(mid_slice, 99.5)  # to ignore outlier peaks

    plt.imshow(mid_slice, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"{title} T₀₀ Midplane Slice")
    plt.tight_layout()
    plt.savefig(f"{file_path}_T00_midplane_slice.png", dpi=dpi)
    plt.close()


    X, Y, Z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
    center = center_of_mass(T00)
    r_grid = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    radii = np.linspace(0, r_grid.max(), 100)
    energy_profile = np.zeros_like(radii)

    for i, r in enumerate(radii):
        mask = (r_grid < r)
        energy_profile[i] = np.sum(T00[mask]) * dx**3

    plt.plot(radii, energy_profile)
    plt.xlabel("Radius")
    plt.ylabel("Cumulative Energy")
    plt.title(f"{title} Cumulative Profile")
    plt.grid()
    plt.savefig(f"{file_path}_Cumulative.png", dpi=dpi)
    plt.close()
        
    print(f"✅ Finished Test: {title} and Tensor Validation.")






if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)
