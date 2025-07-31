"""
UMH_Simulation_Configurable.py

Author: Andrew Dodge  
Date: June 2025

Description:
Simulates a 3D strain field evolution based on the Ultronic Medium Hypothesis (UMH),
including nonlinear solitonic dynamics and scale-invariant noise initial conditions.
Also computes and saves the velocity field for energy and mass validation.

Inputs:
- Optional config JSON for overriding default parameters.

Outputs:
- Strain field time slices (.npy)
- Final strain field (UMH_SimData.npy)
- Final velocity field (UMH_SimData_Velocity.npy)
- Output folder: ./Output/UMH_SimData/
"""

import numpy as np
from numba import njit, prange
import os
import sys
import json

try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp



def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "LATTICE_SIZE": 768, #Disk Space Usage Issues. 512
        "TIME_STEPS": 1600, #500, 1600, 3000
        #"DT": 0.1,

        "NUMBER_SNAPSHOTS":50, #250 #Capture every 3rd frame.  Disk Space Usage Issues.  This should equate to 100gb used with 450-150 Lattice at np.float64.

        "DTYPE":np.float64,

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.

        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)

        #"WAVE_SPEED":1.0, #Speed of light equivalent, we are using 1.0 here for simulation.

        "PML_THICKNESS":40, #Was 30, 7-26-2025.
        "PML_ABORPTION":0.15, #0.07, was 0.15 7-29-2025.
        "PML_POWER": 2, #Was 4, 7-26-2025.

        "power_spectrum_index": 0.96, #0.94, 7-29-2025.
        "k_min": 0.0002, #0.002, 0.005,0.001  #Was 0.0005, 7-26-2025. #Was 0.0002, 0.005 7-29-2025.
        "k_max": 0.60, #0.70, was 0.45  0.60, 0.70 - 7-29-2025.
        "k_damp": 0.55, #Was 0.45, 0.50 7-26-2025, 0.45 7-26-2025.
        "damping_exp": 2.0, #Was 2.2, 7-26-2025. Was 2.0, 2.4 7-29-2025.
        "normalization": 4.2e-6, #1e-6, 5e-6  #Was 1e-6, 7-26-2025. #Was 2e-6, 4e-6, 7-29-2025.
        "k_cutoff": 0.14, #0.20  #Was 0.20, 0.25 7-26-2025, 0.18 7-26-2025.
        #"k_peak": 0.15,
        #"bao_amp": 0.0, #DO NOT USE. UMH Does not require this as this is dependent on an expanding Universe.
        "lowk_amp": 0.24, #0.3, #Was 0.0, 7-26-2025, was 0.02 - 7-29-2025, was 0.05, 0.18, 0.2 - 7-29-2025.
        "suppression_k": 0.0014, #0.01, was 0.004, 0.0018 - 7-29-2025.

        "k_break": 0.02, # Added 7-29-2025.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


@njit(parallel=True, cache=True, fastmath=True)
def umh_transfer_function_flat(K_flat, suppression_k, lowk_amp):
    T = np.empty_like(K_flat)
    for i in prange(K_flat.size):
        lowk = 1 - lowk_amp * np.exp(-(K_flat[i] / suppression_k)**2)
        T[i] = lowk
    return T


def tanh_cutoff_power_spectrum(k, config):
    k_break = config.get("k_break", 0.03)
    k_damp = config.get("k_damp", 0.52)
    damping_exp = config.get("damping_exp", 2.0)
    tilt = config.get("tilt", 0.96)

    # Smooth spectral index transition via tanh
    spectral_index = -3 + (3 - tilt) * 0.5 * (1 + np.tanh((k_break - k) / (0.1 * k_break)))
    spectrum = k**spectral_index

    # High-k exponential damping
    spectrum *= np.exp(-(k / k_damp)**damping_exp)

    return spectrum


def generate_scale_invariant_noise(shape, config):
    kx = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kz = np.fft.fftfreq(shape[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[K == 0] = config["k_min"] / 10

    spectrum = (1.0 / K**(config["power_spectrum_index"] / 2)) * np.exp(-(K / config["k_damp"])**config["damping_exp"])
    #spectrum = (1.0 / K**(tilt / 2)) * np.exp(-(K / k_damp)**damping_exp) * (1 + lowk_amp * np.exp(-(K / suppression_k)**2))
    #spectrum = tanh_cutoff_power_spectrum(K, config)

    spectrum[~((K >= config["k_min"]) & (K <= config["k_max"]))] = 0
    spectrum *= config["normalization"]

    #T = umh_transfer_function_flat(K.ravel(), config["suppression_k"], config["lowk_amp"]).reshape(K.shape)
    #spectrum *= T * config["normalization"]

    random_field = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    return np.fft.ifftn(random_field * spectrum).real.astype(np.float32)

def lowpass_filter(psi, k_cutoff):
    psi_k = np.fft.fftn(psi)
    kx = np.fft.fftfreq(psi.shape[0])
    ky = np.fft.fftfreq(psi.shape[1])
    kz = np.fft.fftfreq(psi.shape[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    psi_k[K > k_cutoff] = 0
    return np.fft.ifftn(psi_k).real.astype(np.float32)


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["LATTICE_SIZE"] # Memory-safe grid size.
    steps = config["TIME_STEPS"]
    #dt = config["DT"] # CFL-safe timestep; lower if unstable.
    dx = config["LATTICE_SPACING"]

    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    #v = config["WAVE_SPEED"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]
    pml_power = config["PML_POWER"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    dtype=config["DTYPE"]
    
    outdir = config["OUTPUT_FOLDER"]

    title="UMH SimData"
    file_hdr="UMH_SimData"
  
    print(f"✅ Starting Test: {title} Generation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")


    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    print(f"{title} Calculated Wave Speed Constant C = {v}")

    # Derived Gravity from Density and Pressure.
    G = dx ** 2 / Tu
    print(f"{title} Calculated Gravitational Constant G = {G}") #Not Used in this logic.

    # Then calculate λ as before
    # Nonlinear Laplacian with Cubic Strain Coupling
    lambda_coeff = dx ** 2 / G
    print(f"{title} Calculated Nonlinear Coupling Coefficient λ = {lambda_coeff}") #Not Used in this logic.


    # Safety Factor Adjust GridSize.
    safety_factor_base=0.5 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor_base, min_factor=min_factor, max_factor=max_factor)
 
    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor) #0.25 0.14
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")

    shape = (size, size, size)

    psi = generate_scale_invariant_noise(shape, config)
    psi = lowpass_filter(psi, config["k_cutoff"])
    psi_prev = psi.copy()
    psi_next = np.zeros_like(psi)

    # Create PML Absorption Mask (adjust thickness/absorption if needed)
    absorption_mask = create_absorption_mask(shape, pml_thickness, pml_absorption,power=pml_power, dtype=dtype)

    v2 = v ** 2
    

    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    for step in range(steps):
        # ✅ Use your Stencil 27 PML update
        update_wave_27_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask)
        #psi_prev, psi = psi, psi_next.copy()
        # Swap references — this is O(1) and avoids .copy()
        psi_prev, psi, psi_next = psi, psi_next, psi_prev

        if step in snapshot_steps: # or step == steps - 1

            np.save(f"{file_path}_{step:03}.npy", psi.astype(dtype))
            print(f"{title} Time step {step}/{steps}, complete.")

    velocity = (psi - psi_prev) / dt
    np.save(f"{file_path}.npy", psi.astype(dtype))
    np.save(f"{file_path}_Prev.npy", psi_prev.astype(dtype))
    np.save(f"{file_path}_Velocity.npy", velocity.astype(dtype))
    

    # ===============================
    # ✅ Save Parameter Summary to JSON
    # ===============================
    param_summary = {
        "GridSize": size,
        "TimeSteps": steps,
        "dt": dt,
        "safety_factor_base": safety_factor_base,
        "safety_factor": safety_factor,
        "dx": dx,
        "WaveSpeed_c": v,
        "MediumDensity_T": Tu,
        "MediumPressure_rho": rho_u,
        "GravitationalConstant_G": G, 
        "CouplingCoefficient_lambda": lambda_coeff,  # Simplifies to Tu 
        "PML_THICKNESS": config["PML_THICKNESS"],
        "PML_ABORPTION": config["PML_ABORPTION"],
        "PML_POWER": config["PML_POWER"],
        "power_spectrum_index": config["power_spectrum_index"],
        "k_min": config["k_min"],
        "k_max": config["k_max"],
        "k_damp": config["k_damp"],
        "damping_exp": config["damping_exp"],
        "normalization": config["normalization"],
        "k_cutoff": config["k_cutoff"],
        #"k_peak": config["k_peak"],
        #"bao_amp": config["bao_amp"],
        "lowk_amp": config["lowk_amp"],
        "suppression_k": config["suppression_k"],
        "NUMBER_SNAPSHOTS": config["NUMBER_SNAPSHOTS"],
        "DTYPE": str(config["DTYPE"])
    }

    param_path=f"{file_path}_Parameters.json"
    with open(param_path, "w") as f:
        json.dump(param_summary, f, indent=4)

    print(f"✅ {title}: Parameter file saved to '{param_path}'")


    print(f"✅ Finished Test: {title} Generation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)
