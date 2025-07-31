"""
UMH_Gauge_Symmetry_SU3.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Gauge Symmetry.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces
"""

import numpy as np
import numba
import os
import sys
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import matplotlib.colors as mcolors


from numba import njit, prange
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter1d
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates
from scipy.fft import fft, fftfreq


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components
    from ..UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_SU3_Constraints import enforce_su3_phase_constraint, apply_center_locking, generate_trefoil_points, initialize_su3_trefoil_knot, project_to_su3, initialize_su3_full_grid, apply_radial_damping, renormalize_su3_fields, compute_dynamic_center, apply_su3_constraint_hard, apply_localized_kick, su3_symmetric_normalize, compute_stress_energy_tensor, sample_ricci_angular_spread, safe_sample_ricci_angular_spread, compute_ricci_scalar, compute_stress_tensor_components
    from ..UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy, save_csv_slice, save_csv_3d, trim_edges, downsample, center_crop, zoom_slice_for_display, crop_field, save_zslice_image, compute_radial_profile, radial_plot, get_iso_value, save_isosurface_3d, low_pass_filter_3d, Vacuum_Einstein_Tensor, Soliton_Einstein_Tensor, Stress_Energy_Tensor, EinsteinTensor_Stats
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components  
    from UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_SU3_Constraints import enforce_su3_phase_constraint, apply_center_locking, generate_trefoil_points, initialize_su3_trefoil_knot, project_to_su3, initialize_su3_full_grid, apply_radial_damping, renormalize_su3_fields, compute_dynamic_center, apply_su3_constraint_hard, apply_localized_kick, su3_symmetric_normalize, compute_stress_energy_tensor, sample_ricci_angular_spread, safe_sample_ricci_angular_spread, compute_ricci_scalar, compute_stress_tensor_components
    from UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy, save_csv_slice, save_csv_3d, trim_edges, downsample, center_crop, zoom_slice_for_display, crop_field, save_zslice_image, compute_radial_profile, radial_plot, get_iso_value, save_isosurface_3d, low_pass_filter_3d, Vacuum_Einstein_Tensor, Soliton_Einstein_Tensor, Stress_Energy_Tensor, EinsteinTensor_Stats



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

       
        "INITIAL_CONSTRAINT_STRENGTH":1.0,

        "VELOCITY_DAMPING":0.98,



        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def compute_scaled_simulation_params(Nx, Ny, Nz):
    base_N = 300  # Reference grid size
    scale_factor = Nx / base_N
    base_kick_strength=0.005 #0.00333

    core_radius = max(int(Nx / 24), 3)  # Keep the kicked region the same *relative* to grid size
    # Compute base region volume for scaling (optional but recommended for robust scaling)
    base_core_radius = 200 / 24
    base_volume = (4/3) * np.pi * (base_core_radius ** 3)
    kick_volume = (4/3) * np.pi * (core_radius ** 3)
    kick_strength_scaled = base_kick_strength * (base_volume / kick_volume)

    params = {
        "dt": 0.005 * scale_factor,               # Keep stable CFL condition
        "lock_radius": int(10 * scale_factor),    # Trefoil core lock radius
        "thickness": 4.0 * scale_factor,         # Trefoil field thickness
        "trefoil_scale": (Nx / 12.0),            # Trefoil curve radius
        "num_points": int(Nx * 1.0),
        "r_max": 0.25 * Nx,                       # For radial damping mask
        "constraint_strength": 2.0 * scale_factor,
        "absorb_thickness": int(12 * scale_factor),  # Absorption mask edge thickness
        "core_radius":core_radius,
        "kick_strength_scaled": kick_strength_scaled,

    }
    return params


def run(config_overrides=None):
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
    
    INITIAL_CONSTRAINT_STRENGTH=config["INITIAL_CONSTRAINT_STRENGTH"]

    VELOCITY_DAMPING=config["VELOCITY_DAMPING"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Gauge Symmetry (SU3)"
    file_root="UMH_Gauge_Symmetry"
    file_sub="UMH_SU3"
    file_hdr="UMH_Gauge_Symmetry_SU3"
  
    print(f"✅ Starting Test: {title} Validated.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    
    radius = 15 #Used for ricci_angular_spread.

    params=compute_scaled_simulation_params(size,size,size)

    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    v2 = v**2 #Wave Speed squared.
    print(f"{title}: Calculated Wave Speed Constant C = {v}")

    #safety_factor=0.50 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07

    # Safety Factor Adjust GridSize.
    #min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    #print(f"{title}: auto_cfl_clamp, Min:{min_factor}, Max:{max_factor}")
    #safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)

    #dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor)
    #print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")    

    center = (size // 2, size // 2, size // 2)
    cx, cy, cz = center

    psi1_vac, psi2_vac, psi3_vac = initialize_su3_full_grid(size, size, size, phase_amplitude=0.02, base_amplitude=0.0)  #Create blank for the iso. cx,cy,cz?

    Vacuum_Einstein_Tensor(file_path, title, psi1_vac, psi2_vac, psi3_vac, dx, cz, dpi=dpi)


    dt=params["dt"]
    lock_radius=params["lock_radius"]
    thickness=params["thickness"]
    trefoil_scale=params["trefoil_scale"]
    r_max=params["r_max"]
    constraint_strength=params["constraint_strength"]
    absorb_thickness=params["absorb_thickness"]
    kick_strength_scaled=params["kick_strength_scaled"]
    core_radius=params["core_radius"]
    num_points=params["num_points"]

    #num_points = 200
    knot_points = generate_trefoil_points(num_points, trefoil_scale, center)

    psi1, psi2, psi3 = initialize_su3_trefoil_knot(size, size, size, knot_points, grid_scale=0.02, amplitude=0.25)

    psi1 = gaussian_filter(psi1.real, sigma=1.0) + 1j * gaussian_filter(psi1.imag, sigma=1.0)
    psi2 = gaussian_filter(psi2.real, sigma=1.0) + 1j * gaussian_filter(psi2.imag, sigma=1.0)
    psi3 = gaussian_filter(psi3.real, sigma=1.0) + 1j * gaussian_filter(psi3.imag, sigma=1.0)

    velocity1 = np.zeros_like(psi1)
    velocity2 = np.zeros_like(psi2)
    velocity3 = np.zeros_like(psi3)

    #kick_strength=0.01
    #core_radius = int(size / 24)
    #apply_localized_kick(psi1, velocity1, cx, cy, cz, strength=kick_strength,core_radius=core_radius)

    X, Y, Z = np.meshgrid(np.arange(size),np.arange(size),np.arange(size),indexing='ij')
    # After psi1 initialized, create X,Y,Z grids as before
    #angle = np.arctan2(Y - cy, X - cx)
    #core_mask = np.abs(psi1) > (0.4 * np.max(np.abs(psi1)))
    #num_core = np.count_nonzero(core_mask)
    #kick_scale = kick_strength * (np.prod(psi1.shape) / num_core)
    #velocity1[core_mask] += psi1[core_mask] * (1j * kick_scale * angle[core_mask])

    apply_localized_kick(psi1, velocity1, cx, cy, cz, strength=kick_strength_scaled, core_radius=core_radius)



    # Apply radial tapering to avoid hard walls or periodic interference
    r_max = np.sqrt((cx/2)**2 + (cy/2)**2 + (cz/2)**2)
    apply_radial_damping(psi1, psi2, psi3, cx//2, cy//2, cz//2, r_max)

    max_psi1 = np.max(np.abs(psi1))
    max_psi2 = np.max(np.abs(psi2))
    max_psi3 = np.max(np.abs(psi3))
    print(f"[MONITOR] Max |Psi1|: {max_psi1:.6f}, |Psi2|: {max_psi2:.6f}, |Psi3|: {max_psi3:.6f}")

    absorption_mask = create_absorption_mask(psi1.shape, thickness=pml_thickness, absorption_coeff=pml_absorption,power=pml_power, dtype=dtype) #PML usage with gentle boundary damping to reduce relection.
   
    # --- UMH Gauge Symmetry SU(3) Simulation Settings ---
    HARD_CONSTRAINT_STEPS = 10           # Steps to hold hard constraint and phase lock
    PHASE_LOCK_EXTRA_STEPS = 0           # Extra steps for phase lock (optional)
    RECENTER_INTERVAL = 1                # Do recentering every 2 steps
    BLEND_START = 1.0                    # Initial soft blend value after hard constraint
    BLEND_END = 0.01                     # Final blend value for open evolution
    BLEND_FADE_STEPS = 10                # Number of steps to fade blend down to final
    BLEND_START_STEP = HARD_CONSTRAINT_STEPS
    HARD_TRIGGER=-1
    ADD_KICK_STEP = HARD_CONSTRAINT_STEPS + BLEND_FADE_STEPS +5
    STABLE=ADD_KICK_STEP+HARD_CONSTRAINT_STEPS+BLEND_FADE_STEPS
    
    #LAP_DIFF = np.linspace(2.0, 0.0, steps)
    LAP_DIFF = np.linspace(2.0, 0.2, BLEND_FADE_STEPS)

    epsilon = 1e-10  # Small constant to prevent division by zero

    snapshot_steps = set(np.linspace(STABLE, steps - 1, num_snapshots, dtype=int))

    max_workers=8

    constraint_error_list,core_field_list,total_energy_list=[],[],[]
    imagesNCF_3d,images_3d,images_sc=[],[],[]
    core_energy_total=[]
    core_trace,core_time_steps=[],[]
    R_samples_norm =[]
    
    print(f"{title}: Starting Loop.")
    for step in range(steps):
        print(f"[DEBUG] Process Step:{step}.")
        print(f"[MONITOR-0] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # --- Wave evolution and Laplacian ---
        lap1 = compute_laplacian_27point(psi1, dx)
        lap2 = compute_laplacian_27point(psi2, dx)
        lap3 = compute_laplacian_27point(psi3, dx)

        # --- Constraint force schedule ---
        if step < 20:
            constraint_strength = INITIAL_CONSTRAINT_STRENGTH
        else:
            constraint_strength = INITIAL_CONSTRAINT_STRENGTH * np.exp(-0.05 * (step - 20))

        norm = np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2
        target_norm = np.mean(norm)
        constraint_force = np.clip(constraint_strength * (norm - target_norm), -1e3, 1e3)
        ramp_factor = min(1.0, step / 10.0)
        constraint_force *= ramp_factor

        lap1 = np.clip(lap1, -5.0, 5.0)
        lap2 = np.clip(lap2, -5.0, 5.0)
        lap3 = np.clip(lap3, -5.0, 5.0)

        lap_mag1 = np.abs(lap1)
        lap_mag2 = np.abs(lap2)
        lap_mag3 = np.abs(lap3)

        lap_ramp = min(1.0, (step - 5) / 10.0)
        accel1 = lap_ramp * v**2 * lap1 - constraint_force * psi1
        accel2 = lap_ramp * v**2 * lap2 - constraint_force * psi2
        accel3 = lap_ramp * v**2 * lap3 - constraint_force * psi3

        # Kick after hard constraint
        if (step == ADD_KICK_STEP):
            print(f"[DEBUG] Velocity KICK_SCALE: {kick_strength_scaled}")
            apply_localized_kick(psi1, velocity1, cx, cy, cz, strength=kick_strength_scaled, core_radius=core_radius)
            BLEND_START_STEP=step+1

        #if (step > ADD_KICK_STEP and step < ADD_KICK_STEP+10):
        #    velocity1 *= 0.9
        #    velocity2 *= 0.9
        #    velocity3 *= 0.9

        #if step in range(HARD_CONSTRAINT_STEPS-1, HARD_CONSTRAINT_STEPS + int(BLEND_FADE_STEPS/2)):
        if step in range(BLEND_START_STEP, BLEND_START_STEP + BLEND_FADE_STEPS):
            lapdiff = LAP_DIFF[step - BLEND_START_STEP] #+ 1e-12
            #lapdiff = LAP_DIFF[step]
            print(f"[DEBUG] Damping blend:[{lapdiff}].")
            damping1 = np.exp(-lap_mag1 / lapdiff)
            damping2 = np.exp(-lap_mag2 / lapdiff)
            damping3 = np.exp(-lap_mag3 / lapdiff)
            lap1 *= damping1
            lap2 *= damping2
            lap3 *= damping3

        velocity1 += accel1 * dt
        velocity2 += accel2 * dt
        velocity3 += accel3 * dt
        velocity1 *= VELOCITY_DAMPING
        velocity2 *= VELOCITY_DAMPING
        velocity3 *= VELOCITY_DAMPING

        velocity1[np.abs(velocity1) < 1e-6] = 0.0
        velocity2[np.abs(velocity2) < 1e-6] = 0.0
        velocity3[np.abs(velocity3) < 1e-6] = 0.0

        velocity1 = np.clip(velocity1, -5.0, 5.0)
        velocity2 = np.clip(velocity2, -5.0, 5.0)
        velocity3 = np.clip(velocity3, -5.0, 5.0)

        print(f"[MONITOR-1] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # --- Phase recentering and core locking every RECENTER_INTERVAL steps ---
        if (step < HARD_CONSTRAINT_STEPS + PHASE_LOCK_EXTRA_STEPS and (step % RECENTER_INTERVAL == 0)): # or (step >= ADD_KICK_STEP and step < ADD_KICK_STEP+ HARD_CONSTRAINT_STEPS)
            print(f"[DEBUG] Recenter and Core Locking Applied")
            mask = np.zeros_like(psi1, dtype=bool)
            mask[cx-8:cx+8, cy-8:cy+8, cz-8:cz+8] = True
            phase_product = psi1[mask] * np.conj(psi2[mask]) * np.conj(psi3[mask])
            complex_mean = np.mean(phase_product)
            phase_center = np.angle(complex_mean)
            psi1[mask] *= np.exp(-1j * phase_center)
            psi2[mask] *= np.exp(-1j * phase_center)
            psi3[mask] *= np.exp(-1j * phase_center)
            # Core lock
            prev_cx, prev_cy, prev_cz = cx, cy, cz
            cx2, cy2, cz2 = compute_dynamic_center(psi1, psi2, psi3)
            psi1, psi2, psi3 = apply_center_locking(psi1, psi2, psi3, cx2, cy2, cz2, params["lock_radius"], constraint_strength)
            max_shift = 2
            cx = int(np.clip(cx2, prev_cx - max_shift, prev_cx + max_shift))
            cy = int(np.clip(cy2, prev_cy - max_shift, prev_cy + max_shift))
            cz = int(np.clip(cz2, prev_cz - max_shift, prev_cz + max_shift))

        # --- Absorbing boundaries ---
        velocity1 *= absorption_mask
        velocity2 *= absorption_mask
        velocity3 *= absorption_mask

        psi1 += velocity1 * dt
        psi2 += velocity2 * dt
        psi3 += velocity3 * dt

        print(f"[MONITOR-2] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # --- Smoothing: every 50 steps after 50 ---
        if step >= 50 and step % 50 == 0:
            for psi in [psi1, psi2, psi3]:
                magnitude = np.abs(psi)
                phase = np.angle(psi)
                smoothed_mag = gaussian_filter(magnitude, sigma=1.0)
                psi[:] = smoothed_mag * np.exp(1j * phase)

        # === SU(3) Constraint Application ===
        constraint_error = np.mean(np.abs(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2 - 1.0))

        print(f"[MONITOR-3] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        if step < HARD_CONSTRAINT_STEPS:
            print(f"[DEBUG] Initial Hard constraint applied")
            psi1, psi2, psi3 = apply_su3_constraint_hard(psi1, psi2, psi3)

        elif (step >= ADD_KICK_STEP) and (step < ADD_KICK_STEP+HARD_CONSTRAINT_STEPS):
            print("[DEBUG] ADD_KICK_STEP, Force Hard Phase Constraint")
            psi1, psi2, psi3 = apply_su3_constraint_hard(psi1, psi2, psi3)

            core_mask = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) < core_radius
            print("[DEBUG] Step", step, "Core mean norms:",np.mean(np.abs(psi1[core_mask])),np.mean(np.abs(psi2[core_mask])),np.mean(np.abs(psi3[core_mask])))
            print("[DEBUG] Channel variances:",np.var(np.abs(psi1[core_mask])),np.var(np.abs(psi2[core_mask])),np.var(np.abs(psi3[core_mask])))
            print("[DEBUG] Total norm deviation in core:", np.mean(np.abs(psi1[core_mask])**2 + np.abs(psi2[core_mask])**2 + np.abs(psi3[core_mask])**2) - 1)

        elif (step >= ADD_KICK_STEP+HARD_CONSTRAINT_STEPS) and (step < ADD_KICK_STEP+HARD_CONSTRAINT_STEPS+BLEND_FADE_STEPS):
            print("[DEBUG] ADD_KICK_STEP, Force Blend Phase Constraint")
            blend = BLEND_START - (BLEND_START - BLEND_END) * (step - HARD_CONSTRAINT_STEPS) / BLEND_FADE_STEPS
            psi1_proj, psi2_proj, psi3_proj = project_to_su3(psi1, psi2, psi3)
            psi1 = (1-blend)*psi1 + blend*psi1_proj
            psi2 = (1-blend)*psi2 + blend*psi2_proj
            psi3 = (1-blend)*psi3 + blend*psi3_proj

            if np.max(np.abs(psi1)) < 0.1 or np.max(np.abs(psi2)) < 0.1 or np.max(np.abs(psi3)) < 0.1:
                print(f"[DEBUG] Normalize Triggered")
                psi1, psi2, psi3=su3_symmetric_normalize(psi1, psi2, psi3)
            elif np.mean(np.abs(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2 - 1.0)) > 1.0:
                print(f"[DEBUG] ReNormalize Triggered 2")
                #psi1, psi2, psi3=su3_symmetric_normalize(psi1, psi2, psi3)
                psi1, psi2, psi3 = renormalize_su3_fields(psi1, psi2, psi3)

            if (step % 4) == 0:
                print(f"[DEBUG] Blend Hard constraint applied %4")
                psi1, psi2, psi3 = apply_su3_constraint_hard(psi1, psi2, psi3)
                HARD_TRIGGER=step

            core_mask = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) < core_radius
            print("[DEBUG] Step", step, "Core mean norms:",np.mean(np.abs(psi1[core_mask])),np.mean(np.abs(psi2[core_mask])),np.mean(np.abs(psi3[core_mask])))
            print("[DEBUG] Channel variances:",np.var(np.abs(psi1[core_mask])),np.var(np.abs(psi2[core_mask])),np.var(np.abs(psi3[core_mask])))
            print("[DEBUG] Total norm deviation in core:", np.mean(np.abs(psi1[core_mask])**2 + np.abs(psi2[core_mask])**2 + np.abs(psi3[core_mask])**2) - 1)

        elif step < HARD_CONSTRAINT_STEPS + BLEND_FADE_STEPS:
            blend = BLEND_START - (BLEND_START - BLEND_END) * (step - HARD_CONSTRAINT_STEPS) / BLEND_FADE_STEPS
            psi1_proj, psi2_proj, psi3_proj = project_to_su3(psi1, psi2, psi3)
            psi1 = (1-blend)*psi1 + blend*psi1_proj
            psi2 = (1-blend)*psi2 + blend*psi2_proj
            psi3 = (1-blend)*psi3 + blend*psi3_proj

            if np.max(np.abs(psi1)) < 0.1 or np.max(np.abs(psi2)) < 0.1 or np.max(np.abs(psi3)) < 0.1:
                print(f"[DEBUG] Normalize Triggered")
                psi1, psi2, psi3=su3_symmetric_normalize(psi1, psi2, psi3)

            if (step % 4) == 0:
                print(f"[DEBUG] Blend Hard constraint applied %4")
                psi1, psi2, psi3 = apply_su3_constraint_hard(psi1, psi2, psi3)
                HARD_TRIGGER=step

        elif constraint_error > 1.0:
            print(f"[WARN] Constraint error high ({constraint_error:.3f}) - hard constraint applied")
            psi1, psi2, psi3 = apply_su3_constraint_hard(psi1, psi2, psi3)
            HARD_TRIGGER=step

        else:
            #psi1_proj, psi2_proj, psi3_proj = project_to_su3_qr_mp(psi1, psi2, psi3, n_jobs=16)
            psi1_proj, psi2_proj, psi3_proj = project_to_su3(psi1, psi2, psi3)
            psi1 = (1-BLEND_END)*psi1 + BLEND_END*psi1_proj
            psi2 = (1-BLEND_END)*psi2 + BLEND_END*psi2_proj
            psi3 = (1-BLEND_END)*psi3 + BLEND_END*psi3_proj

            if np.max(np.abs(psi1)) < 0.1 or np.max(np.abs(psi2)) < 0.1 or np.max(np.abs(psi3)) < 0.1:
                print(f"[DEBUG] Normalize Triggered")
                psi1, psi2, psi3=su3_symmetric_normalize(psi1, psi2, psi3)


        print(f"[MONITOR-4] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        if False: #No Longer Needed.
            # --- Norm clipping: only if >2x mean(norm), not every step ---
            norm = np.sqrt(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2)
            threshold = 2.0 * np.mean(norm)
            mask = norm > threshold
            if np.any(mask) and (step % 10 == 0):
                psi1[mask] /= norm[mask]
                psi2[mask] /= norm[mask]
                psi3[mask] /= norm[mask]
                print(f"[DEBUG] Normalization triggered. Exceeded: {np.count_nonzero(mask)}")


        print(f"[MONITOR-5] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # --- Radial damping: ends with phase lock, not hard constraint ---
        if step < HARD_CONSTRAINT_STEPS + PHASE_LOCK_EXTRA_STEPS:
            print(f"[DEBUG] Apply Radial Damping")
            apply_radial_damping(psi1, psi2, psi3, *center, r_max=0.9 * (cx // 2))

        #if step >= ADD_KICK_STEP and step < ADD_KICK_STEP+ HARD_CONSTRAINT_STEPS:
        #    print(f"[DEBUG] Apply Radial Damping")
        #    apply_radial_damping(psi1, psi2, psi3, *center, r_max=0.9 * (cx // 2))
        
        print(f"[MONITOR-6] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # After all updates (kicks, constraints, psi1/2/3 += velocity*dt, etc.)
        if True: #(step % 5 == 0) or (step == steps-1) or (step==HARD_TRIGGER) or (step==ADD_KICK_STEP) or step in range(BLEND_START_STEP, BLEND_START_STEP + BLEND_FADE_STEPS):
            print(f"[DEBUG] Renormalizing")
            psi1, psi2, psi3 = renormalize_su3_fields(psi1, psi2, psi3)

        print(f"[MONITOR-7] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")

        # --- Monitoring ---
        constraint_error = np.mean(np.abs(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2 - 1.0))
        if constraint_error > 1.0:
            print(f"[WARN] Constraint error monitor above (1.0): ({constraint_error:.3f})")

        phase_field = np.angle(psi1 + psi2 + psi3)
        print(f"[MONITOR] Norm Stats: Constraint Residual:{constraint_error:.3f}, min={np.min(norm):.4f}, max={np.max(norm):.4f}, mean={np.mean(norm):.4f}, phase min={phase_field.min()}, max={phase_field.max()}, mean={phase_field.mean()}")
        print(f"[MONITOR] Max |Psi1|: {np.max(np.abs(psi1)):.6f}, |Psi2|: {np.max(np.abs(psi2)):.6f}, |Psi3|: {np.max(np.abs(psi3)):.6f}")


        if step > STABLE: 
            core_norm = np.sqrt(
                np.abs(psi1[cx, cy, cz])**2 +
                np.abs(psi2[cx, cy, cz])**2 +
                np.abs(psi3[cx, cy, cz])**2
            )
            core_trace.append(core_norm) #np.real(psi1[cx, cy, cz]))
            core_time_steps.append(step*dt)

        if step > STABLE and step % 2==0: 
            constraint_error = np.mean(np.abs(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2 - 1.0))
            constraint_error_list.append(constraint_error)

            core_field = np.sqrt(
                np.abs(psi1[cx, cy, cz])**2 +
                np.abs(psi2[cx, cy, cz])**2 +
                np.abs(psi3[cx, cy, cz])**2
            )
            core_field_list.append(core_field) #(core_field_1,core_field_2,core_field_3))

            kinetic = np.sum(np.abs(velocity1)**2 + np.abs(velocity2)**2 + np.abs(velocity3)**2)
            potential = 0.0
            for psi in [psi1, psi2, psi3]:
                # Compute gradient along each axis, for both real and imaginary parts
                grads_real = np.gradient(psi.real)
                grads_imag = np.gradient(psi.imag)
                # grads_real, grads_imag are lists: [dx, dy, dz]
                grad_sq = 0.0
                for g in grads_real:
                    grad_sq += g**2
                for g in grads_imag:
                    grad_sq += g**2
                # Sum over all voxels and accumulate for all three fields
                potential += np.sum(grad_sq)
            total_energy = kinetic + potential
            total_energy_list.append(total_energy)

            center_mask = np.zeros_like(psi1, dtype=bool)
            center_mask[cx-8:cx+8, cy-8:cy+8, cz-8:cz+8] = True
            core_energy = np.sum((np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2)[center_mask])
            core_energy_total.append(core_energy)

        # 8. Save snapshot if requested
        #if step==0 or (step>=ADD_KICK_STEP and step<=ADD_KICK_STEP + HARD_CONSTRAINT_STEPS + BLEND_FADE_STEPS) or step >= STABLE and (step==(steps - 1) or step in snapshot_steps): #if step<40 or step==(steps - 1) or step > 0 and step < steps and step in snapshot_steps:         
        if step==0 or (step==(steps - 1) or step in snapshot_steps): #if step<40 or step==(steps - 1) or step > 0 and step < steps and step in snapshot_steps:         

            # Compute complex mean
            phase_mean_complex = (psi1 + psi2 + psi3) / 3.0
            # Relative phase difference for ψ1 vs the SU(3) average
            safe_mean = np.where(np.abs(phase_mean_complex) < epsilon, epsilon + 0j, phase_mean_complex)
            phase_diff = np.angle(psi1 / safe_mean)
            # Apply smoothing (to reduce spatial noise and wrapping artifacts)
            phase_diff_smoothed = gaussian_filter(phase_diff, sigma=3.0)
            # Compute spatial gradient magnitude
            grad_x, grad_y, grad_z = np.gradient(phase_diff_smoothed, dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)


            T_xx, T_yy, T_zz, T_xy, T_xz, T_yz = compute_stress_tensor_components(psi1, psi2, psi3, dx)
            R_scalar = compute_ricci_scalar(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx)
            thetas, R_samples = safe_sample_ricci_angular_spread(R_scalar, radius, num_angles=360)
            R_samples_norm.append(R_samples)  # normalize


            save_grad_mag = gaussian_filter(np.abs(grad_mag), sigma=1.0)

            with ProcessPoolExecutor(max_workers=max_workers) as snapshot_executor:
                filename=f"{file_path}_PhaseDiff_GradMag_Step_{step}_Slice_xy.png"
                images_sc.append(filename)
                snapshot_executor.submit(save_zslice_image, file_path, title, f"PhaseDiff_GradMag_Step_{step}", save_grad_mag, axis='xy', dpi=dpi, cmap="twilight")

            iso_val = get_iso_value(grad_mag) # 10% of max
            if iso_val > 1e-8:  # Avoid near-zero isovalue failure.
                filename3d=f"{file_path}_PhaseDiff_GradMag_Step_{step}_3d.png"
                images_3d.append(filename3d)
                with ProcessPoolExecutor(max_workers=max_workers) as snapshot_executor_3d:
                    snapshot_executor_3d.submit(save_isosurface_3d, file_path, title, f"PhaseDiff_GradMag_Step_{step}", save_grad_mag, iso_value=iso_val, dpi=dpi, cmap="twilight",color_field=phase_field)
                
                filename3dNCF=f"{file_path}_PhaseDiff_GradMag_Step_{step}_NCF_3d.png"
                imagesNCF_3d.append(filename3dNCF)
                with ProcessPoolExecutor(max_workers=max_workers) as snapshot_executorNCF_3d:
                    snapshot_executorNCF_3d.submit(save_isosurface_3d, file_path, title, f"PhaseDiff_GradMag_Step_{step}_NCF", save_grad_mag, iso_value=iso_val, dpi=dpi, cmap="twilight",color_field=None)


    print(f"{title}: Loop Completed.")
    snapshot_executor.shutdown(wait=True)
    snapshot_executor_3d.shutdown(wait=True)
    snapshot_executorNCF_3d.shutdown(wait=True)

    print(f"{title}: Create Animation.")
    imganm=[]
    for filename in images_sc:
        imganm.append(imageio.imread(filename))
    imageio.mimsave(f"{file_path}_Slice_Evolution.gif", imganm, fps=2)

    imganm=[]
    for filename in images_3d:
        imganm.append(imageio.imread(filename))
    imageio.mimsave(f"{file_path}_3d_Evolution.gif", imganm, fps=2)

    imganm=[]
    for filename in imagesNCF_3d:
        imganm.append(imageio.imread(filename))
    imageio.mimsave(f"{file_path}_3d_NCF_Evolution.gif", imganm, fps=2)

    print(f"{title}: Create Central Slice.")
    # Compute complex mean
    phase_mean_complex = (psi1 + psi2 + psi3) / 3.0
    # Relative phase difference for ψ1 vs the SU(3) average
    safe_mean = np.where(np.abs(phase_mean_complex) < epsilon, epsilon + 0j, phase_mean_complex)
    phase_diff = np.angle(psi1 / safe_mean)
    # Apply smoothing (to reduce spatial noise and wrapping artifacts)
    phase_diff_smoothed = gaussian_filter(phase_diff, sigma=3.0)
    # Compute spatial gradient magnitude
    grad_x, grad_y, grad_z = np.gradient(phase_diff_smoothed, dx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    save_grad_mag = gaussian_filter(np.abs(grad_mag), sigma=1.0)

    save_zslice_image(file_path, title, f"Central_Slice", save_grad_mag, axis='xy', dpi=dpi, cmap="twilight")

    print(f"{title}: Create Central 3d iso.")
    save_isosurface_3d(file_path, title, f"Final", save_grad_mag, iso_value=iso_val, dpi=dpi, cmap="twilight")

    print(f"{title}: Constraint Error vs. Time.")
    plt.figure()
    plt.plot(constraint_error_list)
    plt.xlabel('Step')
    plt.ylabel('Constraint Error')
    plt.title(f"{title}: Constraint Error vs. Time")
    plt.yscale('log')  # If wide dynamic range
    plt.tight_layout()
    plt.savefig(f"{file_path}_Constraint_Error_vs_Time.png", dpi=dpi)
    plt.close()

    print(f"{title}: Core Field vs. Time.")
    plt.figure()
    plt.plot(core_field_list)
    plt.xlabel('Step')
    plt.ylabel('Core |ψ₁|')
    plt.title(f"{title}: Core Field Magnitude vs. Time")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Core_Field_vs_Time.png", dpi=dpi)
    plt.close()

    print(f"{title}: Total Energy vs. Time.")
    plt.figure()
    plt.plot(total_energy_list)
    plt.xlabel('Step')
    plt.ylabel('Total Energy')
    plt.title(f"{title}: Total Energy vs. Time")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Total_Energy_vs_Time.png", dpi=dpi)
    plt.close()

    print(f"{title}: Frequency Spectrum (FFT).")
    core_field_array = np.array(core_field_list)
    N = len(core_field_array)
    T = dt  # If each step is 1 unit in time, otherwise set your dt
    yf = fft(core_field_array)
    xf = fftfreq(N, T)[:N // 2]
    plt.figure()
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.xlabel('Frequency (1/step)')
    plt.ylabel('Amplitude')
    plt.title(f"{title}: Core Field Frequency Spectrum")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Core_Field_FFT.png", dpi=dpi)
    plt.close()

    print(f"{title}: Constraint Error Histogram.")
    final_constraint_error = np.abs(np.abs(psi1)**2 + np.abs(psi2)**2 + np.abs(psi3)**2 - 1.0)
    plt.figure()
    plt.hist(final_constraint_error.flatten(), bins=100)
    plt.xlabel('Constraint Error (Final Step)')
    plt.ylabel('Voxel Count')
    plt.title(f"{title}: Constraint Error Histogram")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Constraint_Error_Histogram.png", dpi=dpi)
    plt.close()

    print(f"{title}: Internal Oscillation at Soliton Core.")
    plt.figure(figsize=(8,4))
    plt.plot(core_time_steps, core_trace) #np.arange(steps)*dt  =  core_time_steps.append(step*dt)
    plt.xlabel("Time")
    plt.ylabel("Re(ψ₁) at Core")
    plt.title(f"{title}: Internal Oscillation at Soliton Core")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Internal_Oscillation_Soliton_Core.png", dpi=dpi)
    plt.close()



    G_soliton = compute_einstein_tensor_complete(psi1, psi2, psi3, dx=dx)

    T_soliton = compute_stress_energy_tensor(psi1, psi2, psi3, dx=dx)

    Soliton_Einstein_Tensor(file_path, title, G_soliton, cz, dpi=dpi) #psi1, psi2, psi3, dx

    Stress_Energy_Tensor(file_path, title, T_soliton, G_soliton, cz, dpi=dpi) #psi1, psi2, psi3, dx

    G_soliton_mag = np.linalg.norm(G_soliton, axis=(0,1))
    T_soliton_mag = np.linalg.norm(T_soliton, axis=(0,1))
    residual = np.abs(G_soliton_mag - 8 * np.pi * T_soliton_mag)

    cz = psi1.shape[2] // 2
    plt.imshow(residual[:, :, cz])
    plt.title(f"{title}: |G - 8πT| (Central Slice)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{file_path}_G_minus_8piT_Residual.png", dpi=dpi)
    plt.close()

    EinsteinTensor_Stats(file_path, title, G_soliton, residual)

    R_samples_norm = np.array(R_samples_norm)  # shape: (n_steps, n_angles)
    R_samples_norm = R_samples_norm / np.nanmax(np.abs(R_samples_norm))

    plt.figure(figsize=(6, 4))
    plt.plot(np.degrees(thetas), R_samples_norm[-1], label="Normalized $R(\\theta)$")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized Ricci Scalar")
    plt.title(f"{title}: Ricci Scalar Angular Profile (r={radius})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Ricci_Angular_Spread.png", dpi=dpi)
    plt.close()

    print(f"✅ Finished Test: {title} Validated.")




if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()