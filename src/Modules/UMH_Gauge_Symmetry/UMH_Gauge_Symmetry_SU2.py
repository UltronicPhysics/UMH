"""
UMH_Gauge_Symmetry.py

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
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd


from numba import njit, prange
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter1d


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_ricci_tensor_from_components
    from ..UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy,save_csv_slice,save_csv_3d
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_ricci_tensor_from_components
    from UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy,save_csv_slice,save_csv_3d


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 300, #500,
        "TIME_STEPS": 500, #500

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":30, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary. 30
        "PML_ABORPTION":0.20, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.18
        "PML_POWER": 3,


        "smoothing_frequency": 15,  # Apply every 20 steps, Increase smoothing_frequency to 50 or 100 for long simulations
        "smoothing_enabled": True,#True,  # Toggle smoothing on/off, Reduce 0.05 to something like 0.01

        "CONSTRAINT_STRENGTH":2.0,

        "VELOCITY_DAMPING":0.995, #0.985, 7-27-2025.
       

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


#Trim beginning and ending frames to ensure good data, and no outside influence of numerical buildup.
def trim_edges(data, trim_start=2, trim_end=2):
    return data[trim_start:-trim_end] if trim_end > 0 else data[trim_start:]

#Not Used, but can downsample for Iso 3d for speed.  Currently async so not a real issue.
def downsample(data, factor=4):
    return data[::factor, ::factor, ::factor]

def center_crop(slice_, crop_size=128):
    cx, cy = np.array(slice_.shape) // 2
    return slice_[cx - crop_size//2: cx + crop_size//2, cy - crop_size//2: cy + crop_size//2]

def zoom_slice_for_display(slice_, zoom_factor=4):
    return zoom(slice_, zoom=zoom_factor, order=1)  # Linear interpolation

def crop_field(field, thickness):
    return field[thickness:-thickness, thickness:-thickness, thickness:-thickness]


#Save ZSlice of wave.
def save_zslice_image(file_path, title, name, array, axis, dpi=300, cmap="seismic"):
    if axis == 'xy':
        slice_ = array[:, :, array.shape[2] // 2]
    elif axis == 'xz':
        slice_ = array[:, array.shape[1] // 2, :]
    elif axis == 'yz':
        slice_ = array[array.shape[0] // 2, :, :]
    else:
        raise ValueError("Axis must be 'xy', 'xz', or 'yz'")
        
    if np.iscomplexobj(slice_):
        slice_ = np.abs(slice_)


    slice_zoomed=slice_
    slice_ctr = center_crop(slice_, crop_size=128)
    slice_zoomed = zoom_slice_for_display(slice_ctr, zoom_factor=1.5) #2

    vmax = np.percentile(slice_zoomed, 85)
    #vmin = -vmax if np.min(slice_zoomed) < 0 else 0
    vmin = np.percentile(slice_zoomed, 5)

    print(f"{title}: Slice Zoomed: max={np.max(slice_zoomed)}, min={np.min(slice_zoomed)}, mean={np.mean(slice_zoomed)}")

    plt.figure(figsize=(12, 12))
    plt.imshow(slice_zoomed, cmap=cmap, interpolation='nearest',vmin=vmin,vmax=vmax) #cmap='seismic', vmin=-0.02, vmax=0.02  norm=LogNorm(vmin=1e-5, vmax=1)
    plt.colorbar()
    plt.title(f"{title}: {name} slice {axis.upper()}")
    plt.tight_layout()

    filename=f"{file_path}_{name}_Slice_{axis}.png"
    plt.savefig(filename, dpi=dpi) #, bbox_inches='tight', transparent=True
    plt.close()


def compute_radial_profile_old(field, center):
    """Computes radial average of a 3D scalar field."""
    grid = np.indices(field.shape).T - center
    radius = np.linalg.norm(grid, axis=-1)
    radius = radius.flatten()
    values = field.flatten()
    # Bin by radius
    bins = np.linspace(0, radius.max(), num=100)
    bin_means = np.zeros_like(bins)
    for i in range(len(bins) - 1):
        mask = (radius >= bins[i]) & (radius < bins[i + 1])
        if np.any(mask):
            bin_means[i] = np.mean(np.abs(values[mask]))
    return bins[:-1], bin_means[:-1]

def compute_radial_profile(field, center):
    """
    Compute the spherical radial profile of a 3D scalar field.
    Args:
        field (ndarray): 3D field (e.g., stress tensor component).
        center (tuple): Center coordinates (x, y, z) of symmetry.
    Returns:
        bin_centers (ndarray), radial_means (ndarray)
    """
    # Build coordinate grids
    x, y, z = np.indices(field.shape)
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    radius = np.sqrt(dx**2 + dy**2 + dz**2)

    # Flatten arrays
    radius_flat = radius.flatten()
    values_flat = np.abs(field).flatten()

    # Remove center (r = 0) to avoid log(0) in logspace
    mask = radius_flat > 0
    radius_flat = radius_flat[mask]
    values_flat = values_flat[mask]

    # Use logarithmic bins for smoother decay detection
    r_min = radius_flat.min()
    r_max = radius_flat.max()
    bins = np.logspace(np.log10(r_min), np.log10(r_max), num=100)

    bin_means = np.zeros(len(bins) - 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean

    for i in range(len(bins) - 1):
        bin_mask = (radius_flat >= bins[i]) & (radius_flat < bins[i + 1])
        if np.any(bin_mask):
            bin_means[i] = np.mean(values_flat[bin_mask])
        else:
            bin_means[i] = np.nan  # or interpolate later

    return bin_centers, bin_means


def radial_plot(file_path, title, name, einstein_G_zz, anmgifary=None, dpi=300):

    # Assuming `einstein_G_zz` is loaded as a 3D numpy array
    center = np.array(einstein_G_zz.shape) // 2
    radii, radial_energy = compute_radial_profile(einstein_G_zz, center)

    plt.figure()
    plt.loglog(radii, radial_energy, marker='o')
    plt.xlabel('Radius (r)')
    plt.ylabel('Mean $|G_{zz}|$')
    plt.title(f"{title}: {name}: Gravitational Wave Energy Flux Decay")
    plt.grid(True)
    plt.savefig(f"{file_path}_{name}.png", dpi=dpi)
    plt.close()

    if anmgifary is not None: anmgifary.append(imageio.imread(filename))



def get_iso_value(array, percentile=99.9):
    return np.percentile(np.abs(array.flatten()), percentile)

#Plot 3d Iso Image.
def save_isosurface_3d(file_path, title, name, tensor_data, iso_value=0.1, dpi=300, color='red', alpha=0.6):
    # Extract isosurface using marching cubes

    # Always plot magnitude for safety (handles real or complex fields)
    verts, faces, normals, values = measure.marching_cubes(np.abs(tensor_data), level=iso_value, step_size=1)

    # Setup Matplotlib 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Build mesh
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    # Set axis limits based on tensor size
    ax.set_xlim(0, tensor_data.shape[0])
    ax.set_ylim(0, tensor_data.shape[1])
    ax.set_zlim(0, tensor_data.shape[2])

    # Aesthetics
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}: {name} Isosurface at {iso_value}")

    ax.view_init(elev=30, azim=45)  # Set viewing angle

    plt.tight_layout()
    filename=f"{file_path}_{name}_3d.png"
    plt.savefig(filename, dpi=dpi)
    plt.close()


def compute_tensor_residuals(file_path, title, G, T, label="zz", dpi=300):
    factor = 8 * np.pi
    residual = G - factor * T
    l2_norm = np.linalg.norm(residual)
    mean_abs = np.mean(np.abs(residual))

    print(f"{title}: Residual G_{label} - 8πT_{label}: L2 norm = {l2_norm:.3e}, Mean |residual| = {mean_abs:.3e}")

    if file_path:
        slice_xy = residual[:, :, residual.shape[2] // 2]
        plt.figure(figsize=(6, 5))
        plt.imshow(slice_xy, cmap='seismic', vmin=-np.max(np.abs(slice_xy)), vmax=np.max(np.abs(slice_xy)))
        plt.title(f"{title}: Residual $G_{{{label}}} - 8\pi T_{{{label}}}$")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{file_path}_Residual_G_{label}_minus_8piT_{label}.png", dpi=dpi)
        plt.close()

    return residual


def low_pass_filter_2d(data, cutoff_fraction=0.1):
    """
    Apply a low-pass FFT filter to a 2D field.
    
    Parameters:
        data (2D array): The input field (e.g., divergence_xy slice).
        cutoff_fraction (float): Fraction of frequencies to keep (e.g., 0.1 keeps lowest 10%)
        
    Returns:
        2D array: Filtered field (real part).
    """
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    nx, ny = data.shape
    cx, cy = nx // 2, ny // 2
    cutoff_x = int(cutoff_fraction * nx // 2)
    cutoff_y = int(cutoff_fraction * ny // 2)

    # Zero out high frequencies
    mask = np.zeros_like(fft_data, dtype=bool)
    mask[cx - cutoff_x: cx + cutoff_x, cy - cutoff_y: cy + cutoff_y] = True
    fft_data_filtered = np.where(mask, fft_data, 0)

    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fft_data_filtered)))
    return filtered

def low_pass_filter_3d(data, cutoff_fraction=0.1):
    shape = data.shape
    data_fft = np.fft.fftn(data)
    kx = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kz = np.fft.fftfreq(shape[2])
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    data_fft[k_mag > cutoff_fraction] = 0
    return np.real(np.fft.ifftn(data_fft))


def compute_ricci_scalar(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx):
    # Second derivatives (Laplacians) of the stress-energy tensor components
    d2T_xx = compute_laplacian_27point(T_xx, dx)
    d2T_yy = compute_laplacian_27point(T_yy, dx)
    d2T_zz = compute_laplacian_27point(T_zz, dx)

    # Mixed second derivatives: divergence of off-diagonal terms
    d2T_xy = compute_laplacian_27point(T_xy, dx)
    d2T_xz = compute_laplacian_27point(T_xz, dx)
    d2T_yz = compute_laplacian_27point(T_yz, dx)

    # Combine components: this is a proxy for Ricci scalar
    R_scalar = d2T_xx + d2T_yy + d2T_zz - (d2T_xy + d2T_xz + d2T_yz)
    return R_scalar

@njit(parallel=True, fastmath=True, cache=True)
def sample_ricci_angular_spread(R_scalar, radius, thetas):
    Nx, Ny, Nz = R_scalar.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    num_angles = len(thetas)

    results = np.full(num_angles, np.nan)

    for i in prange(num_angles):
        theta = thetas[i]
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta))
        z = cz
        if 0 <= x < Nx and 0 <= y < Ny:
            results[i] = R_scalar[x, y, z]

    return results

def safe_sample_ricci_angular_spread(R_scalar, radius=15, num_angles=360):
    thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    R_samples = sample_ricci_angular_spread(R_scalar, radius, thetas)
    return thetas, R_samples


@njit(parallel=True, cache=True, fastmath=True)
def compute_G_zz_norm(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz):
    Nx, Ny, Nz = T_xx.shape
    G_zz_total = 0.0

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Ricci tensor components at this point
                R_zz = T_zz[i, j, k]
                R_trace = T_xx[i, j, k] + T_yy[i, j, k] + T_zz[i, j, k]

                # G_zz = R_zz - 0.5 * R
                G_zz = R_zz - 0.5 * R_trace
                G_zz_total += G_zz * G_zz

    return np.sqrt(G_zz_total)


@njit(parallel=True, cache=True, fastmath=True)
def enforce_su2_phase_constraint(psi1, psi2):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                norm = np.sqrt(np.abs(psi1[i, j, k])**2 + np.abs(psi2[i, j, k])**2)
                if norm > 1e-8:  # Avoid divide by zero
                    psi1[i, j, k] /= norm
                    psi2[i, j, k] /= norm
    return psi1, psi2


@njit(parallel=True, cache=True, fastmath=True)
def initialize_su2_field(Nx, Ny, Nz, phase_amplitude=0.05, base_amplitude=0.0):
    psi1 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    psi2 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                theta = np.random.uniform(-phase_amplitude, phase_amplitude)
                phi = np.random.uniform(-phase_amplitude, phase_amplitude)
                
                # Apply base amplitude uniformly or leave at 0 for noise-free vacuum
                psi1[i, j, k] = np.exp(1j * theta) * base_amplitude
                psi2[i, j, k] = np.exp(1j * phi) * base_amplitude
    return psi1, psi2



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

    smoothing_frequency=config["smoothing_frequency"]
    smoothing_enabled=config["smoothing_enabled"]

    CONSTRAINT_STRENGTH=config["CONSTRAINT_STRENGTH"]

    VELOCITY_DAMPING=config["VELOCITY_DAMPING"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Gauge Symmetry (SU2)"
    file_root="UMH_Gauge_Symmetry"
    file_sub="UMH_SU2"
    file_hdr="UMH_Gauge_Symmetry_SU2"
  
    print(f"✅ Starting Test: {title} Validated.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    radius = 15 #Used for ricci_angular_spread.

    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    v2 = v**2 #Wave Speed squared.
    print(f"{title}: Calculated Wave Speed Constant C = {v}")

    safety_factor=0.50 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07

    damp_radius=6 #Radius toward center to dampen from kick artifact, caused by numerical simulation.
    damping_steps=8
    #max_damping = 0.999 #How much to quadratically dampen inward for kick artifact.

    # Safety Factor Adjust GridSize.
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    print(f"{title}: auto_cfl_clamp, Min:{min_factor}, Max:{max_factor}")
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)

    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor)
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")
    
    #dt = dt * 0.5

    # Before the loop, allocate three buffers 
    psi1, psi2 = initialize_su2_field(size, size, size, phase_amplitude=0.05, base_amplitude=0.05)

    velocity1 = np.zeros_like(psi1)  
    velocity2 = np.zeros_like(psi2)

    # Apply localized "kick" at center
    center = (size // 2, size // 2, size // 2)
    cx, cy, cz = center

    kick_radius=4 #Radius of spherical kick.
    kick_strength=0.25 #How strong of a kick, amplitude.
    kick_phase_shift = np.pi / 4  # any angle to shift ψ₂ out of phase

    initial_energy1=apply_gaussian_kick(None, psi1, center, kick_radius=4, kick_strength=0.4)

    # For ψ₂, apply a phase shifted version of the kick:
    gaussian_field = create_gaussian_field(psi2, center, radius=4, strength=0.4)
    psi2 += gaussian_field * np.exp(1j * kick_phase_shift)

    print(f"{title}: Initial Total Energy After Small Radius - Point Kick: {initial_energy1}")

    #center = (size // 2, size // 2, size // 2)

    absorption_mask = create_absorption_mask(psi1.shape, thickness=pml_thickness, absorption_coeff=pml_absorption,power=pml_power, dtype=dtype) #PML usage with gentle boundary damping to reduce relection.
   
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    monitor_total_energy=[]
    time_steps,div_x_norms,div_y_norms,div_z_norms,G_zz_norms = [],[],[],[],[]
    R_samples_norm=[]

    print(f"{title}: Starting Loop.")
    #Loop and step through wave progression.
    for step in range(steps):

        # 1️⃣ Gauge Evolution with Laplacian + Constraint Force
        lap1 = compute_laplacian_27point(psi1, dx)
        lap2 = compute_laplacian_27point(psi2, dx)

        if step > 0 and CONSTRAINT_STRENGTH > 2.0 and step % 50 == 0:
            CONSTRAINT_STRENGTH *= 0.995 #0.98 7-27-2025.
            print(f"[DEBUG] {title}: Constraint Strength, Step {step}. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}, Adjust CS to:{CONSTRAINT_STRENGTH}")

        constraint_force = np.clip(CONSTRAINT_STRENGTH * (np.abs(psi1)**2 + np.abs(psi2)**2 - 1.0), -1e6, 1e6)

        accel1 = v**2 * lap1 - constraint_force * psi1
        accel2 = v**2 * lap2 - constraint_force * psi2

        velocity1 += accel1 * dt
        velocity2 += accel2 * dt
        velocity1 *= VELOCITY_DAMPING
        velocity2 *= VELOCITY_DAMPING
        velocity1 = np.clip(velocity1, -5.0, 5.0)
        velocity2 = np.clip(velocity2, -5.0, 5.0)

        # 🟢 Apply Phase Drift Correction Before Psi Update
        phase_center = np.angle(psi1[cx, cy, cz]) - np.angle(psi2[cx, cy, cz])
        psi1 *= np.exp(-1j * phase_center / 2)
        psi2 *= np.exp(1j * phase_center / 2)

        # Apply absorption
        velocity1 *= absorption_mask
        velocity2 *= absorption_mask

        # Update Psi with velocity
        psi1 += velocity1 * dt
        psi2 += velocity2 * dt

        # 2️⃣ Apply Optional Laplacian Smoothing
        if config["smoothing_enabled"] and step % config["smoothing_frequency"] == 0 and step > 0:
            print(f"[DEBUG] {title}: SMOOTHING TRIGGERED PRE: Step {step}. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}")
            lap_diff1 = compute_laplacian_27point(psi1, dx)
            psi1 += 0.005 * lap_diff1 * dt
            psi1 = np.where(np.abs(psi1) < 1e-7, 0, psi1)

            lap_diff2 = compute_laplacian_27point(psi2, dx)
            psi2 += 0.005 * lap_diff2 * dt
            psi2 = np.where(np.abs(psi2) < 1e-7, 0, psi2)

            print(f"[DEBUG] {title}: SMOOTHING TRIGGERED POST: Step {step}. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}")

        # 3️⃣ Apply Threshold Clipping
        threshold = 1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            mask1 = np.abs(psi1) > threshold
            mask2 = np.abs(psi2) > threshold

            if np.any(mask1) or np.any(mask2):
                psi1[mask1] = psi1[mask1] / np.abs(psi1[mask1]) * threshold
                psi2[mask2] = psi2[mask2] / np.abs(psi2[mask2]) * threshold
                print(f"[DEBUG] {title}: THRESHOLD HIT at Step {step}. Max Psi1: {np.max(np.abs(psi1)):.4f}, Max Psi2: {np.max(np.abs(psi2)):.4f}")

        # 4️⃣ Enforce SU(2) Phase Constraint
        if step % 5 == 0:
            psi1, psi2 = enforce_su2_phase_constraint(psi1, psi2)
            print(f"[DEBUG] {title}: enforce_su2_phase_constraint, Step {step}. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}")

        # 5️⃣ Apply Center Damping After Constraint
        if (step % 5 == 0) and (step > damp_radius + kick_radius):
            apply_center_damping(psi1, center, damp_radius, max_damping=0.5)
            apply_center_damping(psi2, center, damp_radius, max_damping=0.5)
            damping_steps -= 1
            print(f"[DEBUG] {title}: CENTER DAMPENING, Step {step}. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}")

        # Optional Monitor / Debugging Here if needed

        
        print(f"[DEBUG] {title}: POST:THRESHOLD, Step {step} complete. Max Psi1:{np.max(np.abs(psi1))}, Max Psi2: {np.max(np.abs(psi2))}")

        #Only save every so many frames for size and performance.
        if step>0 and step<steps and step in snapshot_steps: # or step == steps - 1     #Snapshot every so many frames for speed and size.  Wait till it gets going for first.
            max_psi1 = np.max(np.abs(psi1))
            max_psi2 = np.max(np.abs(psi2))
            print(f"{title}, Step: {step} — Max |Psi1|: {max_psi1:.6f}, |Psi2|: {max_psi2:.6f}")
            total_energy1 = np.sum(np.abs(psi1)**2)
            total_energy2 = np.sum(np.abs(psi2)**2)
            monitor_total_energy.append((total_energy1,total_energy2))

            grad_psi1_x = np.gradient(np.angle(psi1), dx, axis=0)
            grad_psi1_y = np.gradient(np.angle(psi1), dx, axis=1)
            grad_psi1_z = np.gradient(np.angle(psi1), dx, axis=2)

            grad_psi2_x = np.gradient(np.angle(psi2), dx, axis=0)
            grad_psi2_y = np.gradient(np.angle(psi2), dx, axis=1)
            grad_psi2_z = np.gradient(np.angle(psi2), dx, axis=2)

            # Construct tensor components similar to post-loop
            T_xx = (grad_psi1_x**2 + grad_psi2_x**2) * Tu
            T_yy = (grad_psi1_y**2 + grad_psi2_y**2) * Tu
            T_zz = (grad_psi1_z**2 + grad_psi2_z**2) * Tu
            T_xy = (grad_psi1_x * grad_psi1_y + grad_psi2_x * grad_psi2_y) * Tu
            T_xz = (grad_psi1_x * grad_psi1_z + grad_psi2_x * grad_psi2_z) * Tu
            T_yz = (grad_psi1_y * grad_psi1_z + grad_psi2_y * grad_psi2_z) * Tu

            # Now compute divergence
            div_x, div_y, div_z = compute_divergence(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx)
            # Compute norms
            div_x_norms.append(np.linalg.norm(div_x))
            div_y_norms.append(np.linalg.norm(div_y))
            div_z_norms.append(np.linalg.norm(div_z))

            G_zz_norm = compute_G_zz_norm(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz)
            G_zz_norms.append(G_zz_norm)

            R_scalar = compute_ricci_scalar(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx)

            thetas, R_samples = safe_sample_ricci_angular_spread(R_scalar, radius, num_angles=360)
            R_samples_norm.append(R_samples)  # normalize

            time_steps.append(step)

            if True:
                phase_diff = np.angle(psi1) - np.angle(psi2)
                unwrapped_diff = np.unwrap(np.unwrap(np.unwrap(phase_diff, axis=0), axis=1), axis=2)
                # Apply Gaussian smoothing to unwrapped phase before gradient
                unwrapped_diff_smoothed = gaussian_filter(unwrapped_diff, sigma=2.0)
                grad_x, grad_y, grad_z = np.gradient(unwrapped_diff_smoothed, dx)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

                save_zslice_image(file_path, title, f"DBG_PhaseDiff_GradMag_Step_{step}", grad_mag, axis='xy', dpi=dpi, cmap="plasma")



    print(f"{title}: Loop Completed.")

    # 🟢 Final center damping to suppress central residuals
    apply_center_damping(psi1, center, damp_radius=8, max_damping=0.6)
    apply_center_damping(psi2, center, damp_radius=8, max_damping=0.6)

    # 🟢 Final absorption damping
    psi1 *= absorption_mask ** 2
    psi2 *= absorption_mask ** 2

    # ✅ Composite Phase for SU(2)
    phase_diff = np.angle(psi1) - np.angle(psi2)

    # ✅ Fully unwrap across all three axes
    unwrapped_diff = np.unwrap(np.unwrap(np.unwrap(phase_diff, axis=0), axis=1), axis=2)

    # 🟢 Optional: Dampen unwrapped_diff before gradient to reduce artifacts
    apply_center_damping(unwrapped_diff, center, damp_radius=8, max_damping=0.3)

    # ✅ Compute gradients of the unwrapped phase difference
    grad_diff_x = np.gradient(unwrapped_diff, dx, axis=0)
    grad_diff_y = np.gradient(unwrapped_diff, dx, axis=1)
    grad_diff_z = np.gradient(unwrapped_diff, dx, axis=2)

    # ✅ Construct tensor components
    tensors = {
        'T_xx': grad_diff_x**2 * Tu,
        'T_yy': grad_diff_y**2 * Tu,
        'T_zz': grad_diff_z**2 * Tu,
        'T_xy': grad_diff_x * grad_diff_y * Tu,
        'T_xz': grad_diff_x * grad_diff_z * Tu,
        'T_yz': grad_diff_y * grad_diff_z * Tu,
    }

    print(f"{title}, Max T_zz: {np.max(tensors['T_zz'])}, Min: {np.min(tensors['T_zz'])}")

    # ✅ Process tensor fields: damp, absorb, crop, smooth, and threshold
    for name, array in tensors.items():
        apply_center_damping(array, center, damp_radius=10, max_damping=0.2)
        array *= absorption_mask ** 2
        array = crop_field(array, pml_thickness * 2)
        array = gaussian_filter(array, sigma=2.0)
        flt_array = low_pass_filter_3d(array, cutoff_fraction=0.1)
        flt_array = np.where(np.abs(flt_array) < 1e-8, 0, flt_array)  # updated tighter threshold
        tensors[name] = flt_array

    # ✅ Compute curvature (Laplacian of tensors)
    curvatures = {k: compute_laplacian_27point(v, dx) for k, v in tensors.items()}

    # ✅ Compute divergence of the tensor field
    div_x, div_y, div_z = compute_divergence(
        tensors['T_xx'], tensors['T_yy'], tensors['T_zz'],
        tensors['T_xy'], tensors['T_xz'], tensors['T_yz'], dx
    )

    print(f"[DEBUG] Max Divergence X: {np.max(np.abs(div_x))}, Y: {np.max(np.abs(div_y))}, Z: {np.max(np.abs(div_z))}")

    # ✅ Einstein Tensor via Ricci Tensor and Scalar Curvature
    R_xx, R_yy, R_zz = compute_ricci_tensor_from_components(curvatures, dx)

    scalar_curv = compute_scalar_curvature(R_xx, R_yy, R_zz)
    G_xx, G_yy, G_zz = compute_einstein_tensor(R_xx, R_yy, R_zz, scalar_curv)

    print(f"[DEBUG] Max Einstein G_xx: {np.max(np.abs(G_xx))}, G_yy: {np.max(np.abs(G_yy))}, G_zz: {np.max(np.abs(G_zz))}")

    # ✅ Process composite Psi field for export
    psiT = (psi1 + 1j * psi2).copy()
    apply_center_damping(psiT, center, damp_radius=10, max_damping=0.2)
    psiT *= absorption_mask ** 2
    psiT = crop_field(psiT, pml_thickness * 2)
    psiT = gaussian_filter(psiT, sigma=1.5)
    psiT = np.where(np.abs(psiT) < 1e-8, 0, psiT)

    # ✅ Export all processed fields
    output_fields = {
        'Psi': psiT,
        **tensors,
        'curvature_T_xx': curvatures['T_xx'],
        'curvature_T_yy': curvatures['T_yy'],
        'curvature_T_zz': curvatures['T_zz'],
        'divergence_x': div_x, #flt_div_x,
        'divergence_y': div_y, #flt_div_y,
        'divergence_z': div_z, #flt_div_z,
        'Einstein_G_xx': G_xx,
        'Einstein_G_yy': G_yy,
        'Einstein_G_zz': G_zz,
        'Ricci_R_xx': R_xx,
        'Ricci_R_yy': R_yy,
        'Ricci_R_zz': R_zz,
    }


    print(f"{title}: Post-Processing Export Starting.")

    plt.figure()
    plt.plot(time_steps, div_x_norms, label='Div X')
    plt.plot(time_steps, div_y_norms, label='Div Y')
    plt.plot(time_steps, div_z_norms, label='Div Z')
    plt.xlabel("Step")
    plt.ylabel("||Div||")
    plt.title(f"{title}: Divergence Norm vs Time")
    plt.yscale('log')  # Optional, useful if divergence decays exponentially
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_Divergence_Norm_vs_Time.png", dpi=dpi)
    plt.close()

    # Save Scalar Curvature
    save_npy(f"{file_path}_Scalar_Curvature", scalar_curv)
    save_csv_slice(f"{file_path}_Scalar_Curvature", scalar_curv)
    print(f"{title}, Saved Scalar_Curvature.csv")

    save_zslice_image(file_path, title, "Scalar_Curvature", scalar_curv, axis='xy', dpi=dpi, cmap="seismic")
    iso_val = get_iso_value(scalar_curv)
    #save_isosurface_3d(file_path, title, "Scalar_Curvature", scalar_curv, iso_value=iso_val, dpi=dpi)
    print(f"{title}, Scalar_Curvature isosurface disabled (iso_val={iso_val:.4e})")

    # Radial & Isosurface for Ricci and Einstein
    for name, field in [("Ricci_R_zz", R_zz), ("Einstein_G_zz", G_zz)]:
        radial_plot(file_path, title, name, field, dpi=dpi)
        #iso_val = get_iso_value(field)
        #save_isosurface_3d(file_path, title, name, field, iso_value=iso_val, dpi=dpi)
        print(f"{title}, {name} isosurface disabled (iso_val={iso_val:.4e})")

    # Save Output Fields
    for name, array in output_fields.items():
        save_npy(f"{file_path}_{name}", array)
        save_csv_slice(f"{file_path}_{name}_Slice", array)
        print(f"{title}, Saved {name}_Slice.csv")

        try:
            save_zslice_image(file_path, title, name, array, axis='xy', dpi=dpi, cmap="seismic")
            print(f"{title}, Saved {name}_Slice.png")
        except Exception as e:
            print(f"Plot error for {name}: {e}")

        try:
            #iso_val = get_iso_value(array)
            #save_isosurface_3d(file_path, title, name, array, iso_value=iso_val, dpi=dpi)
            print(f"{title}, {name} isosurface disabled (iso_val={iso_val:.4e})")
        except Exception as e:
            print(f"Isosurface plot error for {name}: {e}")

    # ✅ Energy Plot with Step vs. Energy
    energy_arr = np.array(monitor_total_energy) 
    energy_sum = energy_arr[:, 0] + energy_arr[:, 1]
    energy_smooth = uniform_filter1d(energy_sum, size=3)
    plt.figure()
    plt.plot(energy_arr[:, 0], label='Energy Psi1')
    plt.plot(energy_arr[:, 1], label='Energy Psi2')
    plt.plot(energy_sum, label='Total Energy')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(f"{title}: Energy Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}_Energy_Evolution.png", dpi=dpi)
    plt.close()


    div_x_nrm = np.linalg.norm(div_x)
    div_y_nrm = np.linalg.norm(div_y)
    div_z_nrm = np.linalg.norm(div_z)

    G_xx_nrm = np.linalg.norm(G_xx)
    G_yy_nrm = np.linalg.norm(G_yy)
    G_zz_nrm = np.linalg.norm(G_zz)

    print(f"{title}: Divergence Norms — X: {div_x_nrm:.4e}, Y: {div_y_nrm:.4e}, Z: {div_z_nrm:.4e}")
    print(f"{title}: Einstein Tensor Norms — G_xx: {G_xx_nrm:.4e}, G_yy: {G_yy_nrm:.4e}, G_zz: {G_zz_nrm:.4e}")

    plt.figure(figsize=(8,6))
    plt.bar(['Div_X', 'Div_Y', 'Div_Z', 'G_xx', 'G_yy', 'G_zz'], 
            [div_x_nrm, div_y_nrm, div_z_nrm, G_xx_nrm, G_yy_nrm, G_zz_nrm], 
            color=['blue', 'blue', 'blue', 'red', 'red', 'red'])
    plt.ylabel('L2 Norm')
    plt.title(f"{title}: Divergence vs. Einstein Tensor Norms")
    plt.grid(True)
    plt.savefig(f"{file_path}_Divergence_vs_Einstein_Norms.png", dpi=dpi)
    plt.close()

    ratio_x = div_x_nrm / (G_xx_nrm + 1e-12)
    ratio_y = div_y_nrm / (G_yy_nrm + 1e-12)
    ratio_z = div_z_nrm / (G_zz_nrm + 1e-12)

    print(f"{title}: Ratio Divergence/Einst. — X: {ratio_x:.4e}, Y: {ratio_y:.4e}, Z: {ratio_z:.4e}")

    #Get T_zz.
    #radii, mean_T_zz = compute_radial_profile(T_zz, center)
    #plt.loglog(radii, mean_T_zz, label='T_zz (Sim)')
    #plt.loglog(radii, 1 / radii**2, '--', label='Expected $1/r^2$ Decay')
    #plt.xlabel('Radius')
    #plt.ylabel('Mean |T_zz|')
    #plt.legend()
    #plt.title(f"{title}: Curvature vs. Radial Distance")
    #plt.savefig(f"{file_path}_Curvature_Radial_Profile.png", dpi=dpi)
    #plt.close()

    # Compute profile
    centerRP = tuple(s // 2 for s in T_zz.shape)
    radii, radial_means = compute_radial_profile(T_zz, centerRP)

    # Plot radial profile and expected decay
    plt.figure(figsize=(6, 4))
    plt.loglog(radii, radial_means, label="UMH SU(2) $|T_{zz}|$")
    plt.loglog(radii, 1 / radii**2, '--', label="Analytic $1/r^2$")

    plt.xlabel("Radius (grid units)")
    plt.ylabel("Mean $|T_{zz}|$")
    plt.title(f"{title}: Radial Decay of Stress Tensor in SU(2) Gauge Field")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(f"{file_path}_Curvature_Radial_Profile.png", dpi=dpi)
    plt.close()


    plt.plot(time_steps, div_z_norms, label="||Div_Z||")
    plt.plot(time_steps, G_zz_norms, label="||G_zz||")
    plt.xlabel('Time Step')
    plt.ylabel('Norm')
    plt.legend()
    plt.title(f"{title}: Einstein Tensor vs. Divergence Decay")
    plt.savefig(f"{file_path}_Einstein_vs_Divergence_Decay.png", dpi=dpi)
    plt.close()


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


    #3d ISO.
    # --- Compute |ψ|² and select ROI ---
    psi_mag = np.abs(psiT)**2
    center = np.array(psi_mag.shape) // 2
    zoom = 40  # half-width of region

    roi = np.s_[center[0]-zoom:center[0]+zoom,
                center[1]-zoom:center[1]+zoom,
                center[2]-zoom:center[2]+zoom]
    psi_roi = psi_mag[roi]

    # Print value range for diagnostics
    min_val = np.min(psi_roi)
    max_val = np.max(psi_roi)
    print(f"[3D ISO] ψ² min: {min_val:.4e}, max: {max_val:.4e}")

    # --- Figure ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- Safe dynamic isosurface rendering ---
    for raw_level, color in [(0.5, 'steelblue'), (0.8, 'navy')]:
        iso_level = raw_level * max_val  # scale relative to max

        # Ensure iso_level is valid
        if min_val < iso_level < max_val:
            verts, faces, _, _ = measure.marching_cubes(psi_roi, level=iso_level)
            mesh = Poly3DCollection(verts[faces], alpha=0.5, facecolor=color)
            ax.add_collection3d(mesh)
            print(f"[3D ISO] Rendered level {iso_level:.3e}")
        else:
            print(f"[3D ISO] Skipped level {iso_level:.3e} (out of range)")

    # --- Add Central Slice with Nan Mask ---
    slice_cropped = psi_roi[:, :, zoom]
    X, Y = np.meshgrid(np.arange(2 * zoom), np.arange(2 * zoom), indexing='ij')
    Z = np.ones_like(X, dtype=np.float32) * zoom

    r = np.sqrt((X - zoom)**2 + (Y - zoom)**2)
    fade = np.clip(1 - r / zoom, 0, 1)
    normalized = slice_cropped / np.max(slice_cropped)
    faded_slice = normalized * fade
    faded_slice[normalized < 0.05] = np.nan
    Z[np.isnan(faded_slice)] = np.nan

    ax.plot_surface(
        X, Y, Z,
        facecolors=plt.cm.plasma(np.nan_to_num(faded_slice)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        alpha=0.9, shade=False
    )

    # --- Axes ---
    ax.set_xlim(0, 2*zoom)
    ax.set_ylim(0, 2*zoom)
    ax.set_zlim(0, 2*zoom)
    plot_title = r"$|\psi_T|^2$ with Central Slice"
    ax.set_title(f"{title}: {plot_title}")

    plt.tight_layout()
    plt.savefig(f"{file_path}_3d.png", dpi=dpi)
    plt.close()



    print(f"✅ Finished Test: {title} Validated.")




if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()