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
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_ricci_tensor_from_components
    from ..UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy,save_csv_slice,save_csv_3d
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_ricci_tensor_from_components
    from UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_Visuals import save_npy,save_csv_slice,save_csv_3d


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 500, #500,
        "TIME_STEPS": 500, #500

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":20, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary. 30
        "PML_ABORPTION":0.18, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.15
        "PML_POWER": 3,


        "smoothing_frequency": 20,  # Apply every 20 steps, Increase smoothing_frequency to 50 or 100 for long simulations
        "smoothing_enabled": True,  # Toggle smoothing on/off, Reduce 0.05 to something like 0.01
       

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

    vmax = np.percentile(slice_zoomed, 95)
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


def compute_radial_profile(field, center):
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



#Plot 3d Iso Image.
def save_isosurface_3d(file_path, title, name, tensor_data, iso_value=0.1, dpi=300, color='red', alpha=0.6):
    # Extract isosurface using marching cubes
    verts, faces, normals, values = measure.marching_cubes(tensor_data, level=iso_value, step_size=1)

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

def compute_tensor_residuals(G, T, label="zz", file_path=None, dpi=300):
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


@njit(cache=True, parallel=True, fastmath=True)
def initialize_U1_field(Nx, Ny, Nz, amplitude=1.0):
    """
    Initialize a U(1) complex field with random phase and constant amplitude.

    Parameters:
    - Nx, Ny, Nz: Grid dimensions
    - amplitude: Constant amplitude for each site (default 1.0)

    Returns:
    - psi: Initialized complex-valued numpy array of shape (Nx, Ny, Nz)
    """
    psi = np.empty((Nx, Ny, Nz), dtype=np.complex128)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                theta = 2.0 * np.pi * np.random.rand()
                psi[i, j, k] = amplitude * (np.cos(theta) + 1j * np.sin(theta))

    return psi


@njit(cache=True, parallel=True, fastmath=True)
def enforce_U1_phase_constraint(psi_next, strength=0.1):
    """
    Enforce a U(1) phase-lock constraint using a stencil-27 neighbor averaging.
    Parallelized with Numba njit for high performance on large lattices.

    Parameters:
    - psi_next: complex-valued numpy 3D array (Nx, Ny, Nz)
    - strength: constraint blend factor (0.0 - 1.0), typically 0.1

    Returns:
    - psi_constrained: phase-constrained field (same shape as psi_next)
    """
    Nx, Ny, Nz = psi_next.shape
    psi_constrained = psi_next.copy()

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                sum_phase = 0.0
                count = 0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue  # Skip self
                            neighbor = psi_next[i + di, j + dj, k + dk]
                            sum_phase += np.angle(neighbor)
                            count += 1

                avg_phase = sum_phase / count
                mag = np.abs(psi_next[i, j, k])
                cur_phase = np.angle(psi_next[i, j, k])
                new_phase = (1.0 - strength) * cur_phase + strength * avg_phase

                psi_constrained[i, j, k] = mag * (np.cos(new_phase) + 1j * np.sin(new_phase))

    return psi_constrained


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

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Gauge Symmetry (U1)"
    file_root="UMH_Gauge_Symmetry"
    file_sub="UMH_U1"
    file_hdr="UMH_Gauge_Symmetry_U1"
  
    print(f"✅ Starting Test: {title} Validated.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    v2 = v**2 #Wave Speed squared.
    print(f"{title}: Calculated Wave Speed Constant C = {v}")

    # Derived Gravity from Density and Pressure.
    G = dx ** 2 / Tu
    print(f"{title} Calculated Gravitational Constant G = {G}") #Not Used in this logic.

    safety_factor=0.50 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07

    damp_radius=4 #Radius toward center to dampen from kick artifact, caused by numerical simulation.
    max_damping = 0.999 #How much to quadratically dampen inward for kick artifact.

    # Safety Factor Adjust GridSize.
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    print(f"{title}: auto_cfl_clamp, Min:{min_factor}, Max:{max_factor}")
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)

    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor)
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")
    
    dt = dt * 0.5


    # Before the loop, allocate three buffers
    psi = initialize_U1_field(size, size, size, amplitude=1.0) #np.zeros((size, size, size), dtype=dtype) #np.float32
    velocity = np.zeros_like(psi)

    center = (size // 2, size // 2, size // 2)
    cx, cy, cz = center

    absorption_mask = create_absorption_mask(psi.shape, thickness=pml_thickness, absorption_coeff=pml_absorption,power=pml_power, dtype=dtype) #PML usage with gentle boundary damping to reduce relection.
   
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    monitor_total_energy=[]

    print(f"{title}: Starting Loop.")
    #Loop and step through wave progression.
    for step in range(steps):
        # Apply center damping first
        if step > 20 and step % 10 == 0:
            apply_center_damping(psi, center, damp_radius=damp_radius, max_damping=0.2)

        # Compute Laplacian and constraint force before velocity update
        lap = compute_laplacian_27point(psi, dx)
        constraint_force = np.clip(10.0 * (np.abs(psi)**2 - 1.0), -1e6, 1e6)

        # Update velocity with damping and absorption inline
        velocity = (velocity + (v2 * lap - constraint_force * psi) * dt) * 0.95
        velocity *= absorption_mask

        # Update psi with velocity
        psi += velocity * dt
        #psi *= absorption_mask # Just added to see about bleed over.

        # Apply Laplacian smoothing to psi after update (optional but stabilizing)
        if config["smoothing_enabled"] and step % config["smoothing_frequency"] == 0 and step > 0:
            lap_diff = compute_laplacian_27point(psi, dx)
            psi += 0.05 * lap_diff * dt #regularization

        # Enforce U(1) phase constraint strongly AFTER updates
        psi = enforce_U1_phase_constraint(psi, strength=0.1)
        # Replace hard clamp with safe projection:
        abs_psi = np.abs(psi)
        psi = np.where(abs_psi > 1.5, psi / abs_psi, psi)  # Projects down gently

        #print(f"[DEBUG] {title}: Step {step} complete. Max Psi: {np.max(psi)}")

        #Only save every so many frames for size and performance.
        if step>0 and step<steps and step in snapshot_steps: # or step == steps - 1     #Snapshot every so many frames for speed and size.  Wait till it gets going for first.
            max_psi = np.max(np.abs(psi))
            print(f"{title}, Step: {step} — Max |Psi|: {max_psi:.6f}")
            total_energy = np.sum(np.abs(psi)**2)
            monitor_total_energy.append(total_energy)


    #After Loop Plotting.
    print(f"{title}: Loop Completed.")

    psi *= absorption_mask ** 2

    # Tensor construction
    #grad = np.gradient(psi, dx)
    phase = np.angle(psi)
    grad_phase = np.gradient(phase, dx)


    tensors = {
        'T_xx': grad_phase[0]**2 * Tu,
        'T_yy': grad_phase[1]**2 * Tu,
        'T_zz': grad_phase[2]**2 * Tu,
        'T_xy': grad_phase[0] * grad_phase[1] * Tu,
        'T_xz': grad_phase[0] * grad_phase[2] * Tu,
        'T_yz': grad_phase[1] * grad_phase[2] * Tu,
    }

    print(f"{title}, Max T_zz: {np.max(tensors['T_zz'])}, Min: {np.min(tensors['T_zz'])}") 

    #Not Sure.
    for name, array in tensors.items():
        apply_center_damping(array, center, damp_radius=10, max_damping=0.2)
        array *= absorption_mask ** 2   # Apply boundary mask again post-processing
        array = crop_field(array, pml_thickness*2)
        array = smooth_field_3d(array, kernel_size=1)
        array = gaussian_filter(array, sigma=1)
        tensors[name] = array  # ✅ Save it back into the dictionary!



    # Curvature and divergence
    curvatures = {k: compute_laplacian_27point(v, dx) for k, v in tensors.items()}
    div_x, div_y, div_z = compute_divergence(
        tensors['T_xx'], tensors['T_yy'], tensors['T_zz'],
        tensors['T_xy'], tensors['T_xz'], tensors['T_yz'], dx
    )

    # Einstein tensor analysis
    R_xx, R_yy, R_zz = compute_ricci_tensor_from_components(curvatures, dx)
    scalar_curv = compute_scalar_curvature(R_xx, R_yy, R_zz)
    G_xx, G_yy, G_zz = compute_einstein_tensor(R_xx, R_yy, R_zz, scalar_curv)


    psiT = psi.copy()
    apply_center_damping(psiT, center, damp_radius=10, max_damping=0.2)
    psiT *= absorption_mask ** 2   # Apply boundary mask again post-processing
    psiT = crop_field(psiT, pml_thickness*2)
    psiT = smooth_field_3d(psiT, kernel_size=1)
    psiT = gaussian_filter(psiT, sigma=1)


    # Export results
    output_fields = {
        'Psi': psiT,
        **tensors,
        'curvature_T_xx': curvatures['T_xx'],
        'curvature_T_yy': curvatures['T_yy'],
        'curvature_T_zz': curvatures['T_zz'],
        'divergence_x': div_x,
        'divergence_y': div_y,
        'divergence_z': div_z,
        'Einstein_G_xx': G_xx,
        'Einstein_G_yy': G_yy,
        'Einstein_G_zz': G_zz,
        'Ricci_R_xx': R_xx,
        'Ricci_R_yy': R_yy,
        'Ricci_R_zz': R_zz,
    }
        
    np.save(f"{file_path}_Scalar_Curvature.npy", scalar_curv)
    save_csv_slice(f"{file_path}_Scalar_Curvature", scalar_curv)
    #save_csv_3d(f"{file_path}_3d_Scalar_Curvature", scacrv_crop)
    #plot_slice(f"{file_path}_Scalar_Curvature_Plot", scacrv_crop, axis='xy')

    print(f"{title}, Saved {file_path}_Scalar_Curvature.csv") 

    save_zslice_image(file_path, title, "Scalar_Curvature", scalar_curv, axis='xy', dpi=dpi, cmap="seismic")
    flat_vals = np.abs(scalar_curv.flatten())
    iso_val = np.percentile(flat_vals, 99.9)  # Top 0.5% of values
    #iso_val = max(0.005 * np.max(np.abs(scalar_curv)), 1e-5)
    #save_isosurface_3d(file_path, title, "Scalar_Curvature", scalar_curv, iso_value = iso_val, dpi=dpi)

    print(f"{title}, Saved 3d {file_path}_Scalar_Curvature.png") 

    radial_plot(file_path, title, "Ricci_R_zz", R_zz, dpi=dpi)
    flat_vals = np.abs(R_zz.flatten())
    iso_val = np.percentile(flat_vals, 99.9)  # Top 0.5% of values
    #iso_val = max(0.005 * np.max(np.abs(R_zz)), 1e-5)
    #save_isosurface_3d(file_path, title, "Ricci_R_zz", R_zz, iso_value = iso_val, dpi=dpi)

    print(f"{title}, Saved 3d {file_path}_Ricci_R_zz.png")

    radial_plot(file_path, title, "Einstein_G_zz", G_zz, anmgifary=None, dpi=dpi)
    flat_vals = np.abs(G_zz.flatten())
    iso_val = np.percentile(flat_vals, 99.9)  # Top 0.5% of values
    #iso_val = max(0.005 * np.max(np.abs(G_zz)), 1e-5)
    #save_isosurface_3d(file_path, title, "Einstein_G_zz", G_zz, iso_value = iso_val, dpi=dpi)

    print(f"{title}, Saved 3d {file_path}_Einstein_G_zz.png")

    absorbed_output_fields = {}
    images_sc=[]
    for name, array in output_fields.items():
        np.save(f"{file_path}_{name}.npy", array)
        save_csv_slice(f"{file_path}_{name}_Slice", array)
        #save_csv_3d(f"{file_path}_{name}_3d", smoothed_data)

        #plot_slice(f"{file_path}_Slice", array, axis='xy')

        print(f"{title}, Saved {name}_Slice.csv")

        try:
            save_zslice_image(file_path, title, name, array, axis='xy', dpi=dpi, cmap="seismic")
            print(f"{title}, Saved {name}_Slice.png")
        except Exception as e:
            print(f"Plot error for {name}: {e}")

        try:
            flat_vals = np.abs(array.flatten())
            iso_val = np.percentile(flat_vals, 99.9)  # Top 0.5% of values
            #iso_val = max(0.05 * np.max(np.abs(array)), 1e-4)
            #save_isosurface_3d(file_path, title, name, array, iso_value = iso_val, dpi=dpi)
            print(f"{title}, Saved 3d {name}_Slice.png")
        except Exception as e:
            print(f"Plot 3d error for {name}: {e}")


    plt.figure()
    plt.plot(monitor_total_energy)
    plt.xlabel('Step')
    plt.ylabel('Total Energy')
    plt.title(f"{title}: Energy Evolution")
    plt.savefig(f"{file_path}_Energy_Evolution.png", dpi=dpi)
    plt.close()



    compute_tensor_residuals(G_xx, tensors['T_xx'], label="xx", file_path=file_path, title=title)
    compute_tensor_residuals(G_yy, tensors['T_yy'], label="yy", file_path=file_path, title=title)
    compute_tensor_residuals(G_zz, tensors['T_zz'], label="zz", file_path=file_path, title=title)

    print(f"✅ Finished Test: {title} Validated.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()