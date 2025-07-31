"""
UMH_Wave_Propagation_Speed.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Wave Propagation Speed Test, using Stencil 27 Routines to Simulate a linear wave propagating outward from an initial momnentum.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""


import numpy as np
import numba
from numba import njit, prange
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import matplotlib.gridspec as gridspec

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import savgol_filter


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
        "LATTICE_SIZE": 384, #768,
        "TIME_STEPS": 256, #400

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":20, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary.
        "PML_ABORPTION":0.10, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.04
        "PML_POWER": 4,

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }




#Not Used, but can downsample for Iso 3d for speed.  Currently async so not a real issue.
def downsample(data, factor=4):
    return data[::factor, ::factor, ::factor]


#Optional, helps to provide running wave speed for console during loop steps.
def compute_windowed_linear_fit(radius_times, radius_values, min_time=2.0, window_size=100):
    times = np.array(radius_times)
    values = np.array(radius_values)

    # Select window based on min_time or last N points
    mask = times >= min_time
    if np.sum(mask) < 2:
        return None  # Not enough points yet

    times = times[mask][-window_size:]
    values = values[mask][-window_size:]

    A = np.vstack([times, np.ones_like(times)]).T
    m, _ = np.linalg.lstsq(A, values, rcond=None)[0]

    return m

#Detect Wavefront for speed test.
def detect_wavefront_outermost(energy_per_bin, radii, energy_cutoff=1e-12):
    indices = np.where(energy_per_bin > energy_cutoff)[0]
    
    if len(indices) == 0:
        return 0.0  # No detection
    
    detected_idx = indices[-1]  # Outermost
    return radii[detected_idx]

#Trim beginning and ending frames to ensure good data, and no outside influence of numerical buildup.
def trim_edges(data, trim_start=2, trim_end=2):
    return data[trim_start:-trim_end] if trim_end > 0 else data[trim_start:]


#Adjust for Numerical Artifical Jagged.
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


#Plot 3d Iso Image.
def plot_wavefront_isosurface(psi, file_path, isovalue=0.01, title="3D Wavefront Isosurface", dpi=300):
    psi_max = np.max(psi)
    psi_min = np.min(psi)

    if isovalue > psi_max or isovalue < psi_min:
        print(f"⚠️ Skipping isosurface plot — isovalue {isovalue} out of data range [{psi_min}, {psi_max}]")
        return

    verts, faces, normals, values = measure.marching_cubes(psi, level=isovalue)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, psi.shape[0])
    ax.set_ylim(0, psi.shape[1])
    ax.set_zlim(0, psi.shape[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(file_path, dpi=dpi)
    plt.close()

    #if anmgifary is not None: anmgifary.append(imageio.imread(filename))

#Save ZSlice of wave.
def save_zslice_image(zslice, step, size, center, file_path, title, dpi=300):
    padded_zslice = np.full((size, size), 1e-5)
    center_x = size // 2 - zslice.shape[0] // 2
    center_y = size // 2 - zslice.shape[1] // 2
    padded_zslice[center_x:center_x + zslice.shape[0], center_y:center_y + zslice.shape[1]] = np.abs(zslice)

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    # Set axes facecolor to match the colormap's low-end blue
    ax.set_facecolor(plt.get_cmap('seismic')(0))  # color at low-end of colormap

    im = ax.imshow(padded_zslice, 
                   cmap='seismic', 
                   norm=LogNorm(vmin=1e-5, vmax=1),
                   extent=[0, size, 0, size], 
                   origin='lower',
                   aspect='auto')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_title(f"{title}, Central Z-Slice at Step: {step}")

    cax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Strain Amplitude')

    plt.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


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

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Wave Propagation Speed"
    file_hdr="UMH_Wave_Propagation_Speed"
  
    print(f"✅ Starting Test: {title} High-Order Test (Linear Wave Simulation).")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    # Derived Wave Speed from Density and Pressure.
    v = np.sqrt(Tu / rho_u)
    print(f"{title}: Calculated Wave Speed Constant C = {v}")

    safety_factor=0.5 #Seems to work well here, as to ensure a good dt taking numerical stepping into account, 0.6 works well.  #0.25 10?  0.07
    energy_cutoff=1e-14 #Cutoff for Wave Edge detection.  This seems to work well. 1e-15

    damp_radius=4 #Radius toward center to dampen from kick artifact, caused by numerical simulation.
    max_damping = 0.9990 #How much to quadratically dampen inward for kick artifact.

    max_workers=8 #Worker threads for async image saves.  Math should be real slow down, so doesn't need to be to high.
    
    v2 = v**2 #Wave Speed squared.

    # Safety Factor Adjust GridSize.
    min_factor, max_factor = get_auto_cfl_clamp(size, steps)
    print(f"{title}: auto_cfl_clamp, Min:{min_factor}, Max:{max_factor}")
    safety_factor = get_adaptive_safety_factor(size, base=safety_factor, min_factor=min_factor, max_factor=max_factor)

    dt=calculate_cfl_dt(v,dx,dimensions=3,safety_factor=safety_factor)
    print(f"{title}: Using [safety_factor={safety_factor}], Derived [DT={dt}] from [DX={dx}] and [V={v}], which is Derived from [v=np.sqrt(Tu / rho_u)={v}=np.sqrt({Tu}/{rho_u})].")

    # Before the loop, allocate three buffers
    psi = np.zeros((size, size, size), dtype=dtype) #np.float32
    psi_prev = np.zeros_like(psi)
    psi_next = np.zeros_like(psi)

    center = (size // 2, size // 2, size // 2)
    cx, cy, cz = center

    #gb=30  #causes weird problems.  Give it one more try now that things are all correct.  Not really needed for this test though.
    #initialize_gaussian(psi, center, width=gb) #Do not use, causes strain across the grid.  Test one more time then remove.
    #psi_prev = np.copy(psi)
    
    #Kick, to start the wave.
    damping_steps=8 #Use to remove central artifacting from kick.
    kick_radius=2 #Radius of spherical kick.
    kick_strength=0.0075 #How strong of a kick, amplitude.
    initial_energy=apply_gaussian_kick(psi_prev, psi, center, kick_radius=kick_radius, kick_strength=kick_strength) #Trigger Gaussian Kick, reduces simulation artifacts.
    print(f"{title}: Initial Total Energy After Small Radius - Point Kick: {initial_energy}")

    #absorption_mask = create_absorption_mask(psi.shape, thickness=15, absorption_coeff=0.1, dtype=dtype)
    #absorption_mask = np.ones((size, size, size), dtype=dtype)  # No absorption
    absorption_mask = create_absorption_mask(psi.shape, thickness=pml_thickness, absorption_coeff=pml_absorption,power=pml_power, dtype=dtype) #PML usage with gentle boundary damping to reduce relection.
   
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    images_sc,images_3d=[],[]
    cumulative_energy_log,energy_log,radius_log = [],[],[]
    times = []
            
    x, y, z = (np.arange(size) - cx), (np.arange(size) - cy), (np.arange(size) - cz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    radii = np.arange(0, int(np.max(R))+1, 1) # Radii used as bin edges (0 to max(R), inclusive)

    # Before the loop
    Xt, Yt, Zt = np.indices(psi.shape)
    dist_grid = np.sqrt((Xt - center[0])**2 + (Yt - center[1])**2 + (Zt - center[2])**2)

    bin_edges = np.arange(0, np.max(R) + 0.02, 0.02)  # Finer binning
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    #Loop and step through wave progression.
    for step in range(steps):
        update_wave_27_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask)  #Try next with the tiny adjustment recommended for center_coeff=2.325 for cmb.
        #update_wave_49_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask) #Try next to see if this works.
        #update_wave_7_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask) #With PML.
        # Swap references — this is O(1) and avoids .copy()
        psi_prev, psi, psi_next = psi, psi_next, psi_prev

        # Artifact center damping from Kick, only apply larger than kick as to not impact kick.
        if (kick_radius+step+0) >= damp_radius and damping_steps>0 and step % 1 == 0:
            apply_center_damping(psi, center, damp_radius = damp_radius, max_damping = max_damping) #0.98
            psi[center[0]:center[1]:center[2]]=0
            damping_steps-=1
        
        #Use Histogram bin to log.
        hist, _ = np.histogram(R.ravel(), bins=bin_edges, weights=(psi ** 2).ravel())
        smoothed_hist = smooth_data_numba(hist, window=3)  # Use lighter smoothing

        detected_radius = detect_wavefront_outermost(smoothed_hist, bin_centers, energy_cutoff=energy_cutoff)  # Detect Wave edge for speed test.

        # Save Radius and Energy.
        radius_log.append((step * dt, detected_radius))
        energy_log.append((step * dt, np.sum(hist)))

        # Save energy bins for cumulative later
        cumulative_energy_log.append(hist.copy())

        times.append(step * dt)

        #Only save every so many frames for size and performance.
        if step in snapshot_steps: # or step == steps - 1 step>0 and step<steps and    #Snapshot every so many frames for speed and size.  Wait till it gets going for first.
            psi_temp=psi.copy(order='C')  #Make copy for async saves.
            zslice=psi_temp[:, :, center[2]]  #Make zslice copy for async snapshot save.

            filename=f"{file_path}_Slice_Step_{step}.png"
            images_sc.append(filename)
            #save_zslice_image(zslice, step, size, center, f"{file_path}_Slice_Step_{step}.png", title, dpi=dpi)
            
            with ProcessPoolExecutor(max_workers=max_workers) as snapshot_executor:
                snapshot_executor.submit(save_zslice_image, zslice, step, size, center, filename, title, dpi=dpi)

            adaptive_isovalue = np.max(psi) * 0.05  # 10% of max
            if adaptive_isovalue > 1e-8:  # Avoid near-zero isovalue failure.
                filename3d=f"{file_path}_3d_Step_{step}.png"
                images_3d.append(filename3d)
                with ProcessPoolExecutor(max_workers=max_workers) as snapshot_executor_3d:
                    snapshot_executor_3d.submit(plot_wavefront_isosurface, psi_temp, filename3d, isovalue=adaptive_isovalue, title=f"{title} 3D Wavefront Isosurface Step: {step}",dpi=dpi)

            # Diagnostic
            #np.savetxt(f"{file_path}_Energy_vs_Radius_Step_{step}.csv", np.column_stack((radii, cumulative_energy_log)),delimiter=',', header='Radius, CumulativeEnergy')

            #Calculate current wave speed for diagnostics.
            radius_times = [entry[0] for entry in radius_log]
            radius_values = [entry[1] for entry in radius_log]
            min_time = 0.0  # Disable for testing
            window_size = min(len(radius_log), 50)  # Fit on last 50 or less if fewer exist
            fit_speed = compute_windowed_linear_fit(radius_times, radius_values, min_time=min_time, window_size=window_size)
            if fit_speed is not None:
                print(f"{title}, Step: {step}: Linear Fit Wave Speed ≈ {fit_speed:.4f}")
            else:
                print(f"{title}, Step: {step}: Linear Fit Wave Speed ≈ [Not enough data yet]")


    #After Loop Plotting.
    snapshot_executor.shutdown(wait=True)
    snapshot_executor_3d.shutdown(wait=True)

    imganm=[]
    for filename in images_sc:
        imganm.append(imageio.imread(filename))

    imageio.mimsave(f"{file_path}_Slice_Evolution.gif", imganm, fps=2)

    imganm=[]
    for filename in images_3d:
        imganm.append(imageio.imread(filename))

    imageio.mimsave(f"{file_path}_3d_Evolution.gif", imganm, fps=2)

    radius_log = np.array(radius_log)
    times = radius_log[:, 0]
    radii = radius_log[:, 1]
    
    times=trim_edges(times,trim_start=4,trim_end=4)
    radii=trim_edges(radii,trim_start=4,trim_end=4)

    radii = smooth_data_numba(radii, window=5)  # For plotting only

    # Average energy bins over last steps and smooth
    if len(cumulative_energy_log) >= 6:
        avg_energy_bins = sum(cumulative_energy_log[-6:]) / 6
    else:
        avg_energy_bins = cumulative_energy_log[-1]

    avg_energy_bins_smoothed = savgol_filter(avg_energy_bins, window_length=11, polyorder=2)
    smoothed_avg_energy_bins = smooth_data_numba(avg_energy_bins_smoothed, window=5)  # For plotting only
    smoothed_avg_cumulative_energy = np.cumsum(smoothed_avg_energy_bins)

    np.savetxt(f"{file_path}_Energy_Log.csv",
               energy_log, delimiter=",", header="Time(s),TotalEnergy", comments='')

    np.savetxt(f"{file_path}_Radius_Log.csv",
           radius_log, delimiter=",", header="Time(s),MaxRadius", comments='')

    # Adjust for Initial Velocity Kick by ignoring first snapshot or two
    start_idx = 0
    if len(radius_log) > start_idx + 1:
        simulated_speed = (radius_log[-1][1] - radius_log[start_idx][1]) / (radius_log[-1][0] - radius_log[start_idx][0])
        print(f"{title} Simulated Wave Speed (adjusted, start at index {start_idx}): {simulated_speed:.4f} (Expected: {v})")
        if simulated_speed > v*1.05: #Check ensure speed is correct, with a 0.05% tolerance for artifacts.
            print("⚠️ Warning: Initial momentum kick exceeds recommended physical limits, this artificially can cause invalid results.")
        else :
            if simulated_speed < v*0.95:  #Check ensure speed is correct, with a 0.05% tolerance for artifacts.
                print("⚠️ Warning: Speed is too slow in results.")

        # Stability check over multiple starting points
        print(f"{title} Wave Speed Stability Check:")
        for idx in range(1, min(5, len(radius_log) - 1)):
            speed_est = (radius_log[-1][1] - radius_log[idx][1]) / (radius_log[-1][0] - radius_log[idx][0])
            print(f"  Start idx {idx}: Speed = {speed_est:.4f}")
    else:
        print(f"❗ Not enough data points to compute wave speed with start index {start_idx}.")


    # Unpack into separate lists
    times2, energies = zip(*energy_log)
    times2=trim_edges(times2,trim_start=2,trim_end=2)
    energies=trim_edges(energies,trim_start=2,trim_end=2)

    # Plot
    plt.figure(figsize=(8, 5))
    #plt.plot(times2, energies, marker='o')
    plt.plot(times2, moving_average(energies, window_size=5), marker='o')
    plt.title(f"{title}: Total Strain Energy vs Time")
    plt.xlabel('Time (s)')
    plt.ylabel('Total Strain Energy')
    plt.grid(True)
    plt.savefig(f"{file_path}_Total_Energy_vs_Time.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(times, radii, marker='o')
    plt.title(f"{title}: Wavefront Radius vs Time")
    plt.xlabel('Time (s)')
    plt.ylabel('Radius (grid units)')
    plt.grid(True)
    plt.savefig(f"{file_path}_Wavefront_Radius_vs_Time.png", dpi=dpi)
    plt.close()

    plt.figure()
    plt.plot(bin_centers, smoothed_avg_cumulative_energy)
    plt.xlabel('Radius (grid units)')
    plt.ylabel('Cumulative Strain Energy')
    plt.title(f"{title}: Cumulative Strain Energy vs Radius")
    plt.grid(True)
    plt.savefig(f"{file_path}_Cumulative_Strain_Energy_vs_Radius.png", dpi=dpi)
    plt.close()

    plt.figure()
    plt.loglog(bin_centers[1:], smoothed_avg_cumulative_energy[1:])  # Skip very first bin to avoid radius=0.
    plt.xlabel('Radius (log scale)')
    plt.ylabel('Cumulative Energy (log scale)')
    plt.title(f"{title} Log-Log: Cumulative Strain Energy vs Radius")
    plt.grid(True, which="both")
    plt.savefig(f"{file_path}_Cumulative_Strain_Energy_vs_Radius_LogLog.png", dpi=dpi)
    plt.close()

    # Align times and radii first
    min_len = min(len(times), len(radii))
    times = times[:min_len]
    radii = radii[:min_len]

    #speed = np.gradient(radii, times)
    #smoothed_speed = moving_average(speed, window_size=10)

    smoothed_radii = savgol_filter(radii, window_length=11, polyorder=2)
    speed = np.gradient(smoothed_radii, times)
    smoothed_speed = moving_average(speed, window_size=10)

    plt.figure(figsize=(8, 5))
    plt.plot(times[-len(smoothed_speed):], smoothed_speed, marker='o')
    plt.title(f"{title} Instantaneous Wave Speed vs Time")
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (grid units per second)')
    plt.grid(True)
    plt.savefig(f"{file_path}_Instantaneous_Speed_vs_Time.png", dpi=dpi)
    plt.close()


    print(f"✅ Finished Test: {title} High-Order Test (Linear Wave Simulation).")

if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()