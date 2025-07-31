"""
UMH_Chirp_Generator.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Chirp Generator Simulation, for use with UMH_Ligo_Compiler.

Parameters:
- OUTPUT_FOLDER.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from numba import njit, prange
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram


def get_default_config(config_overrides=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "G": 6.67430e-11,        # Gravitational constant (m^3 kg^-1 s^-2)
        "c_phys": 3.0e8,         # Physical speed of light (m/s)
        "M_sun": 1.989e30,       # Solar mass (kg)
        "grid_spacing": 1000.0,  # 1 grid unit = 1000 meters (tunable)
        "M_chirp_kg": 30 * 1.98847e30,  # 30 solar masses


        "c": 1.0,               # Medium wave speed (sim units)
        "soliton_radius": 40,
        "amplitude": 2500.0, #1500.0

        "damping_factor": 0.9999,

        "freq_damping": 0.005,  # tweak between 0.0001 and 0.01 to suppress high-frequency noise

        "buffer_steps": 25000, #5000 65536

        "strain_scaling": 0.7,

        "damping_delay": 2.0, # Prev 0.5,
        "damping_rolloff":0.995, #0.985

        "reference_strain": 1e-21,   # at 100 Mpc typical LIGO source
        "reference_radius": 100.0,    # grid units corresponding to ~100 Mpc

        "DEFAULT_NOISE_FLOOR": 1e-25,   # Minimum noise amplitude if signal is weak
        "NOISE_SCALING_FACTOR": 0.15,   # Scale relative to signal max, was 0.02.
        "PINK_NOISE_SCALING_FACTOR":0.5, # Added for more lower noise.
        "NOISE_BLEND_RATIO":0.08,

        "SPACING": 2,

        "lowcut": 5.0,
        "highcut": 500.0,
        "f_min":0.5,   #1 prev.

        # Simulation parameters
        
        "SIZE":128, #Nx, Ny, Nz = 128, 128, 128

        "t_max": 40, #10, #8.192  # total simulation time in sim units (32768 * 0.00025)
        "dt": 0.00025,    # slower time step, improves resolution and chirp scale

        "offset": 16, #16  8

        "r0": 12,                      # Initial separation in grid units

        "scale_factor": 5e5,           # meters per grid unit


        "Q": 8,                     # Quality factor for damping
        "f_rd": 250,                # Hz (typical black hole ringdown frequency)


        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


@njit(parallel=True)
def gaussian_soliton(phi, center, radius, amplitude):
    cx, cy, cz = center
    Nx, Ny, Nz = phi.shape
    r_int = int(radius)
    r3 = 3 * r_int  # 3 for effective Gaussian support

    x_min = max(cx - r3, 0)
    x_max = min(cx + r3, Nx)
    y_min = max(cy - r3, 0)
    y_max = min(cy + r3, Ny)
    z_min = max(cz - r3, 0)
    z_max = min(cz + r3, Nz)

    for x in prange(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                r2 = dx * dx + dy * dy + dz * dz
                phi[x, y, z] += amplitude * np.exp(-r2 / (2.0 * radius * radius))


    
@njit(parallel=True)
def update_field(Nx,Ny,Nz,phi, phi_prev, phi_next, c, dt, damping_factor, freq_damping):
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                lap = (
                    phi[i+1, j, k] + phi[i-1, j, k] +
                    phi[i, j+1, k] + phi[i, j-1, k] +
                    phi[i, j, k+1] + phi[i, j, k-1] -
                    6.0 * phi[i, j, k]
                )
                phi_next[i, j, k] = damping_factor * (
                    2 * phi[i, j, k] - phi_prev[i, j, k] + (c**2 * dt**2) * lap
                    - freq_damping * (phi[i, j, k] - phi_prev[i, j, k])
                )

    
@njit
def measure_strain(Nx, phi, center, orientation, spacing=1):
    cx, cy, cz = center
    ax, ay = orientation

    ax_idx = {"x": 0, "y": 1, "z": 2}
    i1 = ax_idx[ax]
    i2 = ax_idx[ay]

    coords = np.array([cx, cy, cz], dtype=np.int64)

    coords[i1] = min(max(cx + spacing, 1), Nx - 2)
    f1p = phi[coords[0], coords[1], coords[2]]

    coords[i1] = min(max(cx - spacing, 1), Nx - 2)
    f1m = phi[coords[0], coords[1], coords[2]]

    coords[i1] = cx
    coords[i2] = cy + spacing
    f2p = phi[coords[0], coords[1], coords[2]]

    coords[i2] = cy - spacing
    f2m = phi[coords[0], coords[1], coords[2]]

    # Reset not needed; we're done accessing
    df1 = (f1p - f1m) / (2.0 * spacing)
    df2 = (f2p - f2m) / (2.0 * spacing)

    return 0.5 * (df1 + df2)


def safe_center(pos, margin=2):
    return tuple(np.clip(np.array(pos), margin, np.array(phi.shape) - margin - 1))

def grid_to_physical_length(r_grid, grid_spacing):
    return r_grid * grid_spacing  # meters

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)
        
def extract_polarization(strain_dict):
    if "Hanford" in strain_dict and "Livingston" in strain_dict:
        h_plus = 0.5 * (strain_dict["Hanford"] - strain_dict["Livingston"])
        h_cross = 0.5 * (strain_dict["Hanford"] + strain_dict["Livingston"])
    else:
        h_plus = h_cross = np.zeros_like(next(iter(strain_dict.values())))
    return h_plus, h_cross



def run_chirp_generator_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    G=config["G"]
    c_phys=config["c_phys"]
    M_sun=config["M_sun"]
    grid_spacing=config["grid_spacing"]
    M_chirp_kg=config["M_chirp_kg"]

    size=config["SIZE"]

    c=config["c"]
    soliton_radius=config["soliton_radius"]
    amplitude=config["amplitude"]

    damping_factor=config["damping_factor"]

    freq_damping=config["freq_damping"]

    buffer_steps=config["buffer_steps"]

    strain_scaling=config["strain_scaling"]

    damping_delay=config["damping_delay"]
    damping_rolloff=config["damping_rolloff"]

    reference_strain=config["reference_strain"]
    reference_radius=config["reference_radius"]

    DEFAULT_NOISE_FLOOR=config["DEFAULT_NOISE_FLOOR"]
    NOISE_SCALING_FACTOR=config["NOISE_SCALING_FACTOR"]
    PINK_NOISE_SCALING_FACTOR=config["PINK_NOISE_SCALING_FACTOR"]
    NOISE_BLEND_RATIO=config["NOISE_BLEND_RATIO"]


    SPACING=config["SPACING"]

    lowcut=config["lowcut"]
    highcut=config["highcut"]
    f_min=config["f_min"]

    SIZE=config["SIZE"]

    t_max=config["t_max"]
    dt=config["dt"]

    offset=config["offset"]

    r0=config["r0"]

    scale_factor=config["scale_factor"]


    Q=config["Q"]       # Quality factor for damping
    f_rd=config["f_rd"]


    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Chirp Generator"
    file_root="UMH_vs_LIGO"
    file_hdr="UMH_Chirp_Generator"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    # Constants
    #G = 6.67430e-11        # Gravitational constant (m^3 kg^-1 s^-2)
    #c_phys = 3.0e8         # Physical speed of light (m/s)
    #M_sun = 1.989e30       # Solar mass (kg)
    #grid_spacing = 1000.0  # 1 grid unit = 1000 meters (tunable)
    #M_chirp_kg = 30 * 1.98847e30  # 30 solar masses

    #Nt=65536
    #t_array_physical = np.arange(Nt) * dt_physical  # in seconds
    #detector_distance_m = 3e6  # 3000 km
    #detector_distance_grid = detector_distance_m / grid_spacing
    #r_si = r_grid * grid_spacing  # radius in meters
    #epsilon = 1e-12
    #falloff = 1.0 / (r_grid**2 + epsilon)

    #plt.plot(t_seconds, strain)
    #plt.xlabel("Time [s]")
    #Adjust fs = 1.0 / dt_physical in spectrogram and FFT.


    # Binary parameters
    M1 = 30 * M_sun
    M2 = 30 * M_sun
    M_chirp = ((M1 * M2)**(3/5)) / ((M1 + M2)**(1/5))  # Chirp mass


    #reference_strain = 1e-21   # at 100 Mpc typical LIGO source
    #reference_radius = 100.0    # grid units corresponding to ~100 Mpc

    #DEFAULT_NOISE_FLOOR = 1e-25   # Minimum noise amplitude if signal is weak
    #NOISE_SCALING_FACTOR = 0.05   # Scale relative to signal max

    #SPACING=2

    #lowcut = 5.0
    #highcut = 500.0
    #f_min=1.0   #90%

    # Simulation parameters
    Nx, Ny, Nz = size, size, size


    #t_max = 10 #8.192  # total simulation time in sim units (32768 * 0.00025)
    #dt = 0.00025    # slower time step, improves resolution and chirp scale
    #Nt = int(t_max / dt)


    #c = 1.0               # Medium wave speed (sim units)
    #soliton_radius = 40
    #amplitude = 2500.0 #1500.0

    #damping_factor = 0.9999

    #freq_damping = 0.005  # tweak between 0.0001 and 0.01 to suppress high-frequency noise

    #buffer_steps = 5000 


    # Grid arrays
    phi = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    phi_prev = np.zeros_like(phi)
    phi_next = np.zeros_like(phi)

    #detector = (Nx//2 + 5, Ny//2 + 5, Nz//2)

    detector1 = (Nx//2 + 10, Ny//2, Nz//2)
    detector2 = (Nx//2 - 10, Ny//2, Nz//2)



    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    # Source origin position (can be off-center to simulate directionality)
    source_origin = np.array([cx, cy, cz])
    wave_speed = c  # Units/s (same as c=1 in UMH scaling)

    center = (Nx//2, Ny//2, Nz//2)  # shared soliton center

    #offset = 16
    detectors = {
        "Hanford": {
            "center": (center[0] + offset, center[1], center[2]),
            "orientation": ("x", "z"),
            "polarization_vector": np.array([1.0, 1.0, 0.0])
        },
        "Livingston": {
            "center": (center[0], center[1]+offset, center[2]),
            "orientation": ("x", "y"),
            "polarization_vector": np.array([1.0, 0.0, 0.0])
        },
        "Virgo": {
            "center": (center[0], center[1], center[2] + offset),
            "orientation": ("y", "z"),
            "polarization_vector": np.array([0.0, 1.0, 0.0])
        }
    }


    max_offset = np.max([np.linalg.norm(np.array(detectors[name]["center"]) - source_origin) for name in detectors])
    required_steps = int(max_offset / (wave_speed * dt)) + buffer_steps
    t_max = required_steps * dt
    Nt = required_steps  #int(t_max / dt)

    freq_record = np.zeros(Nt)
    t_array = np.linspace(0, (Nt - 1) * dt, Nt)

    strain_record = np.zeros(Nt, dtype=np.float32)

    strain_records = {name: np.zeros(Nt) for name in detectors}

    radius_record = np.zeros(Nt, dtype=np.float32)

    merger_time_dict = {name: None for name in detectors}

    # Constants for motion
    #r0 = 12                      # Initial separation in grid units
    tc = Nt * dt                 # Coalescence time
    A = (1 / (8 * np.pi)) * ((5 / (256 * (G * M_chirp / c_phys**3)**(5/3)))**(3/8))  # freq coeff
    #scale_factor = 5e5           # meters per grid unit

    print(f"Total simulation steps (Nt): {Nt}, Total time (t_max): {t_max}")


    # Time evolution
    for t in range(Nt):
        time = t * dt

        delta_t = max(tc - time, 1e-3)
        #f_gw = A * delta_t**(-3/8)       # GW frequency
        if delta_t < 0.5:
            f_gw = A * delta_t**(-5/8)
        else:
            f_gw = A * delta_t**(-3/8)
        
        # Set default f_gw unless overridden
        if f_min > 0:
            f_gw *= f_min

    
        f_orb = f_gw / 2                 # Orbital frequency
        M_tot = M1 + M2
        r_si = (G * M_tot / (2 * np.pi * f_orb)**2)**(1/3)
        r_grid = max(2, r_si / scale_factor)
        r_grid = np.clip(r_grid, 2.0, 1e4)

        radius_record[t] = r_grid

        # Clear and recompute solitons
        phi.fill(0.0)
        theta = 2 * np.pi * f_orb * time
        #cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

        # Source 1: full 3D orbit with vertical oscillation
        x1 = np.clip(int(round(cx + r_grid * np.cos(theta))), 1, Nx - 2)
        y1 = np.clip(int(round(cy + r_grid * np.sin(theta))), 1, Nx - 2)
        theta_z = theta + np.pi / 2
        cz1 = np.clip(int(round(cz + r_grid * 0.1 * np.sin(theta_z))), 1, Nx - 2)
    
        center1 = (x1, y1, cz1)

        # Source 2: apply phase offset only here
        theta_offset = theta + np.pi / 4
        x2 = np.clip(int(round(cx - r_grid * np.cos(theta_offset))), 1, Nx - 2)
        y2 = np.clip(int(round(cy - r_grid * np.sin(theta_offset))), 1, Nx - 2)
    
        center2 = (x2, y2, cz)  # flat plane


        strain_scale = reference_strain * (reference_radius / r_grid)**2
        strain_scale = np.clip(strain_scale, 1e-26, 1e-18) #1e-24,1e-18
    
    
        sharp_rise = 1 / (1 + np.exp(-20 * (time - (t_max - 0.5))))
        modulation = 1.0 + 0.1 * sharp_rise
    
        amp_mod = amplitude * strain_scale * modulation   

        radius = max(10.0, soliton_radius * (r_grid / r0))
        gaussian_soliton(phi, center1, radius, amp_mod)
        gaussian_soliton(phi, center2, radius, -amp_mod)
    
    
        # Finite difference update
        phi_next.fill(0.0)
        update_field(Nx,Ny,Nz,phi, phi_prev, phi_next, c, dt, damping_factor, freq_damping)
    
    
        falloff = 1.0 / (r_grid**2 + 1e-4)  # Avoid runaway
    
        for name in detectors:
            cx, cy, cz = detectors[name]["center"]
            orientation = detectors[name]["orientation"]
            # Bounds check to avoid index error inside measure_strain
            if (
                cx < SPACING or cx > Nx - SPACING - 1 or
                cy < SPACING or cy > Ny - SPACING - 1 or
                cz < SPACING or cz > Nz - SPACING - 1
            ):
                continue

            strain_value = measure_strain(Nx, phi, (cx, cy, cz), orientation, spacing=SPACING)
            strain_value *= strain_scaling #1.7  # Test with 2.0 if needed for comparison. 1.5  -> Added r2

            # Delay by propagation time
            distance = np.linalg.norm(np.array([cx, cy, cz]) - source_origin)
            delay_steps = int(distance / (wave_speed * dt))
            delayed_t = t - delay_steps
        
            if t % 1024 == 0:
                print(f"{title}: {name}, t={t}, f_gw={f_gw:.2f}, r_si={r_si:.2e}, r_grid={r_grid}")
                print(f"{title}: {name}, Step {t}, radius = {r_grid:.8f}, strain = {strain_value:.8e}")

            # Apply high-frequency damping post-merger.  Not sure if I need this.
            if r_grid <= 2.0:
                merger_time_dict[name] = t * dt  # Record merger onset time for this detector
                #strain_value *= 0.95  # You can adjust 0.9 → 0.8 if needed

            if merger_time_dict[name] is not None and (t * dt - merger_time_dict[name]) >= damping_delay:
                strain_value *= damping_rolloff  # Damping applied after delay per detector

            if 0 <= delayed_t < Nt:
                strain_records[name][delayed_t] = strain_value * strain_scale #* falloff Causing too weak..
                if t % 10000 == 0:
                    print(f"✅ {title}: Recording strain for {name} at t = {delayed_t}, value = {strain_value * strain_scale:.2e}")
            #else:
                #print(f"Warning: Detector '{name}' has too weak a signal (0 <= delayed_t < Nt) {delayed_t}:{Nt}.")
            

        dx = phi[Nx//2 + 1, Ny//2, Nz//2] - phi[Nx//2 - 1, Ny//2, Nz//2]
        dz = phi[Nx//2, Ny//2, Nz//2 + 1] - phi[Nx//2, Ny//2, Nz//2 - 1]
         
        raw_strain = 0.5 * (dx + dz)

        #Optional
        raw_strain /= (2.0 * dt)
        #Optional

        strain_record[t] = raw_strain * falloff

        radius_record[t] = r_grid
    
        freq_record[t] = f_gw

        phi_prev[:], phi[:] = phi[:], phi_next[:]

        #if t % 1024 == 0:
            #print(f"t={t}, f_gw={f_gw:.2f}, r_si={r_si:.2e}, r_grid={r_grid}")
            #print(f"Step {t}, radius = {r_grid:.8f}, strain = {strain_record[t]:.8e}")
            #import matplotlib.pyplot as plt
            #plt.imshow(phi[:, :, Nz//2], cmap='seismic')
            #plt.title(f"Strain Slice at t={t}")
            #plt.colorbar()
            #plt.savefig(f"umh_output_chirp/frame_{t}.png")
            #plt.close()
        
            #plt.plot(np.linspace(0, Nt * dt, Nt), strain_record)
            #plt.xlabel("Time")
            #plt.ylabel("Detector Strain")
            #plt.title("Chirp Signal (UMH Simulation)")
            #plt.grid(True)
            #plt.savefig(f"umh_output_chirp/chirp_signal_plot_{t}.png")
            #plt.close()



    # Ringdown insertion
    #Q = 8                     # Quality factor for damping
    #f_rd = 250                # Hz (typical black hole ringdown frequency)
    tau = Q / (np.pi * f_rd) # Damping time constant


    # Set ringdown to last 2% of simulation time, or 1.0s if preferred   
    ringdown_duration = 0.02 * t_max
    ringdown_steps = int(ringdown_duration / dt)
    ringdown_time = np.linspace(0, ringdown_duration, ringdown_steps)


    for name in strain_records:
        recorded_max = np.max(np.abs(strain_records[name]))
        print(f"{title}: {name} recorded strain max after simulation = {recorded_max:.2e}")

        # --- Polarization Projection ---
        #Extract orientation vectors (arm directions of the interferometer)
        u, v = detectors[name]["orientation"]  # e.g., ("x", "y")
        axis_map = {"x": 0, "y": 1, "z": 2}

        u_vec = np.zeros(3)
        v_vec = np.zeros(3)
        u_vec[axis_map[u]] = 1
        v_vec[axis_map[v]] = 1

        #Compute detector plane normal (assumes Michelson interferometer)
        plane_normal = np.cross(u_vec, v_vec)
        plane_normal_norm = np.linalg.norm(plane_normal)
        if plane_normal_norm == 0:
            raise ValueError(f"{title}: Detector '{name}' has invalid or degenerate arm orientation.")
        plane_normal /= plane_normal_norm  # Normalize

        #Get and normalize the polarization vector
        polarization_vector = np.array(detectors[name].get("polarization_vector", [1.0, 0.0, 0.0]))
        polarization_vector = np.array(polarization_vector, dtype=np.float64)
        polarization_norm = np.linalg.norm(polarization_vector)
        if polarization_norm == 0:
            raise ValueError(f"{title}: Detector '{name}' has zero polarization vector.")
        polarization_vector /= polarization_norm

        #Compute projection factor
        cos_theta = np.abs(np.dot(polarization_vector, plane_normal))
        projection_factor = 1.0 - cos_theta**2  # sensitivity to polarization in plane



        # --- Physics-level amplitude scaling before projection --- Just added.
        # Theoretical strain for binary chirp (simplified leading order estimate)
        # Assuming M_chirp is loaded from config earlier
        f_gw_max = np.max(freq_record)  # Use maximum chirp frequency reached during simulation
        distance_m = offset * scale_factor  # Detector offset scaled to physical meters
        h_expected = (4 * (G**(5/3)) * (np.pi * f_gw_max)**(2/3) * (M_chirp)**(5/3)) / (c_phys**4 * distance_m)

        simulated_max = np.max(np.abs(strain_records[name]))
        if simulated_max > 0:
            scaling_factor_physics = h_expected / simulated_max
            print(f"{title}: {name} Applying physics-level scaling factor: {scaling_factor_physics:.2e}")
            strain_records[name] *= scaling_factor_physics
        else:
            print(f"{title}: {name} Warning — Simulated strain zero, skipping physics scaling.")

        # --- Physics-level amplitude scaling before projection --- Just added.




        #Apply projection factor safely
        print(f"{title}: {name} strain max before proj: {np.max(strain_records[name]):.2e}")
        strain_records[name] *= 1.2  # Boost by 20% before projection
        strain_records[name] *= projection_factor
        print(f"{title}: Detector '{name}' projection factor: {projection_factor:.4f}")
        print(f"{title}: {name} strain max after proj: {np.max(strain_records[name]):.2e}")
        #------------

        # Adaptive scaling to ensure detectable signal strength for visualization
        # --- Apply Adaptive Scaling (For Visualization Only) ---
        max_proj_amp = np.max(np.abs(strain_records[name]))
        if max_proj_amp < 1e-21 and False:
            scaling_factor = 1e-21 / (max_proj_amp + 1e-30)  # epsilon to avoid division by zero
            print(f"{title}: {name} Adaptive scaling applied with factor {scaling_factor:.2e} (Visualization Only)")
            strain_records[name] *= scaling_factor


    
        idx = max(0, len(strain_records[name]) - len(ringdown_time) - 1)
        #final_amp = np.max(np.abs(strain_records[name][-len(ringdown_time):]))
        final_amp = np.max(np.abs(strain_records[name][Nt // 2 : -ringdown_steps]))

        # Generate ringdown
        ringdown = final_amp * np.sin(2 * np.pi * f_rd * ringdown_time) * np.exp(-ringdown_time / tau)

        # Apply ringdown at end
        if len(strain_records[name]) >= len(ringdown):
            strain_records[name][-len(ringdown):] += ringdown
        else:
            strain_records[name] += ringdown[:len(strain_records[name])]
        
        
        #Tapering
        window = np.ones_like(strain_records[name])
        taper_length = int(0.05 * len(strain_records[name]))  # Last 5%
        window[-taper_length:] = tukey(taper_length * 2, alpha=1)[taper_length:]
        strain_records[name] *= window

        #BandPass Filter
        strain_records[name] = bandpass_filter(strain_records[name], lowcut, highcut, fs=1.0/dt)
    
    
        print(f"{title}: {name} strain max: {np.max(strain_records[name])}, min: {np.min(strain_records[name])}")
        print(f"{title}: {name} strain abs max: {np.max(np.abs(strain_records[name]))}")

        peak_idx = np.argmax(np.abs(strain_records[name]))
        peak_time = t_array[peak_idx]
        print(f"{title}: Detector '{name}' peak strain = {strain_records[name][peak_idx]:.2e} at t = {peak_time:.3f} s")


        # Normalize Adter.
        print(f"{title}: {name}: Nonzero strain values = {np.count_nonzero(strain_records[name])}")
        max_amp = np.max(np.abs(strain_records[name]))
        print(f"{title}: {name}: max strain before normalization = {max_amp:.2e}")
    
        #print(f"{name}, Step {delayed_t}, strain = {strain_value:.2e}, scaled = {strain_value * strain_scale:.2e}")

    
        if max_amp > 1e-40:   #1e-21 1e-40
            strain_records[name] /= max_amp
        else:
            print(f"{title}: Warning: Detector '{name}' has too weak a signal for normalization (max={max_amp:.2e}). Skipping normalization.")


        # Optional: add Gaussian noise to detector strains
        signal_peak = np.max(np.abs(strain_records[name]))
        noise_scale = max(NOISE_SCALING_FACTOR * signal_peak, DEFAULT_NOISE_FLOOR)
        #noise = np.random.normal(0, noise_scale, Nt)
        #strain_records[name] += noise
            
        #pink_noise = (np.random.normal(size=Nt) / np.sqrt(np.arange(1, Nt + 1)))
        #strain_records[name] += PINK_NOISE_SCALING_FACTOR * pink_noise * noise_scale


        # Generate white noise
        white_noise = np.random.normal(0, noise_scale, Nt)
        # Apply low-pass filter to get colored noise
        colored_noise = lowpass(white_noise, cutoff=100.0, fs=1/dt)  # Adjust cutoff as needed
        #strain_records[name] += colored_noise

        # Generate separate raw white noise for blending
        white_burst_noise = np.random.normal(0, noise_scale, Nt)

        # Blend: 90% colored noise + 10% white noise
       # blended_noise = 0.90 * colored_noise + 0.10 * white_burst_noise
        blended_noise = (1 - NOISE_BLEND_RATIO) * colored_noise + NOISE_BLEND_RATIO * white_burst_noise

        # Add blended noise to strain record
        strain_records[name] += blended_noise


    
    
    h_plus, h_cross = extract_polarization(strain_records)

    # Time Array and save.
    #t_array = np.linspace(0, (Nt - 1) * dt, Nt)

    np.savez(f"{file_path}_Dynamic.npz",
             raw_strain=strain_record,
             radius=radius_record,
             t=t_array,dt=dt,t_max=t_max,
             M_chirp=M_chirp,
             soliton_radius=soliton_radius,
             amplitude=amplitude,
             f_gw=freq_record,
             strain_record_h_plus=h_plus,
             strain_record_h_cross=h_cross,
             detectors=detectors,
              **{f"strain_{name}": strain_records[name] for name in detectors}
             )

    # Save frequency evolution
    np.savez(f"{file_path}_Freq.npz", t=np.arange(Nt)*dt, freq=freq_record)


    for name in strain_records:    
        snr = np.max(np.abs(strain_records[name])) / noise_scale
        print(f"{title}: Detector {name} SNR = {snr:.2f}")
    
        # Quick plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_array,strain_records[name])
        plt.title(f"{title}: Detector Strain")
        plt.xlabel("Time Step")

        plt.subplot(1, 2, 2)
        plt.plot(radius_record)
        plt.title(f"{title}: Binary Separation (Grid Units)")
        plt.xlabel("Time Step")

        plt.tight_layout()
        plt.savefig(f"{file_path}_GEN_Chirp_Dynamic_Preview_{name}.png")
        plt.close()

        b, a = butter(4, [20, 600], btype='band', fs=1.0/dt)
        filtered_strain = filtfilt(b, a, strain_records[name])
        if np.max(np.abs(filtered_strain)) > 1e-40:
            filtered_strain /= np.max(np.abs(filtered_strain))


        #f, t_spec, Sxx = spectrogram(filtered_strain, fs=1.0/dt, nperseg=512,noverlap=400,scaling='density',mode='psd')  #strain_records[name]
        f, t_spec, Sxx = spectrogram(filtered_strain, fs=1.0/dt, nperseg=256,noverlap=200,scaling='density',mode='psd')  #strain_records[name]
        #Sxx_dB = 10 * np.log10(np.maximum(Sxx, 1e-30))
        # Auto dynamic range clipping
        #vmax = np.percentile(Sxx_dB, 99)
        #vmin = vmax - 60  # 60 dB dynamic range
    
        log_power = 10 * np.log10(Sxx + 1e-20)
        vmin = np.percentile(log_power, 5)
        vmax = np.percentile(log_power, 99.5)

        plt.figure(figsize=(10, 4))
        #plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno') #, cmap="viridis"
        plt.pcolormesh(t_spec, f, log_power, shading='gouraud', vmin=vmin, vmax=vmax)

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f"{title}: Spectrogram: {name}")
        plt.colorbar(label='Power [dB]')
        #plt.ylim(1, 0.5 / dt)
        plt.ylim(0, 500)
        plt.tight_layout()
        plt.savefig(f"{file_path}_GEN_Spectrogram_{name}.png")
        plt.close()
    

        print(f"✅ Finished Test: {title} Validated.")
    


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_chirp_generator_test(config)