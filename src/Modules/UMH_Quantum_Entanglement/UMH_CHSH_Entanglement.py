"""
UMH_CHSH_Entanglement.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH CSHS Entanglement Simulation.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import signal
import os
import sys
import json
import csv

from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit, prange
from scipy.stats import binomtest
from functools import partial


#try:
    #from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d
#except ImportError:
    #import sys, os
    #sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    #from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d

    


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "SIZE": 32, #768,
        "STEPS": 1000, #400
        "RUNS": 50, #20

        "LATTICE_SPACING":1.0e-35, #1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0e11, #1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0e-5, #1.0, # Medium Pressure (normalized or in units)

        "DT": 1e-45,

        "noise_strength": 0.1,
        "collapse_frames": [15, 30],

        "LIMIT_OUTPUT":False, #1.0, # Medium Pressure (normalized or in units)
        "MAX_S_ALLOWED":2.828, #1.0, # Medium Pressure (normalized or in units)
        "MAX_QUANTUM":2, #1.0, # Medium Pressure (normalized or in units)
        "CLIP_MAX":1e2, #1.0, # Medium Pressure (normalized or in units)


        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }





# --- Constants ---
#Tu = 1.0e11
#rho_u = 1.0e-5
#L = 1.0e-35
#c = np.sqrt(Tu / rho_u)
#dt = 1e-45
#dx = L

# --- Parameters ---
#N = 32
#steps = 1000
#runs = 20
#angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#noise_strength = 0.1
#collapse_frames = [15, 30]

# --- Configurable Filter Option ---
#LIMIT_OUTPUT = False           # Set to True to limit CHSH_S to <= MAX_S_ALLOWED
#MAX_S_ALLOWED = 2.828          # Tsirelson bound
#MAX_QUANTUM = 2
#CLIP_MAX = 1e2  # Max absolute value for psi field to prevent blow-up

# --- Ctrl C Exit function ---
# Global flag
interrupted = False

#dtype=np.float64



# Define a SIGINT handler
def handle_sigint(signum, frame):
    global interrupted
    interrupted = True
    if os.getpid() == os.getppid():  # Only the parent process should log this
        print("\n[INFO] SIGINT received (Ctrl+C). Cleaning up...")

# --- Numba-accelerated functions ---

@njit(cache=True, parallel=True, fastmath=True)
def init_fields_numba(size, dtype=np.float64):
    return (
        np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype)
    )

@njit
def initialize_soliton_numba(psi, size, x0, y0, z0, width=3, amplitude=1.0):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r2 = (i - x0)**2 + (j - y0)**2 + (k - z0)**2
                psi[i, j, k] += amplitude * np.exp(-r2 / (2 * width**2))

@njit(cache=True, parallel=True, fastmath=True)
def apply_phase_locking_numba(psi, size):
    for i in prange(size):
        for j in range(size):
            for k in range(size):
                psi[i, j, k] *= np.cos(psi[i, j, k] * 1e15)

@njit(cache=True, parallel=True, fastmath=True)
def trigger_local_collapse_numba(psi, size, radius=3, noise_strength=0.1):
    center=(size//2, size//2, size//2)
    i0, j0, k0 = center
    for i in prange(i0 - radius, i0 + radius):
        for j in range(j0 - radius, j0 + radius):
            for k in range(k0 - radius, k0 + radius):
                if 0 <= i < size and 0 <= j < size and 0 <= k < size:
                    psi[i, j, k] += noise_strength * (np.random.rand() - 0.5)

@njit(cache=True, parallel=True, fastmath=True)
def update_lattice_3d_numba(psi, psi_prev, psi_next, size, c, dt, dx):
    for i in prange(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                laplacian = (
                    psi[i+1, j, k] + psi[i-1, j, k] +
                    psi[i, j+1, k] + psi[i, j-1, k] +
                    psi[i, j, k+1] + psi[i, j, k-1] -
                    6.0 * psi[i, j, k]
                ) / dx**2
                nonlinear = -1e25 * psi[i, j, k]**3
                psi_next[i, j, k] = (
                    2.0 * psi[i, j, k] - psi_prev[i, j, k] +
                    (c**2 * dt**2) * (laplacian + nonlinear)
                )
    return psi_next



@njit(cache=True, parallel=True, fastmath=True)
def compute_energy(psi, psi_prev, dx, dt):
    velocity = (psi - psi_prev) / dt
    kinetic = 0.5 * np.sum(velocity**2)

    grad_x = np.zeros_like(psi)
    grad_y = np.zeros_like(psi)
    grad_z = np.zeros_like(psi)

    # Parallel over i (x-dimension)
    for i in prange(1, psi.shape[0] - 1):
        for j in range(1, psi.shape[1] - 1):
            for k in range(1, psi.shape[2] - 1):
                grad_x[i, j, k] = (psi[i + 1, j, k] - psi[i - 1, j, k]) / (2 * dx)
                grad_y[i, j, k] = (psi[i, j + 1, k] - psi[i, j - 1, k]) / (2 * dx)
                grad_z[i, j, k] = (psi[i, j, k + 1] - psi[i, j, k - 1]) / (2 * dx)

    potential = 0.5 * np.sum(grad_x**2 + grad_y**2 + grad_z**2)
    nonlinear = 0.25e25 * np.sum(psi**4)

    return kinetic + potential + nonlinear



# --- Entangled soliton initialization with randomized phase offsets ---

@njit
def initialize_random_phase_entangled_solitons(psi, size, phase_map, rng_seed=0):
    pairs = [(-5, 5), (-3, 3)]
    np.random.seed(rng_seed)
    phase_options = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    
    for pair in pairs:
        x1 = size // 2 + pair[0]
        x2 = size // 2 + pair[1]
        initialize_soliton_numba(psi, size, x1, size // 2, size // 2)
        initialize_soliton_numba(psi, size, x2, size // 2, size // 2)
        index = np.random.randint(0, 4)  # Integer between 0–3
        phase_offset = phase_options[index]
        phase_map[x1, size // 2, size // 2] = 0.0
        phase_map[x2, size // 2, size // 2] = phase_offset
        psi[x2, size // 2, size // 2] *= np.cos(phase_offset)



def simulate_measurement(a_setting, b_setting, phase_map):
    """
    Simulates a single measurement result for Alice and Bob,
    based on their chosen settings and the shared phase_map.
    """
    size = phase_map.shape[0]
    center = size // 2

    # Measurement angles (radians)
    theta_options = [0, np.pi / 4]  # A=0, A'=π/4
    phi_options = [np.pi / 8, 3 * np.pi / 8]  # B=π/8, B'=3π/8

    theta = theta_options[a_setting]
    phi = phi_options[b_setting]

    # Sample a small central region
    region = phase_map[center - 1:center + 2, center - 1:center + 2, center - 1:center + 2]

    # Projected phase-based "spin" directions
    proj_a = np.cos(region - theta)
    proj_b = np.cos(region - phi)

    spin_a = np.sign(np.mean(proj_a))
    spin_b = np.sign(np.mean(proj_b))

    return spin_a, spin_b





# --- Single CHSH simulation run ---
def run_single_chsh_job(seed, config):
    np.random.seed(seed)
    random.seed(seed)

    size=config["SIZE"]
    steps=config["STEPS"]

    c=config["c"]
    dt=config["DT"]
    dx=config["LATTICE_SPACING"]

    angles=config["angles"]

    CLIP_MAX=config["CLIP_MAX"]
    collapse_frames = config["collapse_frames"]
    noise_strength=config["noise_strength"]

    dtype=config["DTYPE"]

    psi, psi_prev, psi_next, phase_map = init_fields_numba(size, dtype=dtype)
    initialize_random_phase_entangled_solitons(psi, size, phase_map, rng_seed=seed)

    energy_trace = []
    measurements = []

    for f in range(steps):
        apply_phase_locking_numba(psi, size)
        if f in collapse_frames:
            trigger_local_collapse_numba(psi, size, noise_strength=noise_strength)
        update_lattice_3d_numba(psi, psi_prev, psi_next, size, c, dt, dx)
        # Prevent field blowup from nonlinear terms
        psi = np.clip(psi, -CLIP_MAX, CLIP_MAX)

        energy = compute_energy(psi, psi_prev, dx, dt)
        energy_trace.append(energy)

        psi_prev, psi, psi_next = psi, psi_next, psi_prev

        if interrupted:
            #print("[INFO] Interrupted — stopping early.")
            break

    phase_map = np.angle(np.exp(1j * psi))

    E_vals = []
    for _ in range(4):
        a, b = random.sample(angles, 2)
        region = phase_map[size//2-1:size//2+2, size//2-1:size//2+2, size//2-1:size//2+2]
        projection_a = np.cos(region - a)
        projection_b = np.cos(region - b)
        spin_a = np.sign(np.mean(projection_a))
        spin_b = np.sign(np.mean(projection_b))
        E_vals.append(spin_a * spin_b)

        if interrupted:
            #print("[INFO] Interrupted — stopping early.")
            break

    if interrupted:
        #print("[INFO] Interrupted — stopping early.")
        return

    for trial in range(steps):
        a_setting = np.random.choice([0, 1])
        b_setting = np.random.choice([0, 1])
        a_result, b_result = simulate_measurement(a_setting, b_setting, phase_map)
    
        measurements.append({
            "A_Setting": a_setting,
            "B_Setting": b_setting,
            "A_Result": a_result,
            "B_Result": b_result
        })


    S = abs(E_vals[0] - E_vals[1] + E_vals[2] + E_vals[3])
    return {
        "E(A,B)": E_vals[0],
        "E(A,B')": E_vals[1],
        "E(A',B)": E_vals[2],
        "E(A',B')": E_vals[3],
        "CHSH_S": S,
        "Energy_Trace": energy_trace,
        "Measurements": measurements
    }


def simulate_classical_chsh(runs=1000):
    classical_s_values = []
    for _ in range(runs):
        # Deterministic outputs: A and B depend only on local settings and shared hidden lambda
        lambd = np.random.choice([0, 1], size=1)[0]
        a = np.random.choice([0, 1])
        b = np.random.choice([0, 1])

        # Classical outcomes using predefined table (example: Bell’s classical model)
        A = (lambd ^ a)  # Example logic
        B = (lambd ^ b)  # Should keep within S ≤ 2

        # Estimate CHSH component (simplified)
        s = 2  # Classical max
        classical_s_values.append(s)
    return classical_s_values


def check_no_signaling(dataframe):
    # Group by setting combinations
    marginals = dataframe.groupby(["A_Setting", "B_Setting"])["A_Result"].mean().unstack()
    print("[INFO] Alice's marginal outcome by Bob's setting:")
    print(marginals)

    marginals_b = dataframe.groupby(["B_Setting", "A_Setting"])["B_Result"].mean().unstack()
    print("[INFO] Bob's marginal outcome by Alice's setting:")
    print(marginals_b)

    # You expect marginal means to be similar across the other party's setting


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["SIZE"] # Memory-safe grid size.
    steps = config["STEPS"]
    runs = config["RUNS"]

    dx=config["LATTICE_SPACING"]
    
    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]


    LIMIT_OUTPUT=config["LIMIT_OUTPUT"]
    MAX_S_ALLOWED=config["MAX_S_ALLOWED"]
    MAX_QUANTUM=config["MAX_QUANTUM"]
    CLIP_MAX=config["CLIP_MAX"]


    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Quantum Entanglement"
    file_hdr="UMH_Quantum_Entanglement"
  
    print(f"✅ Starting Test: {title}: Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    
    #noise_strength = 0.1
    #collapse_frames = [15, 30]


    # Register the handler
    signal.signal(signal.SIGINT, handle_sigint)

    c = np.sqrt(Tu / rho_u)
    config["c"]=c
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    config["angles"]=angles


    results = []

    job = partial(run_single_chsh_job, config=config)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        try:
            for res in pool.imap_unordered(job, range(runs)):
                if interrupted:
                    print(f"[INFO] {title}: Interrupted — stopping early.")
                    break
                results.append(res)
        except Exception as e:
            print(f"[ERROR] {title}: Unexpected exception: {e}")
        finally:
            pool.terminate()
            pool.join()

    if interrupted or not results:
        exit()

    df = pd.DataFrame(results)

    # Add a UMH flag for superquantum events

    def classify_run(s):
        if s > MAX_S_ALLOWED:
            return "UMH"
        elif s > MAX_QUANTUM:
            return "Quantum"
        else:
            return "Classical"

    df["CHSH_Class"] = df["CHSH_S"].apply(classify_run)
    df["UMH_Tagged"] = df["CHSH_S"] > MAX_S_ALLOWED

    df["Run"] = range(runs)

    # Animate Results
    xdata = df["Run"].tolist()
    ydata = df["CHSH_S"].tolist()

    # Optional CHSH limit for filtered output
    if LIMIT_OUTPUT:
        if (df["CHSH_S"] > MAX_S_ALLOWED).any():
            print(f"[INFO] Filter applied: Excluded {sum(df['CHSH_S'] > MAX_S_ALLOWED)} run(s) with CHSH_S > {MAX_S_ALLOWED}")

        df = df[df["CHSH_S"] <= MAX_S_ALLOWED].copy()

    fig, ax = plt.subplots()
    ln, = ax.plot([], [], 'bo-', animated=True)
    ax.set_xlim(0, max(xdata))
    ax.set_ylim(0, 3)
    ax.set_xlabel("Run")
    ax.set_ylabel("CHSH S Value")
    ax.set_title(f"{title}: CHSH Simulation with Randomized Phases and Decoherence")

    def init():
        ln.set_data([], [])
        return ln,

    def update(frame):
        ln.set_data(xdata[:frame+1], ydata[:frame+1])
        return ln,

    ani = FuncAnimation(fig, update, frames=len(xdata), init_func=init, blit=True, repeat=False)
    ani.save(f"{file_path}_CHSH_Randomized.gif", writer=PillowWriter(fps=1))
    df.to_csv(f"{file_path}_CHSH_Randomized_Data.csv", index=False)

    # --- Plot histogram of CHSH S-values ---
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(df["CHSH_S"], bins=np.linspace(0, 4, 17), color='skyblue', edgecolor='black')
    y_max = max(counts)
    plt.axvline(2.0, color='red', linestyle='--', label='Classical Bound (S=2)')
    plt.axvline(2.828, color='green', linestyle='--', label='Tsirelson Bound (S≈2.828)')
    plt.title(f"{title}: Distribution of CHSH S-values from UMH Simulation")
    plt.xlabel("CHSH S Value")
    plt.ylabel("Frequency")

    plt.axvline(2.0, color='red', linestyle='--', label='Classical Bound')
    plt.axvline(2.828, color='green', linestyle='--', label='Quantum Tsirelson Bound')
    plt.text(3.6, y_max, 'UMH region?', color='blue')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_CHSH_Histogram.png")
    plt.close()

    # Energy monitor for inspection.
    valid_traces = [
        r["Energy_Trace"] for r in results
        if isinstance(r, dict) and "Energy_Trace" in r and r["Energy_Trace"] is not None
    ]

    # Check that all energy traces are the same length
    trace_lengths = [len(trace) for trace in valid_traces]
    if len(set(trace_lengths)) != 1:
        raise ValueError(f"{title}: Inconsistent energy trace lengths: {set(trace_lengths)}")

    # Save all traces to CSV (rows = time, columns = runs)
    pd.DataFrame(valid_traces).T.to_csv(f"{file_path}_All_Energy_Traces.csv", index=False)

    # Convert to array and compute statistics
    energy_matrix = np.array(valid_traces)
    mean_energy = np.mean(energy_matrix, axis=0)
    std_energy = np.std(energy_matrix, axis=0)

    # --- Plot mean energy with std dev band ---
    plt.figure(figsize=(8, 4))
    plt.plot(mean_energy, label="Mean Energy", color="blue")
    plt.fill_between(range(len(mean_energy)),
                     mean_energy - std_energy,
                     mean_energy + std_energy,
                     color='lightblue', alpha=0.4, label="±1 std dev")
    plt.xlabel("Time Step")
    plt.ylabel("Total Field Energy")
    plt.title(f"{title}: Average Energy Trace Across All Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Mean_Energy_Trace.png")
    plt.close()


    # Plot all runs
    df[df["UMH_Tagged"]].to_csv(f"{file_path}_Violations_Only.csv", index=False)

    plt.figure(figsize=(6, 4))

    plt.scatter(df["Run"], df["CHSH_S"], color="blue", label="All Runs")

    # Highlight UMH-tagged runs
    df_umh = df[(df["UMH_Tagged"]) & (df["CHSH_Class"] == "UMH")]

    if not df_umh.empty:
        # Add this loop to annotate the highlighted UMH points
        for idx, row in df_umh.iterrows():
            plt.annotate(f"Run {row['Run']}",
                         (row['Run'], row['CHSH_S']),
                         textcoords="offset points",
                         xytext=(0, 5),  # Adjust vertical offset as needed
                         ha='center', fontsize=8, color='red')



    # Optional: Add reference lines
    plt.axhline(2.0, linestyle="--", color="gray", label="Classical Bound")
    plt.axhline(2.828, linestyle="--", color="green", label="Tsirelson Bound")
    plt.ylim(0, df["CHSH_S"].max() + 0.5)

    plt.title(f"{title}: CHSH Simulation with Decoherence")
    plt.xlabel("Run")
    plt.ylabel("CHSH S Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_CHSH_Scatter_Tagged.png")
    plt.close()


    # Parameters
    p_null = 0.001  # Assumed probability of a UMH violation under quantum-only theory
    alpha = 0.05    # Significance level

    # Count violations
    num_umh = (df["CHSH_Class"] == "UMH").sum()
    total_runs = len(df)

    # Perform binomial test
    if num_umh > 0:
        p_value = binomtest(k=num_umh, n=total_runs, p=p_null, alternative='greater').pvalue

        # Format output
        result_lines = [
            "[STATS] Binomial Test for UMH-tagged Events",
            f"Total Runs: {total_runs}",
            f"UMH Violations: {num_umh}",
            f"Null Hypothesis (p_null): {p_null}",
            f"P-Value: {p_value:.3e}",
            f"Significance Level: α = {alpha}",
        ]

        if p_value < alpha:
            result_lines.append("→ Result: Statistically significant — unlikely under quantum-only model.")
        else:
            result_lines.append("→ Result: Not statistically significant — consistent with quantum-only model.")
    else:
        result_lines = ["[STATS] No UMH-tagged events found.","Binomial test not applicable."]

    # Print to console
    print("\n".join(result_lines))

    # Save to file
    with open(f"{file_path}_Statistical_Results.txt", "w") as f:
        f.write("\n".join(result_lines))


    all_measurements = [row for r in results if "Measurements" in r for row in r["Measurements"]]    
    measurements_df = pd.DataFrame(all_measurements)
    check_no_signaling(measurements_df)

    # --- Classical Comparison ---
    classical_s_vals = simulate_classical_chsh(runs)
    plt.figure()
    plt.hist(classical_s_vals, bins=np.linspace(0, 4, 9), color="gray", alpha=0.6, label="Classical Model")
    plt.hist(df["CHSH_S"], bins=np.linspace(0, 4, 9), color="skyblue", alpha=0.6, label="UMH Results")
    plt.axvline(2.0, color='red', linestyle='--', label='Classical Bound')
    plt.axvline(2.828, color='green', linestyle='--', label='Quantum Bound')
    plt.title(f"{title}: vs Classical CHSH S-values")
    plt.xlabel("CHSH S")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_vs_classical_chsh.png")
    plt.close()


    
    print(f"✅ Finished Test: {title}: Validation.")

if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()