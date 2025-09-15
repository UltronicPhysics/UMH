"""
UMH_CHSH_Entanglement.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH CHSH Entanglement Simulation.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless first

import math, sys, json, csv, random, signal, secrets
import numpy as np; np.seterr(all="warn")

import numba
from numba import njit, prange

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

#num_threads = int(os.environ.get("NUMBA_NUM_THREADS", max(1, os.cpu_count() // max(1, mp.cpu_count()))))
#numba.set_num_threads(num_threads)

#os.environ["NUMBA_NUM_THREADS"] = "4"   # or 6–8 if you have many cores
#num_threads = max(1, int(os.environ.get("NUMBA_NUM_THREADS", int(int(os.cpu_count() or 1)))/5))
#numba.set_num_threads(max(1, num_threads))
#print(f'numba.set_num_threads:{num_threads}')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from functools import partial
from scipy.stats import binomtest, chi2_contingency, norm

from math import sqrt

# --- Ctrl C Exit function ---
# Global flag
interrupted = False


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "SIZE": 32, #768,
        "STEPS": 6500, #400
        "RUNS": 50, #20
        
        "LATTICE_SPACING_DX":1.0e-35, #dx, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY_RHO":1.0e11, #dx, # Medium Density (normalized or in units)
        "MEDIUM_TENSION_TU":1.0e-5, #rho_u, # Medium Tension (normalized or in units)

        "DT": 1e-45,

        "noise_strength": 0.1,
        "collapse_frames": [15, 30],

        "CLIP_MAX":1e2,

        "LIMIT_OUTPUT": False,
        "MAX_S_ALLOWED": 2.828,
        
        "RNG_MODE": "crypto",          # "crypto" or "seeded"
        "MASTER_SEED": 20250601,       # used only if RNG_MODE=="seeded"
        "INDEPENDENT_SETTINGS": True, # False => relaxed measurement independence (UMH baseline)

        "MEASUREMENT_MODEL": "field", #"field" or "quantum"

        "USE_BALANCED": True,

        "ENERGY_RETURN": "relative",  # "density" | "total" | "relative"
        "NONLINEAR_COEFF": 1e25,

        "MEAS_REGION_MODE": "snapshots", # "jitter" | "snapshots"
        "MEAS_REGION_JITTER": 1,       # cells of random offset (0 = off)
        "SNAPSHOT_PERCENT": 0.02,
        "MAX_SNAPSHOTS":   128,


        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }

# Define a SIGINT handler
def handle_sigint(signum, frame):
    global interrupted
    interrupted = True
    if os.getpid() == os.getppid():  # Only the parent process should log this
        print("\n[INFO] SIGINT received (Ctrl+C). Cleaning up...")

def _recommend_numba_threads(size: int) -> int:
    # Aim for ≥ ~200k lattice sites per thread; cap at CPU count
    n_sites = max(0, size - 2)**3
    per_thread = 200_000
    rec = max(1, n_sites // per_thread)
    return min(rec, os.cpu_count() or 1)

def init_threads(config):
    size = int(config.get("SIZE", 32))
    # Allow override via config or env
    cfg_threads = config.get("NUMBA_THREADS", None)
    env_threads = os.environ.get("NUMBA_NUM_THREADS", None)
    if cfg_threads is not None: threads = int(cfg_threads)
    elif env_threads is not None: threads = int(env_threads)
    else: threads = _recommend_numba_threads(size)
    threads = max(1, min(threads, os.cpu_count() or 1))
    os.environ["NUMBA_NUM_THREADS"] = str(threads)  # for transparency
    numba.set_num_threads(threads)
    print(f"[THREADS] Numba={threads}, BLAS=1 (OMP/OPENBLAS/MKL capped)")

# --- pick processes/threads based on your config and machine ---
def _recommend_threads_per_proc(config):
    # If you already set NUMBA_THREADS in your config, honor it
    tpp = int(config.get("NUMBA_THREADS", 0))
    if tpp > 0: return max(1, tpp)
    # Heuristic: more grid → more threads per proc
    size = int(config.get("SIZE", 32))
    n_sites = max(0, size - 2)**3
    # aim for ~2e5 sites per thread; cap at 8 to avoid oversubscription
    return max(1, min(8, n_sites // 200_000)) or 1

def plan_pool(config, runs):
    total_cores = os.cpu_count() or 1
    tpp = _recommend_threads_per_proc(config)     # NUMBA threads per process
    procs = min(runs, max(1, total_cores // tpp))
    # Nice chunksize so each worker gets a handful of jobs at a time
    chunksize = max(1, runs // (procs * 4)) if runs > procs else 1
    return procs, tpp, chunksize


# --- initializer so each worker sets its own thread env ---
def _init_worker(numba_threads: int):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    # choose a stable threading layer; optional
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    numba.set_num_threads(numba_threads)

def scale_steps_for_target_sd(steps_now, sd_now, sd_target):
    scale = (sd_now / sd_target)**2
    return int(math.ceil(steps_now * scale))

def _compute_cfl_safety(c, dt, dx, dims=3):
    # 3D wave equation CFL ~= 1/sqrt(3)
    return (c * dt / dx) <= (1.0 / np.sqrt(dims) + 1e-12)

def validate_and_finalize_config(config):
    config["STEPS"] = scale_steps_for_target_sd(config["STEPS"], 0.0913, 0.05)
    print(f"[INFO] STEPS -> {config['STEPS']} for target sd≈0.05")

    config["SNAPSHOT_EVERY"]=max(1, int(config.get("SNAPSHOT_PERCENT") * int(config["STEPS"])))  # ~every 2% of STEPS;
    #print(f'SNAPSHOT_EVERY:{config["SNAPSHOT_EVERY"]}, SNAPSHOT_PERCENT:{config["SNAPSHOT_PERCENT"]}, steps:{config["STEPS"]}')

    Tu   = float(config["MEDIUM_TENSION_TU"])
    rho  = float(config["MEDIUM_DENSITY_RHO"])
    config["c"] = np.sqrt(Tu / rho)  # UMH wave-speed law

    if not _compute_cfl_safety(config["c"], config["DT"], config["LATTICE_SPACING_DX"]):
        r = config["c"]*config["DT"]/config["LATTICE_SPACING_DX"]
        raise ValueError(f"CFL violated: c*dt/dx={r:.3e} > 1/sqrt(3). "
                         "Reduce DT or increase LATTICE_SPACING_DX.")

    # CHSH angles (store what you actually use in measurements)
    config["angles"] = {"A":[0.0, 0.5*np.pi],"B":[0.25*np.pi, 0.75*np.pi],}

    if config["DTYPE"] not in (np.float32, np.float64):
        raise TypeError("DTYPE must be float32 or float64")
    return config

def _u01_crypto():
    # Fast uniform [0,1) using 53-bit precision (double-safe)
    return secrets.randbits(53) / float(1 << 53)

def _u01(rng):
    # Use NumPy RNG if given; otherwise OS crypto
    return float(rng.random()) if rng is not None else _u01_crypto()


def make_rngs(config, run_idx: int):
    """
    Returns (rng_hidden, rng_settings).
    - If RNG_MODE == 'seeded': reproducible, per-run RNGs derived from MASTER_SEED.
      * INDEPENDENT_SETTINGS=True  -> two independent RNGs (hidden/settings).
      * INDEPENDENT_SETTINGS=False -> same RNG object for both (max measurement dependence).
    - Else: non-deterministic (OS entropy).
    """
    mode  = str(config.get("RNG_MODE", "crypto")).lower()
    indep = bool(config.get("INDEPENDENT_SETTINGS", True))

    if mode == "seeded":
        master = int(config.get("MASTER_SEED", 0))

        # Derive a stable, per-run SeedSequence using spawn_key (doesn't depend on how many
        # children you spawned before; it depends only on master + run_idx).
        run_ss = np.random.SeedSequence(master, spawn_key=(int(run_idx),))

        if indep:
            # Two independent child sequences for hidden vs settings
            ss_hidden, ss_settings = run_ss.spawn(2)
            rng_hidden   = np.random.default_rng(ss_hidden)
            rng_settings = np.random.default_rng(ss_settings)
        else:
            # Maximal measurement dependence: same generator drives both
            rng_hidden = np.random.default_rng(run_ss)
            rng_settings = rng_hidden
    else:
        # Non-deterministic generators from OS entropy
        rng_hidden = np.random.default_rng()
        rng_settings = rng_hidden if not indep else np.random.default_rng()

    return rng_hidden, rng_settings



# --- Numba-accelerated functions ---
@njit(cache=True, fastmath=True, nogil=True, parallel=False)
def apply_noise_cube(psi, i0, j0, k0, radius, noise_cube, scale):
    side = 2 * radius + 1
    nx = ny = nz = side
    s = psi.shape[0]
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = i0 - radius + ix
                j = j0 - radius + iy
                k = k0 - radius + iz
                if 0 <= i < s and 0 <= j < s and 0 <= k < s:
                    psi[i, j, k] += scale * noise_cube[idx]
                idx += 1

@njit(cache=True, fastmath=True, nogil=True)
def init_fields_numba(size, dtype=np.float64):
    return (np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype),
        np.zeros((size, size, size), dtype=dtype))

@njit(cache=True, fastmath=False, nogil=True, parallel=False) #parallel=True
def initialize_soliton_numba(psi, size, x0, y0, z0, width=3.0, amplitude=1.0, clip_sigma=3.5):
    """
    Add a 3D Gaussian soliton to psi centered at (x0,y0,z0).

    psi        : 3D array (size x size x size), contiguous
    size       : cube side length
    x0,y0,z0   : integer voxel coordinates of the center
    width      : Gaussian sigma (in voxels)
    amplitude  : scaling factor
    clip_sigma : compute only within ±clip_sigma*width (speed-up). Set <=0 to scan full domain.
    """
    # guard against degenerate width
    w = float(width)
    if w <= 0.0: return
    inv2w2 = 1.0 / (2.0 * w * w)

    # bounding box (exclusive upper bounds)
    if clip_sigma is not None and clip_sigma > 0.0:
        R = int(math.ceil(clip_sigma * w))
        x_min = max(0, x0 - R); x_max = min(size, x0 + R + 1)
        y_min = max(0, y0 - R); y_max = min(size, y0 + R + 1)
        z_min = max(0, z0 - R); z_max = min(size, z0 + R + 1)
    else:
        x_min = y_min = z_min = 0
        x_max = y_max = z_max = size

    nx = x_max - x_min; ny = y_max - y_min; nz = z_max - z_min
    n  = nx * ny * nz

    for idx in prange(n):
        ix = idx // (ny * nz)
        rem = idx - ix * (ny * nz)
        iy = rem // nz
        iz = rem - iy * nz

        i = x_min + ix; j = y_min + iy; k = z_min + iz

        di = i - x0; dj = j - y0; dk = k - z0
        r2 = di*di + dj*dj + dk*dk

        psi[i, j, k] += amplitude * math.exp(-r2 * inv2w2)

@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def apply_phase_locking_numba(psi, size):
    for i in prange(size):
        for j in range(size):
            for k in range(size):
                psi[i, j, k] *= np.cos(psi[i, j, k] * 1e15)

@njit(cache=True, fastmath=True, nogil=True)
def _idx3_to_coords(idx, nx, ny, nz):
    ix = idx // (ny * nz)
    rem = idx - ix * (ny * nz)
    iy = rem // nz
    iz = rem - iy * nz
    return ix, iy, iz

@njit(cache=True, fastmath=False, nogil=True)
def apply_neumann_bc(psi):
    """Zero normal gradient (∂ψ/∂n = 0) on all faces."""
    nx, ny, nz = psi.shape
    psi[0,   :,  :] = psi[1,   :,  :]
    psi[nx-1,:,  :] = psi[nx-2,:,  :]
    psi[:,  0,  :] = psi[:,  1,  :]
    psi[:,  ny-1,:] = psi[:,  ny-2,:]
    psi[:,  :,  0] = psi[:,  :,  1]
    psi[:,  :,  nz-1] = psi[:,  :,  nz-2]

@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def update_lattice_3d_numba(psi, psi_prev, psi_next, size, c, dt, dx):
    inv_dx2 = 1.0 / (dx * dx)
    cdt2 = (c * c) * (dt * dt)
    lam = -1e25  # or pass nonlinear_coeff in
    for i in prange(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                v = psi[i, j, k]
                lap = (
                    psi[i+1, j, k] + psi[i-1, j, k] +
                    psi[i, j+1, k] + psi[i, j-1, k] +
                    psi[i, j, k+1] + psi[i, j, k-1] - 6.0 * v
                ) * inv_dx2
                psi_next[i, j, k] = (2.0 * v - psi_prev[i, j, k]) + cdt2 * (lap + lam * v * v * v)
    return psi_next

def update_lattice_3d_numba_old(psi, psi_prev, psi_next, size, c, dt, dx):
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

# --- Energy core: returns TOTAL energy (spatial integral) ---
@njit(parallel=True, cache=True, fastmath=False, nogil=True)
def compute_energy_integral(psi, psi_prev, dx, dt, nonlinear_coeff):
    nx, ny, nz = psi.shape
    voxel = dx * dx * dx

    # kinetic via vectorized sum (NumPy single-threaded due to BLAS caps)
    vel = (psi - psi_prev) / dt
    kinetic = 0.5 * (vel * vel).sum() * voxel

    inv2dx = 0.5 / dx

    pot = 0.0
    for i in prange(1, nx - 1):
        loc = 0.0
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                gx = (psi[i+1, j, k] - psi[i-1, j, k]) * inv2dx
                gy = (psi[i, j+1, k] - psi[i, j-1, k]) * inv2dx
                gz = (psi[i, j, k+1] - psi[i, j, k-1]) * inv2dx
                loc += 0.5 * (gx*gx + gy*gy + gz*gz)
        pot += loc
    pot *= voxel

    nonlin = 0.0
    for i in prange(nx):
        loc = 0.0
        for j in range(ny):
            for k in range(nz):
                v = psi[i, j, k]
                loc += v*v*v*v
        nonlin += loc
    nonlin *= 0.25 * nonlinear_coeff * voxel

    return kinetic + pot + nonlin
# --- End Numba-accelerated functions ---


def compute_energy(psi, psi_prev, dx, dt, energy_mode="density", nonlinear_coeff=1e25, total_volume=None, E0=None):
    """
    energy_mode: "density" | "total" | "relative"
      density  -> E_total / Volume
      total    -> E_total
      relative -> E_total / E0  (E0 captured from first call in the loop)
    """
    E_total = compute_energy_integral(psi, psi_prev, dx, dt, nonlinear_coeff)

    if total_volume is None:
        nx, ny, nz = psi.shape
        total_volume = (nx * ny * nz) * (dx**3)

    if energy_mode == "density": return E_total / total_volume
    elif energy_mode == "relative":
        base = E0 if (E0 is not None and E0 != 0.0) else E_total
        return E_total / base
    else: return E_total


def make_balanced_settings(T, rng=None):
    """
    Return arrays a_settings, b_settings with exactly T/4 trials for each pair
    (0,0), (0,1), (1,0), (1,1). If T is not a multiple of 4, leftovers are
    filled randomly.
    """
    if rng is None:rng = np.random.default_rng()

    base = (T // 4)
    a = np.array([0]*base + [0]*base + [1]*base + [1]*base, dtype=np.int8)
    b = np.array([0]*base + [1]*base + [0]*base + [1]*base, dtype=np.int8)

    # handle leftovers
    rem = T - 4*base
    if rem > 0:
        a = np.concatenate([a, rng.integers(0, 2, size=rem, dtype=np.int8)])
        b = np.concatenate([b, rng.integers(0, 2, size=rem, dtype=np.int8)])

    # randomize order (keeps independence from the hidden field)
    idx = rng.permutation(len(a))
    return a[idx], b[idx]


# --- Entangled soliton initialization with randomized phase offsets ---
def initialize_random_phase_entangled_solitons(psi, size, phase_map, rng):
    pairs = [(-5, 5), (-3, 3)]
    phase_options = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    for pair in pairs:
        x1 = size // 2 + pair[0]
        x2 = size // 2 + pair[1]
        initialize_soliton_numba(psi, size, x1, size // 2, size // 2)
        initialize_soliton_numba(psi, size, x2, size // 2, size // 2)
        phase_offset = float(phase_options[rng.integers(0, 4)])
        #phase_map[x1, size // 2, size // 2] = 0.0
        #phase_map[x2, size // 2, size // 2] = phase_offset
        psi[x2, size // 2, size // 2] *= np.cos(phase_offset)

def simulate_measurement_old(a_setting, b_setting, phase_map, angles):
    size = phase_map.shape[0]
    center = size // 2
    theta_options = angles["A"]
    phi_options   = angles["B"]
    theta = theta_options[a_setting]
    phi   = phi_options[b_setting]
    region = phase_map[center-1:center+2, center-1:center+2, center-1:center+2]
    proj_a = np.cos(region - theta)
    proj_b = np.cos(region - phi)
    spin_a = 1.0 if np.mean(proj_a) >= 0 else -1.0
    spin_b = 1.0 if np.mean(proj_b) >= 0 else -1.0
    return spin_a, spin_b

def simulate_measurement(a_setting, b_setting, phase_map, angles, rng=None, jitter=0):
    size = phase_map.shape[0]
    center = size // 2

    # pick a local region
    if jitter > 0 and rng is not None:
        di = int(rng.integers(-jitter, jitter + 1))
        dj = int(rng.integers(-jitter, jitter + 1))
        dk = int(rng.integers(-jitter, jitter + 1))
    else: di = dj = dk = 0

    # keep the 3x3x3 slice valid (need indices 1..size-2)
    i = min(max(center + di, 1), size - 2)
    j = min(max(center + dj, 1), size - 2)
    k = min(max(center + dk, 1), size - 2)

    region = phase_map[i-1:i+2, j-1:j+2, k-1:k+2]

    theta = angles["A"][a_setting]
    phi   = angles["B"][b_setting]
    proj_a = np.cos(region - theta)
    proj_b = np.cos(region - phi)
    spin_a = 1.0 if np.mean(proj_a) >= 0 else -1.0
    spin_b = 1.0 if np.mean(proj_b) >= 0 else -1.0
    return spin_a, spin_b


def simulate_measurement_quantum(a_setting, b_setting, rng, angles):
    """
    Quantum sanity-check sampler:
      E[A B | a,b] = -cos(theta_a - phi_b)
      P(A=B)   = sin^2((theta_a - phi_b)/2)
      P(A!=B)  = cos^2((theta_a - phi_b)/2)
    Marginals for A,B are unbiased (±1 with prob 1/2).
    """
    theta = angles["A"][a_setting]
    phi   = angles["B"][b_setting]
    d     = theta - phi
    p_same = np.sin(0.5 * d)**2

    # Draw A uniformly at random
    a = 1.0 if _u01(rng) < 0.5 else -1.0
    # Match or flip based on p_same
    if _u01(rng) < p_same: b = a
    else: b = -a
    return a, b

def simulate_measurement_on_region(a_setting, b_setting, region, angles):
    theta = angles["A"][a_setting]
    phi   = angles["B"][b_setting]
    proj_a = np.cos(region - theta)
    proj_b = np.cos(region - phi)
    spin_a = 1.0 if np.mean(proj_a) >= 0 else -1.0
    spin_b = 1.0 if np.mean(proj_b) >= 0 else -1.0
    return spin_a, spin_b

# --- Single CHSH simulation run ---
def run_single_chsh_job(run_idx, config):
    rng_hidden, rng_settings = make_rngs(config, run_idx)
    size   = config["SIZE"]
    steps  = config["STEPS"]
    c      = config["c"]
    dt     = config["DT"]
    dx     = config["LATTICE_SPACING_DX"]
    CLIP   = config["CLIP_MAX"]
    noise_strength = config["noise_strength"]
    collapse_frames = set(config["collapse_frames"])
    USE_BALANCED = config.get("USE_BALANCED", False)
    dtype  = config["DTYPE"]

    psi, psi_prev, psi_next, phase_map = init_fields_numba(size, dtype=dtype)
    initialize_random_phase_entangled_solitons(psi, size, phase_map, rng=rng_hidden)

    energy_trace = []
    center = (size//2, size//2, size//2)
    noise_cache = {}
    snapshots = []  # list of 3x3x3 phase regions

    for f in range(steps):
        apply_phase_locking_numba(psi, size)

        if f in collapse_frames:
            if f not in noise_cache:
                # reproducible if rng_hidden is seeded; independent if you choose
                R = 3
                side = 2*R + 1
                n = side*side*side
                noise_cache[f] = (rng_hidden.random(n) - 0.5).astype(np.float64)

            apply_noise_cube(psi, center[0], center[1], center[2], 3, noise_cache[f], noise_strength)

        update_lattice_3d_numba(psi, psi_prev, psi_next, size, c, dt, dx)
        apply_neumann_bc(psi_next)

        np.clip(psi_next, -CLIP, CLIP, out=psi_next)

        if config.get("MEAS_REGION_MODE", "jitter") == "snapshots":
            # take a local phase snapshot without building the whole phase_map
            i, j, k = center
            # small slice of psi_next -> phase region
            block = psi_next[i-1:i+2, j-1:j+2, k-1:k+2]
            region_phase = np.angle(np.exp(1j * block)).copy()
            if (f % int(config.get("SNAPSHOT_EVERY", 50))) == 0:
                snapshots.append(region_phase)
                if len(snapshots) > int(config.get("MAX_SNAPSHOTS", 128)): snapshots.pop(0)

        # energy bookkeeping
        if f == 0: energy0 = None  # reset at the start of each run
        if energy0 is None:
            # first sample → get total to set E0 if using "relative"
            E_total_init = compute_energy_integral(psi_next, psi, dx, dt, config["NONLINEAR_COEFF"])
            energy0 = E_total_init
        energy_value = compute_energy(psi_next, psi, dx, dt,
            energy_mode=config["ENERGY_RETURN"],
            nonlinear_coeff=config["NONLINEAR_COEFF"],
            total_volume=(size*size*size)*(dx**3),
            E0=energy0)

        energy_trace.append(energy_value)

        psi_prev, psi, psi_next = psi, psi_next, psi_prev

        if interrupted: break

    # stable phase map
    phase_map = np.angle(np.exp(1j * psi))
    
    indep = bool(config.get("INDEPENDENT_SETTINGS", True))

    # CHSH trials (settings RNG selection)
    measurements = []
    
    if(USE_BALANCED): a_seq, b_seq = make_balanced_settings(steps, rng_settings if rng_settings is not None else np.random.default_rng())
    for trial in range(steps):
        if(USE_BALANCED):
            a_setting = int(a_seq[trial])
            b_setting = int(b_seq[trial])
        else:
            if rng_settings is None:           # crypto + independent
                a_setting = secrets.randbelow(2)
                b_setting = secrets.randbelow(2)
            else:
                a_setting = int(rng_settings.integers(0, 2))
                b_setting = int(rng_settings.integers(0, 2))

        # --- CHOOSE MEASUREMENT MODEL ---
        if config.get("MEASUREMENT_MODEL", "field") == "quantum": # use quantum sanity-check sampler
            a_result, b_result = simulate_measurement_quantum(a_setting, b_setting, rng_settings, config["angles"])
        else:
            if config.get("MEAS_REGION_MODE", "jitter") == "snapshots" and snapshots:
                # choose which snapshot to use (independent of settings when indep=True)
                idx = int(rng_hidden.integers(0, len(snapshots)))
                region = snapshots[idx]
                a_result, b_result = simulate_measurement_on_region(a_setting, b_setting, region, config["angles"])
            else:
                # fallback to jitter mode (or no snapshots collected)
                a_result, b_result = simulate_measurement(a_setting, b_setting, phase_map, config["angles"],
                    rng=rng_hidden, jitter=int(config.get("MEAS_REGION_JITTER", 0)))

        # --------------------------------
        measurements.append({
            "A_Setting": a_setting,
            "B_Setting": b_setting,
            "A_Result": a_result,
            "B_Result": b_result,
        })

    m = pd.DataFrame(measurements)
    if run_idx < 3: print(m.value_counts(["A_Setting","B_Setting"]))

    def E(a, b):
        sel = (m["A_Setting"]==a) & (m["B_Setting"]==b)
        if sel.sum() == 0: return 0.0, 0.0
        prod = m.loc[sel, "A_Result"] * m.loc[sel, "B_Result"]
        e = float(prod.mean())
        # Wilson-se style SEM for a mean in [-1,1] (conservative)
        n = len(prod)
        var = max(1e-12, 1.0 - e**2) / max(1, n)   # bounded variable variance
        return e, np.sqrt(var)

    Eab,  se_ab  = E(0,0)
    Eabp, se_abp = E(0,1)
    Eapb, se_apb = E(1,0)
    Eapbp,se_apbp= E(1,1)
    S = abs(Eab - Eabp + Eapb + Eapbp)
    S_sem = float(np.sqrt(se_ab**2 + se_abp**2 + se_apb**2 + se_apbp**2))

    if run_idx == 0:
        print(f"[DEBUG] E00={Eab:.3f}  E01={Eabp:.3f}  E10={Eapb:.3f}  E11={Eapbp:.3f}  |  S={S:.3f}")

    return {"E(A,B)": Eab, "E(A,B')": Eabp, "E(A',B)": Eapb, "E(A',B')": Eapbp,
        "CHSH_S": S, "CHSH_S_sem": S_sem,
        "Energy_Trace": energy_trace,
        "Measurements": measurements,
        "Run_Index": run_idx,}


def simulate_classical_chsh(runs=1000, trials=2000, rng=None):
    # local realistic strategy: pre-assign A(a), A(a'), B(b), B(b') ∈ {±1}
    if rng is None: rng = np.random.default_rng()
    s_vals = []
    for _ in range(runs):
        A = {0: rng.choice([-1,1]), 1: rng.choice([-1,1])}
        B = {0: rng.choice([-1,1]), 1: rng.choice([-1,1])}
        data = []
        for _ in range(trials):
            a = rng.integers(0,2)
            b = rng.integers(0,2)
            data.append((a,b, A[a], B[b]))
        df = pd.DataFrame(data, columns=["A","B","Ar","Br"])
        def E(a,b):
            m = (df["A"]==a) & (df["B"]==b)
            return float((df.loc[m, "Ar"]*df.loc[m,"Br"]).mean())
        S = abs(E(0,0) - E(0,1) + E(1,0) + E(1,1))
        s_vals.append(S)
    return s_vals


def check_no_signaling(df):
    print("[INFO] Alice's marginal outcome by Bob's setting:")
    print(df.groupby(["A_Setting","B_Setting"])["A_Result"].mean().unstack())
    print("[INFO] Bob's marginal outcome by Alice's setting:")
    print(df.groupby(["B_Setting","A_Setting"])["B_Result"].mean().unstack())

    out={}
    tabA = pd.crosstab(df["B_Setting"], df["A_Result"]>0)
    tabB = pd.crosstab(df["A_Setting"], df["B_Result"]>0)
    if tabA.shape==(2,2): out["A|B"] = chi2_contingency(tabA)[1]
    if tabB.shape==(2,2): out["B|A"] = chi2_contingency(tabB)[1]
    print(f"[STATS] No-signalling chi2 p-values: {out}")
    return out


def run(config_overrides=None):
    # --- config ---
    config = get_default_config()
    if config_overrides: config.update(config_overrides)
    config = validate_and_finalize_config(config)

    RUNS           = int(config["RUNS"])
    DPI            = int(config["DPI"])
    LIMIT_OUTPUT   = bool(config["LIMIT_OUTPUT"])
    MEASUREMENT_MODEL = config["MEASUREMENT_MODEL"]
    MAX_S_ALLOWED  = float(config["MAX_S_ALLOWED"])
    CLASSICAL_BOUND = 2.0
    TSIRELSON_BOUND = float(2.0*np.sqrt(2.0))
    K = 4.0  # 4σ above Tsirelson = very conservative

    # paths
    outroot = config["OUTPUT_FOLDER"]
    title   = "UMH Quantum Entanglement"
    file_hdr= "UMH_Quantum_Entanglement"

    print(f"✅ Starting Test: {title}: Validation.")
    os.makedirs(outroot, exist_ok=True)
    outdir = os.path.join(outroot, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path = os.path.join(outdir, file_hdr)
    print(f"{title}: Files Will be Saved to {outdir}.")

    # signals
    signal.signal(signal.SIGINT, handle_sigint)

    #init_threads(config)

    # --- UMH physics: wave speed + CFL guard ---
    c = float(config["c"])
    print(f"[INFO] Measurement model: {config['MEASUREMENT_MODEL']}")
    # --- run workers ---
    results = []
    job = partial(run_single_chsh_job, config=config)


    procs, tpp, chunksize = plan_pool(config, RUNS)
    _init_worker(tpp)
    ctx = mp.get_context("spawn")  # safer with NumPy/Numba than 'fork' on some systems
    print(f"[POOL] processes={procs}, NUMBA_THREADS/proc={tpp}, chunksize={chunksize}")

    try:
        with ctx.Pool(processes=min(RUNS, procs)) as pool:
            for res in pool.imap_unordered(job, range(RUNS)):
                if interrupted:
                    print(f"[INFO] {title}: Interrupted — stopping early.")
                    pool.terminate()
                    break
                results.append(res)
            # normal exit: pool closes/join automatically via context manager
            if not interrupted:
                pool.close()  # no more tasks; let workers exit after finishing current ones
                pool.join()   # wait for them here
    except Exception as e:
        print(f"[ERROR] {title}: Unexpected exception: {e}")

    if interrupted or not results:
        print("[INFO] No results to process.")
        return

    df_all = pd.DataFrame(results)

    print(f"Preparing Classical Check.")

    # classification
    def classify_run(s):
        if s > MAX_S_ALLOWED: return "UMH"
        elif s > TSIRELSON_BOUND: return "Quantum>Tsirelson"
        elif s > CLASSICAL_BOUND: return "Quantum"
        else: return "Classical"

    target_bound = TSIRELSON_BOUND if MEASUREMENT_MODEL == "quantum" else CLASSICAL_BOUND
    target_name  = "Tsirelson"       if MEASUREMENT_MODEL == "quantum" else "Classical (S=2)"

    df_all["CHSH_Class"] = df_all["CHSH_S"].apply(classify_run)
    df_all["UMH_Tagged"] = df_all["CHSH_S"] > (TSIRELSON_BOUND + K * df_all.get("CHSH_S_sem", 0.0))

    df_all["Run"] = np.arange(len(df_all), dtype=int)

    classical_s_vals = simulate_classical_chsh(runs=RUNS)

    print(f"Preparing UMH Graphs.")

    # choose the view to plot/save
    df_view = df_all.copy()
    if LIMIT_OUTPUT:
        n_excl = int((df_view["CHSH_S"] > MAX_S_ALLOWED).sum())
        if n_excl: print(f"[INFO] Filter applied: Excluded {n_excl} run(s) with CHSH_S > {MAX_S_ALLOWED}")
        df_view = df_view[df_view["CHSH_S"] <= MAX_S_ALLOWED].copy()

    # --- Animation (use df_view consistently) ---
    xdata = df_view["Run"].to_list()
    ydata = df_view["CHSH_S"].to_list()

    if len(xdata) > 0:
        # contiguous x for animation
        idx = np.arange(len(df_view), dtype=int)
        ydata = df_view["CHSH_S"].to_numpy()

        fig, ax = plt.subplots(dpi=DPI)
        (ln,) = ax.plot([], [], "bo-")
        ax.set_xlim(0, max(1, len(idx)-1))
        ax.set_ylim(0.0, max(3.0, ydata.max() + 0.2))
        ax.axhline(2, ls="--", c="0.6")
        ax.set_xlabel("Run")
        ax.set_ylabel("CHSH S Value")
        ax.set_title(f"{title}: CHSH Simulation with Randomized Phases and Decoherence",
                fontsize=8, fontweight="bold") #fontweight="bold" pad=10, 

        def init():
            ln.set_data([], [])
            return (ln,)

        def update(frame):
            ln.set_data(idx[: frame + 1], ydata[: frame + 1])
            return (ln,)

        ani = FuncAnimation(fig, update, frames=len(idx), init_func=init, blit=False, repeat=False)

        ani.save(f"{file_path}_CHSH_Randomized.gif", writer=PillowWriter(fps=2))
        plt.close(fig)
    else: print("[WARN] No data to animate.")

    # save the plotted dataframe (view) and the full one
    df_view.to_csv(f"{file_path}_CHSH_Randomized_Data.csv", index=False)
    df_all.to_csv(f"{file_path}_CHSH_All_Data.csv", index=False)

    Svals = df_view["CHSH_S"].to_numpy()
    N = len(Svals)
    mean_S = float(Svals.mean())
    sd_S = float(Svals.std(ddof=1)) if N > 1 else 0.0
    sem_S = sd_S / math.sqrt(N) if N > 1 else 0.0
    print(f"N:{N}, sd_S:{sd_S}, sem_S:{sem_S}, TSIRELSON_BOUND:{TSIRELSON_BOUND}, mean_S:{mean_S}")
    
    # --- CHSH histogram (honest range) ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)

    if N == 0:
        ax.set_title(f"{title}: Distribution of CHSH S-values from UMH Simulation")
        ax.set_xlabel("CHSH S Value"); ax.set_ylabel("Frequency")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    else:
        # focus x-range around the data (NO clamp at 2.0)
        pad = 0.1; lo  = float(Svals.min() - pad); hi  = float(Svals.max() + pad)
        # keep a reasonable minimum width so bars are visible
        if hi - lo < 0.4: half = 0.2; lo -= half; hi += half

        # ~10 bins for ~50 runs
        bins = np.linspace(lo, hi, 11)

        # histogram of UMH/field results
        counts, bins, _ = ax.hist(Svals, bins=bins, color="skyblue", edgecolor="black",
            label="UMH results", zorder=3)
        y_max = float(counts.max()) if counts.size else 1.0

        # reference lines
        ax.axvline(CLASSICAL_BOUND, color="red", linestyle="--",
                   label="Classical bound (S=2)", zorder=4)
        ax.axvline(TSIRELSON_BOUND, color="green", linestyle="--",
                   label="Tsirelson (≈2.828)", zorder=4)

        # sample mean (show ±SEM in label text)
        ax.axvline(mean_S, color="black", linestyle=":", zorder=4,
                   label=f"Mean S = {mean_S:.3f} ± {sem_S:.3f}")

        # annotate extreme outliers if any
        hi_mask = Svals > 3.0
        if np.any(hi_mask):
            xs = Svals[hi_mask]
            ys = np.full_like(xs, y_max * 0.92, dtype=float)
            ax.scatter(xs, ys, marker="v", color="red", zorder=5, label="S > 3.0")

        # OPTIONAL: overlay a true classical baseline only if you computed it as data
        if 'classical_s_vals' in locals() and len(classical_s_vals):
            ax.hist(np.asarray(classical_s_vals, float), bins=bins, color="gray",
                    alpha=0.35, label="Classical baseline", zorder=1)

        # Tsirelson shading makes sense only in quantum-oracle validation
        if MEASUREMENT_MODEL == "quantum":
            ax.axvspan(TSIRELSON_BOUND, hi, color="tab:blue", alpha=0.07, label="> Tsirelson", zorder=0)

        #ax.set_xlim(lo, hi)
        ax.set_xlim(Svals.min()-0.05, Svals.max()+0.05)
        ax.set_title(f"{title}: Distribution of CHSH S-values from UMH Simulation")
        ax.set_xlabel("CHSH S Value"); ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{file_path}_CHSH_Histogram.png")
    plt.close(fig)


    # --- Energy traces (all runs) ---
    valid_traces = [r["Energy_Trace"] for r in results if isinstance(r, dict) and r.get("Energy_Trace") is not None]
    if valid_traces:
        lens = {len(t) for t in valid_traces}
        if len(lens) != 1: raise ValueError(f"{title}: Inconsistent energy trace lengths: {lens}")
        pd.DataFrame(valid_traces).T.to_csv(f"{file_path}_All_Energy_Traces.csv", index=False)
        energy_matrix = np.array(valid_traces)
        mean_energy = energy_matrix.mean(axis=0)
        std_energy  = energy_matrix.std(axis=0)
        plt.figure(figsize=(8,4), dpi=DPI)
        plt.plot(mean_energy, label="Mean Energy")
        plt.fill_between(range(len(mean_energy)), mean_energy-std_energy, mean_energy+std_energy,
                         alpha=0.4, label="±1 std dev")
        mode = config.get("ENERGY_RETURN", "density")
        y_label = {"density":  "Energy density (arb. units)",
            "total":    "Total energy (arb. units)",
            "relative": "Relative energy (E / E₀)"}[mode]
        plt.xlabel("Time Step")
        plt.ylabel(y_label)
        plt.title(f"{title}: Average Energy Trace Across All Runs")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{file_path}_Mean_Energy_Trace.png"); plt.close()

    # Save violations (from the full df)
    df_all[df_all["UMH_Tagged"]].to_csv(f"{file_path}_Violations_Only.csv", index=False)

    # --- Scatter (view) ---
    plt.figure(figsize=(6,4), dpi=DPI)
    plt.scatter(df_view["Run"], df_view["CHSH_S"], label="All Runs")
    plt.axhline(CLASSICAL_BOUND, linestyle="--", color="gray",  label="Classical Bound")
    plt.axhline(TSIRELSON_BOUND, linestyle="--", color="green", label="Tsirelson Bound")
    plt.ylim(0, max(3.0, df_view["CHSH_S"].max() + 0.5))
    plt.title(f"{title}: CHSH Simulation with Decoherence", fontsize=10, pad=10)
    plt.xlabel("Run"); plt.ylabel("CHSH S Value")
    # annotate UMH runs (full indexing preserved via Run column)
    df_umh = df_all[df_all["UMH_Tagged"]]

    hi = df_all[df_all["CHSH_S"] > 3.0]
    for _, r in hi.iterrows():
        plt.annotate(f"Run {int(r['Run'])}", (r["Run"], r["CHSH_S"]),
                     textcoords="offset points", xytext=(0,5), ha="center",
                     fontsize=8, color='red')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{file_path}_CHSH_Scatter_Tagged.png"); plt.close()

    total_runs = int(len(df_all))
    if(False):
        # --- Binomial test on *full* df ---
        p_null = 0.001; alpha = 0.05
        num_umh = int((df_all["CHSH_Class"] == "UMH").sum())
        if total_runs > 0 and num_umh > 0:
            p_value = binomtest(k=num_umh, n=total_runs, p=p_null, alternative='greater').pvalue
            result_lines = [
                "[STATS] Binomial Test for UMH-tagged Events",
                f"Total Runs: {total_runs}",
                f"UMH Violations: {num_umh}",
                f"Null Hypothesis (p_null): {p_null}",
                f"P-Value: {p_value:.3e}",
                f"Significance Level: α = {alpha}",
                "→ Result: " + ("Statistically significant — unlikely under quantum-only model."
                                if p_value < alpha else
                                "Not statistically significant — consistent with quantum-only model.")
            ]
        else: result_lines = ["[STATS] No UMH-tagged events found.", "Binomial test not applicable."]

        print("\n".join(result_lines))
        with open(f"{file_path}_Statistical_Results.txt", "w") as f:
            f.write("\n".join(result_lines))

    if total_runs > 0:
        # Guard against zero-variance case
        if sem_S < 1e-15:
            z = float("nan")
            p_two = float("nan")
            note = "Variance across runs is zero; hypothesis test undefined."
        else:
            z = (mean_S - target_bound) / sem_S
            p_two = 2.0 * (1.0 - norm.cdf(abs(z)))
            note = "→ Mean S significantly different from " + target_name if p_two < 0.05 else \
                   "→ Mean S consistent with " + target_name

        # 95% CI for the mean
        ci_lo = mean_S - 1.96*sem_S
        ci_hi = mean_S + 1.96*sem_S

        lines = [
            f"[STATS] Test of mean S vs {target_name}",
            f"Runs: {N}",
            f"Mean S: {mean_S:.4f}",
            f"Std dev across runs: {sd_S:.4f}",
            f"SEM (mean S): {sem_S:.4e}",
            f"95% CI (mean S): [{ci_lo:.4f}, {ci_hi:.4f}]",
            f"{target_name} bound: {target_bound:.4f}",
            f"z-score: {z}",
            f"Two-sided p-value: {p_two}",
        ]
        if not (sem_S < 1e-15):
            lines.append("→ Mean S significantly different from " + target_name if p_two < 0.05
                         else "→ Mean S consistent with " + target_name)
        else: lines.append("Variance across runs is zero; hypothesis test undefined.")

        print("\n".join(lines))
        with open(f"{file_path}_Statistical_Results.txt", "w") as f:
            f.write("\n".join(lines))

    # --- No-signalling & classical comparison ---
    all_measurements = [row for r in results if "Measurements" in r for row in r["Measurements"]]
    if all_measurements:
        measurements_df = pd.DataFrame(all_measurements)
        check_no_signaling(measurements_df)

    print("min S:", Svals.min(), "below-2 count:", (Svals < 2.0).sum())

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    # 1) classical reference: red line at S=2
    ax.axvline(CLASSICAL_BOUND, color="red", linestyle="--", linewidth=2, label="Classical bound (S=2)")
    # 2) thin grey band: 2.0 ± 2·SEM(field)
    band_lo, band_hi = CLASSICAL_BOUND - 2*sem_S, CLASSICAL_BOUND + 2*sem_S
    ax.axvspan(band_lo, band_hi, color="grey", alpha=0.20, label=f"Classical ±2·SEM (±{2*sem_S:.3f})")
    # 3) UMH/field histogram on top
    lo = min(Svals.min(), band_lo) - 0.01; hi = max(Svals.max(), band_hi) + 0.01
    bins = np.linspace(lo, hi, 11)  # ~10 bins is good for ~50 runs
    ax.hist(Svals, bins=bins, color="C0", alpha=0.6, label="UMH (field) results", edgecolor="none")
    # cosmetics
    ax.axvline(mean_S, ls=":", c="k", lw=1.5, label=f"Mean S = {mean_S:.3f}")
    ax.grid(axis='y', alpha=0.25)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("CHSH S")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title}: vs Classical CHSH S-values")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{file_path}_vs_Classical_CHSH.png"); plt.close(fig)


    # --- Summary JSON (full df) ---
    summary = {
        "SIZE": int(config["SIZE"]),
        "STEPS": int(config["STEPS"]),
        "RUNS": int(config["RUNS"]),
        "LATTICE_SPACING_DX": float(config["LATTICE_SPACING_DX"]),
        "MEDIUM_DENSITY_RHO": float(config["MEDIUM_DENSITY_RHO"]),
        "MEDIUM_TENSION_TU": float(config["MEDIUM_TENSION_TU"]),
        "DT": float(config["DT"]),
        "C": float(config["c"]),
        "LIMIT_OUTPUT": bool(config["LIMIT_OUTPUT"]),
        "RNG_MODE": config.get("RNG_MODE"),
        "INDEPENDENT_SETTINGS": bool(config.get("INDEPENDENT_SETTINGS", True)),
        "Max_S": float(df_all["CHSH_S"].max()),
        "Mean_S": float(df_all["CHSH_S"].mean()),
        "Std_S": float(df_all["CHSH_S"].std(ddof=1)),
        "USE_BALANCED": bool(config["USE_BALANCED"]),
        "MEASUREMENT_MODEL": str(config["MEASUREMENT_MODEL"]),
    }
    with open(f"{file_path}_Run_Summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Finished Test: {title}: Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)