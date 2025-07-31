""" 
UMH Planck Constant Emergence Test

Author: Andrew Dodge
Date: July 2025

Description:
Tests whether energy in the Ultronic Medium scales with wave frequency as E = ℏω.
Generates wave packets at different frequencies, computes energy, performs FFT, and fits E(ω).
Outputs CSV of results and scatter plot with linear regression.

Parameters:
    - grid_size
    - dx
    - dt
    - frequencies to test

Output:
    - planck_energy_vs_freq.csv
    - planck_e_vs_w.png
    - hbar_estimate.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "SIZE": 8192,
        #"DT": 0.01,

        "LATTICE_SPACING":0.01, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.

        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)

        "DPI":300, #PNG Resolution.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def generate_wave_packet(N, k0, dx):
    x = np.linspace(0, N * dx, N, endpoint=False)
    wave = np.exp(-((x - N*dx/2)**2) / (2 * (0.1*N*dx)**2)) * np.sin(2 * np.pi * k0 * x)
    return wave.astype(np.float64)

def compute_energy(wave, dx):
    grad = np.gradient(wave, dx)
    kinetic = 0.5 * np.sum(grad**2) * dx
    potential = 0.5 * np.sum(wave**2) * dx
    return kinetic + potential

def estimate_dominant_frequency(wave, dt):
    fft_vals = np.abs(rfft(wave))
    freqs = rfftfreq(len(wave), dt)
    dominant = freqs[np.argmax(fft_vals)]
    return 2 * np.pi * dominant  # angular frequency ω

def calculate_cfl_dt(c, dx, dimensions=3, safety_factor=0.25):
    return dx / (c * (dimensions ** 0.5)) * safety_factor

def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["SIZE"] # Memory-safe grid size.
    #dt = config["DT"]
    dx=config["LATTICE_SPACING"]

    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    dpi=config["DPI"]

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Planck Emergence"
    file_hdr="UMH_Planck_Emergence"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")

    v = np.sqrt(Tu / rho_u)
    print(f"{title} Calculated Wave Speed Constant C = {v}")

    v = np.sqrt(Tu / rho_u)
    dt = calculate_cfl_dt(v, dx, safety_factor=0.25)
    print(f"{title} Calculated DT = {dt}")

    #size = 2048
    #dx = 0.01
    #dt = 0.01

    k_values = np.linspace(5, 30, 10)
    energies = []
    omegas = []

    for k0 in k_values:
        wave = generate_wave_packet(size, k0, dx)

        # ✅ Trim wave edges to remove boundary interference
        trim_frac = 0.1
        start = int(len(wave) * trim_frac)
        end = int(len(wave) * (1 - trim_frac))
        wave_trimmed = wave[start:end]

        # ✅ Compute energy and frequency from trimmed region
        energy = compute_energy(wave_trimmed, dx)
        omega = estimate_dominant_frequency(wave_trimmed, dt)

        # ✅ Optional: Normalize energy by effective window size
        normalized_energy = energy / len(wave_trimmed)

        energies.append(normalized_energy)
        omegas.append(omega)


    # Save CSV
    with open(f"{file_path}_Energy_vs_Freq.csv", "w") as f:
        f.write("omega,energy\n")
        for ω, E in zip(omegas, energies):
            f.write(f"{ω:.6e},{E:.6e}\n")

    # Linear fit
    def linear_fit(x, hbar): return hbar * x
    popt, _ = curve_fit(linear_fit, omegas, energies)
    hbar_est = popt[0]


    # Calculate residuals: (Measured Energy - Fitted Energy)
    energies_fit = linear_fit(np.array(omegas), hbar_est)
    residuals = np.array(energies) - energies_fit

    r2 = r2_score(energies, energies_fit)
    print(f"{title}: Linear Fit R² = {r2:.5f}")


    with open(f"{file_path}_HBar_Estimate.json", "w") as f_json:
        json.dump({"hbar_estimate": hbar_est, "units": "J·s"}, f_json, indent=2)

    # Plot
    plt.scatter(omegas, energies, label="Data")
    plt.plot(omegas, linear_fit(np.array(omegas), hbar_est), 'r--', label=f"Fit: E = ℏω\nℏ ≈ {hbar_est:.2e}")
    plt.xlabel("Angular Frequency ω [rad/s]")
    plt.ylabel("Energy [J]")
    plt.title(f"{title} Planck Constant Emergence: E vs ω")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Emergence_vs_W.png", dpi=dpi)
    plt.close()


    # Plot Residuals vs Omega
    plt.figure(figsize=(8, 5))
    plt.scatter(omegas, residuals, color='purple', label='Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Angular Frequency ω [rad/s]")
    plt.ylabel("Residual Energy [J]")
    plt.ylim(np.min(residuals) * 1.1, np.max(residuals) * 1.1)
    plt.title(f"{title} Residuals of Energy vs ω Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals.png", dpi=dpi)
    plt.close()


    plt.figure(figsize=(8, 5))
    plt.loglog(omegas, energies, 'o', label='Data')
    plt.loglog(omegas, linear_fit(np.array(omegas), hbar_est), 'r--', label=f"Fit: slope ≈ 1\nℏ ≈ {hbar_est:.2e}")
    plt.xlabel("Angular Frequency ω [rad/s] (log scale)")
    plt.ylabel("Energy [J] (log scale)")
    plt.title(f"{title} Log-Log Plot: Energy vs Angular Frequency")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_LogLog_E_vs_w.png", dpi=dpi)
    plt.close()




    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)