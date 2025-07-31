""" 
UMH RedShift Test

Author: Andrew Dodge
Date: July 2025

Description:
Tests whether a RedShift under Ultronic Medium can occur without Universe expansion.

Parameters:
    - Redshift_z
    - Distance_Mpc

Output:
    - UMH_RedShift.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import sys
import json
from scipy.stats import linregress


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


def run(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    outdir=os.makedirs(config["OUTPUT_FOLDER"], exist_ok=True)



    dpi=config["DPI"]

    outdir = config["OUTPUT_FOLDER"]

    title="UMH RedShift"
    file_hdr="UMH_RedShift"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title} Files Will be Saved to {outdir}.")



    # Replace with your actual redshift and distance data
    data = {
        "Redshift_z": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        "Distance_Mpc": [40, 80, 120, 200, 280, 400]
    }


    df = pd.DataFrame(data)

    # Compute ln(z + 1)
    df["ln_z_plus_1"] = np.log(df["Redshift_z"] + 1)

    # Perform linear regression for UMH model: ln(z + 1) = alpha * x
    slope, intercept, r_value, p_value, std_err = linregress(df["Distance_Mpc"], df["ln_z_plus_1"])

    # Calculate fitted UMH redshifts
    df["UMH_ln_z_plus_1_fit"] = slope * df["Distance_Mpc"] + intercept
    df["UMH_z_fit"] = np.exp(df["UMH_ln_z_plus_1_fit"]) - 1

    # Hubble Law prediction: z = H0 * x / c
    H0 = 70  # Hubble constant in km/s/Mpc
    c = 3e5  # Speed of light in km/s
    df["Hubble_z_fit"] = H0 * df["Distance_Mpc"] / c

    # Print regression result
    print(f"UMH Model Fit: ln(z + 1) = {slope:.5f} * x + {intercept:.5f}")
    print(f"R-squared: {r_value**2:.5f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Distance_Mpc"], df["Redshift_z"], label="Observed z", color="black")
    plt.plot(df["Distance_Mpc"], df["UMH_z_fit"], label="UMH Fit", linestyle="--")
    plt.plot(df["Distance_Mpc"], df["Hubble_z_fit"], label="Hubble Law", linestyle=":")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Redshift (z)")
    plt.title(f"{title} Redshift vs Distance: UMH vs Hubble Law")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Redshift.png", dpi=dpi)
    plt.close()


    # Assuming you have: distances, z_observed from your dataset

    lnz_plus_1 = np.log(df["Redshift_z"] + 1)
    distances = df["Distance_Mpc"]
    distances_with_const = sm.add_constant(distances)

    model = sm.OLS(lnz_plus_1, distances_with_const)
    results = model.fit()

    # Predictions + Confidence Intervals
    predictions = results.get_prediction(distances_with_const)
    summary_frame = predictions.summary_frame(alpha=0.05)  # 95% CI

    # Residuals Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(distances, results.resid, color='purple', label='Residuals')
    plt.axhline(0, color='black', linestyle='dashed')
    plt.xlabel('Distance (Mpc)')
    plt.ylabel('Residuals (ln(z+1))')
    plt.title(f"{title} Residuals of UMH Log(z+1) Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Residuals.png", dpi=dpi)
    plt.close()


    # Plot with Confidence Bands
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, lnz_plus_1, color='black', label='Observed ln(z+1)')
    plt.plot(distances, results.fittedvalues, color='blue', linestyle='--', label='UMH Fit')
    plt.fill_between(distances, summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Distance (Mpc)')
    plt.ylabel('ln(z + 1)')
    plt.title(f"{title} Confidence Bands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Confidence_Bands.png", dpi=dpi)
    plt.close()


    # UMH Hypothetical Parameters
    attenuation_coeff = 2.4e-4  # Tune this to match your redshift scaling

    distances = np.linspace(0, 500, 100)  # 0 to 500 Mpc
    simulated_z = np.exp(attenuation_coeff * distances) - 1

    plt.figure(figsize=(10, 6))
    plt.plot(distances, simulated_z, label='UMH Simulated Redshift', color='green')
    plt.xlabel('Distance (Mpc)')
    plt.ylabel('Redshift (z)')
    plt.title(f"{title} Simulated UMH Redshift vs Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_path}_RedShift_vs_Distance.png", dpi=dpi)
    plt.close()




    print(f"✅ Finished Test: {title} Validation.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run(config)