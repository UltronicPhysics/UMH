"""
UMH_Quantum_Stats.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Quantum_Stats Simulation.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces Wave Slices and 3d models.
"""
import numpy as np
import numba
import os
import sys
import json
import csv
import imageio.v2 as imageio
import matplotlib.pyplot as plt


try:
    from ..UMH_Tensor.utils.UMH_Lattice import Lattice
    from ..UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_einstein_tensor
    from ..UMH_Tensor.utils.export_tools import export_distribution_data, export_metadata
    from ..UMH_Tensor.utils.tracking import track_energy_evolution
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_einstein_tensor
    from UMH_Tensor.utils.export_tools import export_distribution_data, export_metadata
    from UMH_Tensor.utils.tracking import track_energy_evolution


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 128, #768,
        "TIME_STEPS": 200, #400

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)

        'GRID_SIZES': [64, 128, 256],

        "DT": 0.01,
        
        "PML_THICKNESS":20, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary.
        "PML_ABORPTION":0.10, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.04
        "PML_POWER": 4,

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }



def run_quantum_stats_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["LATTICE_SIZE"] # Memory-safe grid size.
    steps = config["TIME_STEPS"]

    dx=config["LATTICE_SPACING"]
    
    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    dt=config["DT"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]
    pml_power = config["PML_POWER"]

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Quantum Stats"
    file_root="UMH_Tensor"
    file_sub="UMH_Quantum_Stats"
    file_hdr="UMH_Quantum_Stats"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")
    
    
    lattice = Lattice(size)
    lattice.seed_quantum_stats()


    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))
    
    energy_log = []

    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)

        track_energy_evolution(lattice, energy_log, step)

        if step in snapshot_steps or step == steps - 1:
            
            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")



    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    with open(f"{file_path}_Quantum_Energy_vs_Time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "KE", "PE", "NE", "TE"])
        writer.writerows(energy_log)

        
    #export_distribution_data(energy_log, "quantum_stats")
    steps, ke, pe, ne, total = zip(*energy_log)


    # Plot energy components over time
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ke, label="Kinetic Energy", linestyle='--')
    plt.plot(steps, pe, label="Potential Energy", linestyle='-.')
    plt.plot(steps, ne, label="Nonlinear Energy", linestyle=':')
    plt.plot(steps, total, label="Total Energy", color='black', linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.yscale('log')
    plt.title(f"{title} Quantum Simulation: Energy vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}_Quantum_Energy_vs_Time.png",dpi=dpi)
    plt.close()


    metadata={
      "test_type": title,
      "description": "Quantum statistics evolution tracking over time using UMH lattice field simulation. Energies computed include kinetic, potential, nonlinear, and total.",
      "data_file": f"{file_path}_Quantum_Energy_vs_Time.csv",
      "fields": [
        {
          "name": "step",
          "description": "Simulation step index (discrete time)"
        },
        {
          "name": "kinetic_energy",
          "description": "Sum of kinetic energy over entire lattice"
        },
        {
          "name": "potential_energy",
          "description": "Sum of potential energy (typically quadratic in strain)"
        },
        {
          "name": "nonlinear_energy",
          "description": "Nonlinear self-interaction energy, e.g., Ψ⁴ term"
        },
        {
          "name": "total_energy",
          "description": "Total energy = KE + PE + NE"
        }
      ],
      "metadata": {
        "field_shape": "inferred from lattice at runtime",
        "units": "arbitrary units",
        "energy_conservation": "Evaluated by tracking total_energy across steps",
        "initialization": "Seeded with quantum noise (Gaussian, σ=0.01)",
        "num_steps": steps,
        "dt": dt,
      },
      "visualization": {
        "recommended_plot": "Line chart of all energy components vs. step",
        "possible_extension": [
          "Exponential decay fit to total_energy",
          "Variance or entropy of energy distribution"
        ]
      }
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")
    
    return energy_log


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_quantum_stats_test(config)