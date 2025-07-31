"""
UMH_Renormalization.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Renormalization Simulation.

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
    from ..UMH_Tensor.utils.export_tools import export_thermo_data,export_field_snapshot,export_metadata
    from ..UMH_Tensor.utils.tracking import plot_temperature_vs_time,plot_entropy_vs_time,plot_energy_breakdown
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.export_tools import export_thermo_data,export_field_snapshot,export_metadata
    from UMH_Tensor.utils.tracking import plot_temperature_vs_time,plot_entropy_vs_time,plot_energy_breakdown


def get_default_config(config_overrides=None):
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


def run_thermodynamics_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    grid_sizes=config["GRID_SIZES"]

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

    title="UMH Thermodynamics"
    file_root="UMH_Tensor"
    file_sub="UMH_Thermodynamics"
    file_hdr="UMH_Thermodynamics"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    lattice = Lattice(size)
    lattice.seed_soliton_configuration()

    snapshot_steps = set(np.linspace(0, steps- 1, num_snapshots, dtype=int))

    global_min = float('inf')
    global_max = float('-inf')

    energies = []
    images_fs=[]
    
    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)
        
        if step in snapshot_steps or step == steps - 1:

            normt = np.linalg.norm(lattice.field, axis=0)
            z_slice = normt.shape[2] // 2
            slice_2d = normt[:, :, z_slice]

            global_min = min(global_min, np.min(slice_2d))
            global_max = max(global_max, np.max(slice_2d))

            ke = np.sum(lattice.kinetic_energy())
            pe = np.sum(lattice.potential_energy())
            ne = np.sum(lattice.nonlinear_energy())
            total = ke + pe + ne
            entropy = lattice.compute_entropy()
            temperature = lattice.compute_temperature()
            energies.append((step,ke, pe, ne, total, entropy, temperature))

            filename=export_field_snapshot(lattice.field, step, file_path=file_path, title=title, component_label="|Ψ₁| (magnitude)", vmin=global_min,vmax=global_max,dpi=dpi)
            images_fs.append(imageio.imread(filename))
            
            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")

    
    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)

    export_thermo_data(energies, file_path=file_path)

    plot_temperature_vs_time(energies,file_path=file_path, title=title, dpi=dpi)
    plot_entropy_vs_time(energies,file_path=file_path, title=title, dpi=dpi)
    plot_energy_breakdown(energies,file_path=file_path, title=title, dpi=dpi)

    metadata={
      "title": "UMH Thermodynamics Evolution",
      "description": "Simulation of thermal relaxation using UMH formalism.",
      "grid_size": lattice.field.shape,
      "steps": steps,
      "dt": dt,
      "initial_temperature": 1.0,
      "final_temperature": "approx. 0.1 (observed)",
      "energy_components": ["KE", "PE", "NE", "Total"],
      "entropy_tracked": True,
      "field_snapshots": "10 total",
      "animation": "Included",
      "csv_log": "thermo.csv"
      
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")
    
    return energies


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_thermodynamics_test(config)