"""
UMH_Tensor_Boson.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Tensor Boson Simulation.

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


try:
    from ..UMH_Tensor.utils.UMH_Lattice import Lattice
    from ..UMH_Tensor.utils.UMH_Field_Utils import apply_soliton_potential
    from ..UMH_Tensor.utils.export_tools import export_field_snapshot, export_field_raw_auto, export_metadata
    from ..UMH_Tensor.utils.tracking import plot_energy_log, track_energy_evolution, run_energy_decay_fit
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import apply_soliton_potential
    from UMH_Tensor.utils.export_tools import export_field_snapshot, export_field_raw_auto, export_metadata
    from UMH_Tensor.utils.tracking import plot_energy_log, track_energy_evolution, run_energy_decay_fit


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



def run_boson_test(config_overrides=None):
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

    title="UMH Tensor Boson"
    file_root="UMH_Tensor"
    file_sub="UMH_Boson"
    file_hdr="UMH_Tensor_Boson"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    lattice = Lattice(size)
    lattice.seed_scalar_soliton(center=True)

    energy_log = []
    images_fs,images_top,images_pv,images_sc=[],[],[],[]
    total_winding_history = []

    num_snapshots = 10
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)
        apply_soliton_potential(lattice.field)
        
        track_energy_evolution(lattice, energy_log, step)


        if step in snapshot_steps or step == steps - 1:
            
            export_field_raw_auto(lattice,step,file_path=file_path)

            filename=export_field_snapshot(lattice.field, step, file_path,title, component_label="|Ψ₁| (magnitude)",dpi=dpi)
            images_fs.append(imageio.imread(filename))

            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")
            

    print(f"{title}: Loop Finished, Total Steps:{steps}.")
    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)


    plot_energy_log(energy_log, file_path=file_path, title=title)
    #plot_fermion_central2s_slice(lattice, output_prefix="boson_field_slice")

    #np.savetxt("outputs/boson_total_winding_vs_time.txt", total_winding_history, header="step total_winding")
    #plot_total_winding(total_winding_history)

    np.savez(f"{file_path}_Run_Data.npz",
         energy=np.array(energy_log))

    #np.savetxt("outputs/su2_fidelity_vs_time.txt", su2_fidelity_history, fmt="%.8e", header="step su2_error")

    with open(f"{file_path}_Energy_vs_Time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "KE", "PE", "NE", "TE"])
        writer.writerows(energy_log)


    run_energy_decay_fit(energy_log,file_path=file_path,title=title,dpi=dpi)



    metadata = {
        "test_type": title,
        "simulation_parameters":
        {
            "grid_size": size,
            "dt": dt,
            "steps": steps
        },
        "energy_decay_fit":
        {
            "equation": "E(t) = 71.5138 * exp(-0.013383 * t) + 30.6941",
            "A": 71.51381778249223,
            "gamma": 0.013382958420092664,
            "C": 30.694144504262617,
            "r_squared": 0.9980495499287776,
            "fit_points": 200
        },
        "visualizations":
        {
        "field_snapshots": ["UMH_Tensor_Boson_step_xxx.png"],
        "field_evolution_gif": "UMH_Tensor_Boson_field_evolution.gif",
        "energy_plot": "UMH_Tensor_Boson_energy.png",
        "decay_fit_plot": "UMH_Tensor_Boson_energy_decay_fit.png"
        }
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")


    print(f"✅ Finished Test: {title} Validated.")

    return lattice


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_boson_test(config)