"""
UMH_Multibodyn.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Tensor Multibody Simulation.

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
    from ..UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor
    from ..UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_snapshot, export_field_raw_auto, export_metadata
    from ..UMH_Tensor.utils.tracking import plot_vector_quiver_with_norm, plot_tensor_norm_vs_time
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor
    from UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_snapshot, export_field_raw_auto, export_metadata
    from UMH_Tensor.utils.tracking import plot_vector_quiver_with_norm, plot_tensor_norm_vs_time


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



def run_multibody_tensor_test(config_overrides=None):
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

    title="UMH Tensor Multibody"
    file_root="UMH_Tensor"
    file_sub="UMH_Multibody"
    file_hdr="UMH_Tensor_Multibody"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    lattice = Lattice(size)
    lattice.seed_multiple_solitons()

    norm_history=[]
    images_fs=[]

    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))
    
    global_min = float('inf')
    global_max = float('-inf')
    
    
    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)

        norm = np.sqrt(np.sum(np.abs(lattice.field)**2))
        norm_history.append((step, norm))

        if step in snapshot_steps or step == steps - 1:

            export_field_raw_auto(lattice, step, file_path=file_path)

            normt = np.linalg.norm(lattice.field, axis=0)
            z_slice = normt.shape[2] // 2
            slice_2d = normt[:, :, z_slice]

            global_min = min(global_min, np.min(slice_2d))
            global_max = max(global_max, np.max(slice_2d))

            mid_z = lattice.field.shape[-1] // 2
            slice_data = np.linalg.norm(lattice.field[..., mid_z], axis=0)
            vmin = np.min(slice_data)
            vmax = np.max(slice_data)

            plt.imshow(slice_data, cmap='inferno', vmin=global_min, vmax=global_max)
            plt.title(f"{title} Slice Step: {step}")
            plt.colorbar(label="||T|| (Frobenius norm)")
            plt.savefig(f"{file_path}_Slice_Step_{step}.png",dpi=dpi)
            plt.close()
            
            images_fs.append(imageio.imread(f"{file_path}_Slice_Step_{step}.png"))
            
            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")


    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)

    plot_vector_quiver_with_norm(lattice.field,"All",component=("0", "1"),z_slice=None,file_path=file_path,title=title,dpi=dpi)

    plot_tensor_norm_vs_time(norm_history,file_path=file_path,title=title,dpi=dpi)

    np.savetxt(f"{file_path}_Norm_vs_Time.csv", norm_history, delimiter=",", header="step,norm")



    ricci = compute_ricci_tensor(lattice.strain)
    export_tensor_fields(ricci, file_path=file_path)


    metadata={
      "field_shape": lattice.field.shape,
      "steps": steps,
      "dt": dt,
      "description": "Multibody tensor evolution under UMH"
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")
    
    return ricci


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_multibody_tensor_test(config)