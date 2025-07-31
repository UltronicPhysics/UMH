"""
UMH_Gravity.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Tensor Gravity Simulation.

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
    from ..UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_snapshot, export_field_raw_auto, export_metadata
    from ..UMH_Tensor.utils.tracking import plot_tensor_norm_from_data, visualize_tensor_vectors
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_einstein_tensor
    from UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_snapshot, export_field_raw_auto, export_metadata
    from UMH_Tensor.utils.tracking import plot_tensor_norm_from_data, visualize_tensor_vectors


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




def run_gravity_tensor_test(config_overrides=None):
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

    title="UMH Tensor Gravity"
    file_root="UMH_Tensor"
    file_sub="UMH_Gravity"
    file_hdr="UMH_Tensor_Gravity"
  
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

    tensor_norm_history=[]
    energy_log = []
    images_fs,images_top,images_pv,images_sc=[],[],[],[]

    global_min = float('inf')
    global_max = float('-inf')

    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))
    
    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)

        tensor = lattice.field  # or lattice.gravity_tensor, if defined separately

        tensor_norm = np.sqrt(np.sum(np.abs(tensor)**2))  # Frobenius norm
        tensor_norm_history.append((step, tensor_norm))

       
        if step in snapshot_steps or step == steps - 1:
            
            export_field_raw_auto(lattice, step, file_path=file_path)

            #field = all_fields[step]  # however you store or recompute the field
            norm = np.linalg.norm(tensor, axis=0)
            z_slice = norm.shape[2] // 2
            slice_2d = norm[:, :, z_slice]

            global_min = min(global_min, np.min(slice_2d))
            global_max = max(global_max, np.max(slice_2d))

            print(f"[colormap] Global range: min={global_min}, max={global_max}")

            filename=export_field_snapshot(tensor, step, file_path=f"{file_path}_Txx",title=title,component_label="Txx", vmin=global_min, vmax=global_max)
            images_fs.append(imageio.imread(filename))

            background_field=np.linalg.norm(tensor, axis=0)
            #visualize_tensor_vectors(tensor, step, component=("xx", "xy"))
            visualize_tensor_vectors(lattice.field, background_field, step,file_path=file_path,title=title, component=("0", "1"))

            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")


    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)

    ricci = compute_ricci_tensor(lattice.strain)
    einstein = compute_einstein_tensor(ricci)

    export_tensor_fields(ricci, einstein, file_path=file_path)

    plot_tensor_norm_from_data(tensor_norm_history, steps=None, file_path=file_path, title=title,dpi=dpi)


    metadata={
      "field_shape": lattice.field.shape,
      "steps": steps,
      "dt": dt,
      "tensor_description": "Symmetric 3x3 gravity tensor field evolving under UMH dynamics",
      "norm_description": "Frobenius norm of full tensor at each step"
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")

    return ricci, einstein



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_gravity_tensor_test(config)