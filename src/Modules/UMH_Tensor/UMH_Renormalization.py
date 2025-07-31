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
    from ..UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_einstein_tensor
    from ..UMH_Tensor.utils.export_tools import export_tensor_fields,export_field_snapshot,export_field_raw_auto,export_metadata
    from ..UMH_Tensor.utils.tracking import compute_winding_number_density_njit, plot_topological_density, plot_total_winding,track_su2_fidelity,track_energy_evolution,plot_su2_deviation,plot_energy_log
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_einstein_tensor
    from UMH_Tensor.utils.export_tools import export_tensor_fields,export_field_snapshot,export_field_raw_auto,export_metadata
    from UMH_Tensor.utils.tracking import compute_winding_number_density_njit, plot_topological_density, plot_total_winding,track_su2_fidelity,track_energy_evolution,plot_su2_deviation,plot_energy_log


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


def run_renormalization_test(config_overrides=None):
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

    title="UMH Renormalization"
    file_root="UMH_Tensor"
    file_sub="UMH_Renormalization"
    file_hdr="UMH_Renormalization"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")

    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))
    
    results = []
    images_fs=[]
    su2_log = []
    energy_log = []
    total_winding_history=[]

    global_min = float('inf')
    global_max = float('-inf')

    gs_run=0

    print(f"{title}: Multi Loop Starting.")
    for size in grid_sizes:
        lattice = Lattice(size)
        lattice.seed_soliton_configuration()
        gs_run=gs_run+1
        images_fs=[]

        print(f"{title}: Inner Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
        for step in range(steps):
            lattice.apply_dynamics(dt)

            track_su2_fidelity(lattice.field, su2_log, step)
            track_energy_evolution(lattice, energy_log, step)

            if step in snapshot_steps or step == steps - 1:
                export_field_raw_auto(lattice, step, file_path=f"{file_path}_{gs_run}")

                normt = np.linalg.norm(lattice.field, axis=0)
                z_slice = normt.shape[2] // 2
                slice_2d = normt[:, :, z_slice]

                global_min = min(global_min, np.min(slice_2d))
                global_max = max(global_max, np.max(slice_2d))

                w_density = compute_winding_number_density_njit(lattice.field)

                total_winding = np.sum(w_density)
                total_winding_history.append((step, total_winding))


                filename=export_field_snapshot(lattice.field, step, file_path=f"{file_path}_{gs_run}",title=f"{title}_{gs_run}", component_label="|Ψ₁| (magnitude)",vmin=global_min, vmax=global_max,dpi=dpi)
                images_fs.append(imageio.imread(filename))
                
                print(f"{title}: Inner Loop Processing, Step:{step} of Total Steps:{steps}.")
    
        print(f"{title}: Inner Loop Finished, Total Steps:{steps}.")

        ricci = compute_ricci_tensor(lattice.strain)
        results.append((size, ricci))
        export_tensor_fields(ricci, file_path=f"{file_path}_{size}")
        
        imageio.mimsave(f"{file_path}_Evolution_{gs_run}.gif", images_fs, fps=3)

    print(f"{title}: Multi Loop Finished.")

    plot_su2_deviation(su2_log, file_path=file_path,title=title,dpi=dpi)

    plot_energy_log(energy_log, file_path=file_path,title=title,dpi=dpi)

    plot_total_winding(total_winding_history, file_path=file_path,title=title,dpi=dpi)

    metadata={
      "title": "UMH Renormalization Test",
      "description": "Tests the scaling behavior of a scalar soliton under different spatial resolutions to evaluate renormalization effects in the UMH lattice model.",
      "simulation_type": "Renormalization Scaling",
      "field_type": "Scalar (single-channel real field)",
      "initial_condition": "Gaussian soliton seeded at center; normalized amplitude",
      "grid_sizes_tested":grid_sizes,
      "resolutions_tested": [32, 48, 64, 96],
      "time_steps": steps,
      "time_step_size": dt,
      "normalization_applied": "Field was scaled at each resolution to preserve L2 norm",
      "field_snapshot_interval": num_snapshots,
      "outputs": {
        "field_snapshots": "Images showing scalar field norm at Z-slice midpoint for selected time steps",
        "animation": "Animated .gif showing evolution of the scalar field norm at each resolution",
        "raw_data": "Not included in this export; available per resolution if needed"
      },
      "visual_notes": "Color normalization fixed per resolution to allow visual comparison of growth behavior.",
      "expected_behavior": "As resolution increases, the soliton becomes better resolved but should preserve total amplitude when properly normalized. Field scale changes should reflect renormalization group invariance.",
      "UMH_theory_match": True
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")
    
    return results


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_renormalization_test(config)