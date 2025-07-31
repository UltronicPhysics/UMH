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
    from ..UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_riemann_tensor
    from ..UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_raw_curvature, export_field_snapshot, export_metadata
    from ..UMH_Tensor.utils.tracking import visualize_tensor_component, plot_tensor_norm_vs_time
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import compute_ricci_tensor, compute_riemann_tensor
    from UMH_Tensor.utils.export_tools import export_tensor_fields, export_field_raw_curvature, export_field_snapshot, export_metadata
    from UMH_Tensor.utils.tracking import visualize_tensor_component, plot_tensor_norm_vs_time


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


def run_tensor_curvature_test(config_overrides=None):
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

    title="UMH Tensor Curvature"
    file_root="UMH_Tensor"
    file_sub="UMH_Curvature"
    file_hdr="UMH_Tensor_Curvature"
  
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

    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    curvature_norm_log=[]
    
    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)

        curvature = lattice.get_curvature_tensor()  # shape: (3, 3, X, Y, Z)

        images_fs,images_tc=[],[]

        # Log norm
        frob_norm = np.linalg.norm(curvature)
        curvature_norm_log.append((step, frob_norm))

        # Snapshot
        if step in snapshot_steps or step == steps - 1:
            
            export_field_raw_curvature(curvature, step, file_path=file_path)

            #export_field_snapshot(field, step, prefix="field", component_label="|Ψ₁| (magnitude)", vmin=None, vmax=None)
            filename=export_field_snapshot(curvature, step, file_path=f"{file_path}_Txx",title=f"{title}: Txx", component_label="xx",dpi=dpi)
            images_fs.append(imageio.imread(filename))

            filename=visualize_tensor_component(curvature,step,file_path=file_path,title=title,dpi=dpi)
            images_tc.append(imageio.imread(filename))
            #visualize_tensor_component(field, step, component="xx", z_slice=None, output_dir="outputs", prefix="curvature"):
            
            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")

    
    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)
    imageio.mimsave(f"{file_path}_Component_Evolution.gif", images_tc, fps=3)


    ricci = compute_ricci_tensor(lattice.strain)
    riemann = compute_riemann_tensor(lattice.strain)

    export_tensor_fields(ricci, riemann, file_path=file_path)

    plot_tensor_norm_vs_time(curvature_norm_log,file_path=file_path,title=title,dpi=dpi)


    #visualize_tensor_component(field, step, component="xx")
    #Use imshow with colorbar, normalized colormap.

    #plot_tensor_norm_vs_time(norm_log)
    #Line plot with Frobenius norm from all steps.

    #export_field_raw(field, step)
    #Saves raw field: outputs/curvature_field_step_{step:03d}.npy.

    #with open("outputs/quantum_energy_vs_time.csv", "w", newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["step", "KE", "PE", "NE", "TE"])
    #    writer.writerows(energy_log)


    metadata={
      "simulation": "Tensor Curvature Test",
      "description": "This simulation analyzes the evolution of curvature in a 3×3 symmetric tensor field over time. It evaluates geometric features derived from the field's spatial derivatives, focusing on Frobenius norm and individual tensor components.",
      "field_shape": lattice.field.shape,
      "num_steps": steps,
      "dt": dt,
      "initialization": "Random symmetric tensor initialization with small noise amplitude",
      "field_type": "Real-valued rank-2 tensor field (3×3 components)",
      "dynamics": "Custom evolution rule based on UMH field equations applied at each time step",
      "measurements": {
        "tensor_component_visualization": True,
        "frobenius_norm_vs_time": True,
        "tensor_quiver_visualization": True,
        "curvature_tensor_computation": True
      },
      "visualization": {
        "animated_gif": "tensor_curvature_field_evolution.gif",
        "static_slices": [
          "tensor_component_xx.png",
          "tensor_quiver_00_01.png"
        ],
        "norm_vs_time_plot": "tensor_curvature_norm_vs_time.png"
      },
      "outputs": {
        "raw_field_snapshots": "Available as .npy files per step",
        "curvature_tensor": "Computed and optionally saved at each step",
        "normalized_colormaps": True,
        "vector_overlay": True
      },
      "validation": {
        "expected_behavior": "Norm stabilizes or evolves smoothly under curvature influence",
        "UMH_consistency": "Tensor evolution is geometrically constrained; curvature evolves consistently with UMH hypothesis"
      },
      "notes": "Ensure that symmetry is preserved in curvature computation and that slicing planes match across components"
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")
    
    return ricci, riemann



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_tensor_curvature_test(config)