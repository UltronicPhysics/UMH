"""
UMH_MultiBody_GW.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH MultiBody Gravitational Wave Simulation.

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


try:
    from ..UMH_MultiBody_GW.dynamics import initialize_grid, update_field
    from ..UMH_MultiBody_GW.tensor_construction import compute_riemann_tensor,compute_ricci_tensor,compute_einstein_tensor
    from ..UMH_MultiBody_GW.tensor_curvature_divergence import compute_curvature, compute_divergence
    from ..UMH_MultiBody_GW.tensor_analysis import analyze_tensor_fields
    from ..UMH_MultiBody_GW.data_export import save_npy,save_csv
    from ..UMH_MultiBody_GW.visualization import plot_tensor_slice,plot_time_series_overlay,plot_tensor_components
    from ..UMH_MultiBody_GW.animation_generator import generate_animation,create_tensor_evolution_animation
    from ..UMH_MultiBody_GW.tensor_validation import validate_tensor_divergence
    from ..UMH_MultiBody_GW.injector import inject_solitons
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_MultiBody_GW.dynamics import initialize_grid, update_field
    from UMH_MultiBody_GW.tensor_construction import compute_riemann_tensor,compute_ricci_tensor,compute_einstein_tensor
    from UMH_MultiBody_GW.tensor_curvature_divergence import compute_curvature, compute_divergence
    from UMH_MultiBody_GW.tensor_analysis import analyze_tensor_fields
    from UMH_MultiBody_GW.data_export import save_npy,save_csv
    from UMH_MultiBody_GW.visualization import plot_tensor_slice,plot_time_series_overlay,plot_tensor_components
    from UMH_MultiBody_GW.animation_generator import generate_animation,create_tensor_evolution_animation
    from UMH_MultiBody_GW.tensor_validation import validate_tensor_divergence
    from UMH_MultiBody_GW.injector import inject_solitons


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 128, #768,
        "TIME_STEPS": 100, #400

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":30, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary.
        "PML_ABORPTION":0.15, #0.15 How impactful is the PML layers, adjust for reflection at boundary.
        "PML_POWER": 5,

        "DT":0.05,

        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }


def run(config_overrides=None):
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

    title="UMH MultiBody GW"
    file_hdr="UMH_MultiBody_GW"
  
    print(f"✅ Starting Test: {title} Validated.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_hdr)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    # === Parameters ===
    shape = (size, size, size)

    # === Initialize grid and inject multiple solitons ===
    field = initialize_grid(shape)
    inject_solitons(field, positions=[(32, 64, 64), (96, 64, 64)], amplitudes=[5.0, -5.0])

    
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    vmax=1
    vmin=0
    # === Time evolution loop ===
    ricci_series = []
    einstein_series = []

    print(f"{title}: Starting Loop, Grid Size[{size},{size},{size}] with Total Steps:{steps}.")
    for step in range(steps):
        update_field(field, dt, dx)
        ricci_tensor = compute_ricci_tensor(field, dx)
        scalar_curvature = np.trace(ricci_tensor, axis1=0, axis2=1)
        einstein_tensor = compute_einstein_tensor(ricci_tensor, scalar_curvature)

        ricci_series.append(ricci_tensor.copy())
        einstein_series.append(einstein_tensor.copy())

        if step in snapshot_steps:
            vmax = np.max(np.abs(field))
            vmin = -vmax
            print(f"{title}: Currently Processing: Step:{step} of Total Steps:{steps}.")


    print(f"{title}: Loop Finished.")

    # === Export final data ===
    np.save(f"{file_path}_Final_Field.npy", field)
    np.save(f"{file_path}_Ricci_Final.npy", ricci_tensor)
    np.save(f"{file_path}_Einstein_Final.npy", einstein_tensor)

    # === Analyze tensor fields ===

    # === Construct full Riemann tensor before analysis ===
    riemann_tensor = compute_riemann_tensor(field, dx)

    # === Analyze tensor fields ===
    analysis_results = analyze_tensor_fields(ricci_tensor, einstein_tensor, riemann_tensor, file_path=file_path, title=title)

    save_csv(f"{file_path}_Tensor_Analysis.csv", analysis_results)
    print(f"{title}: Saving Tensor Analysis.")

    # === Visualizations ===
    plot_tensor_slice(ricci_tensor, component=(0, 0), z_slice=64, file_path=f"{file_path}_Ricci_00.png" ,title=f"{title} Ricci 00", dpi=dpi)
    plot_tensor_slice(einstein_tensor, component=(0, 0), z_slice=64, file_path=f"{file_path}_Einstein_00.png" ,title=f"{title} Einstein 00", dpi=dpi)
    plot_tensor_components(ricci_tensor, file_path=f"{file_path}_Ricci", title="Ricci", dpi=dpi)
    plot_tensor_components(einstein_tensor, file_path=f"{file_path}_Einstein", title="Einstein", dpi=dpi)

    print(f"{title}: Finished Plotting Images.")

    # === Generate animation ===
    create_tensor_evolution_animation(ricci_series, f"{file_path}_Ricci_Evolution", f"{title} Ricci Evolution", component=(0, 0), vmin=vmin,vmax=vmax, dpi=dpi)
    create_tensor_evolution_animation(einstein_series, f"{file_path}_Einstein_Evolution", f"{title} Einstein Evolution", component=(0, 0), vmin=vmin,vmax=vmax, dpi=dpi)
    print(f"{title}: Finished Animations.")

    plot_time_series_overlay(ricci_series, component=(0, 0), file_path=f"{file_path}_Ricci_00_Timeseries.png", title=f"{title} Ricc (0,0)", label="Ricci (0,0)", dpi=dpi)


    print(f"✅ Finished Test: {title} Validated.")


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run()