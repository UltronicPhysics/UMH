"""
UMH_Tensor_Fermion.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Tensor Fermion Simulation.

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
    from ..UMH_Tensor.utils.UMH_Field_Utils import apply_soliton_potential, apply_su2_constraint
    from ..UMH_Tensor.utils.export_tools import export_metadata
    from ..UMH_Tensor.utils.tracking import plot_fermion_central2s_slice, plot_field_snapshot, track_su2_fidelity, track_energy_evolution, plot_su2_deviation, plot_energy_log, compute_winding_number_density_njit, plot_topological_density, plot_total_winding, plot_spinor_phase_vector, plot_spinor_components
    from ..UMH_Tensor.utils.tracking_advanced import apply_su3_constraint
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Tensor.utils.UMH_Lattice import Lattice
    from UMH_Tensor.utils.UMH_Field_Utils import apply_soliton_potential, apply_su2_constraint
    from UMH_Tensor.utils.export_tools import export_metadata
    from UMH_Tensor.utils.tracking import plot_fermion_central2s_slice, plot_field_snapshot, track_su2_fidelity, track_energy_evolution, plot_su2_deviation, plot_energy_log, compute_winding_number_density_njit, plot_topological_density, plot_total_winding, plot_spinor_phase_vector, plot_spinor_components
    from UMH_Tensor.utils.tracking_advanced import apply_su3_constraint


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



def run_fermion_test(config_overrides=None):
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

    title="UMH Tensor Fermion"
    file_root="UMH_Tensor"
    file_sub="UMH_Fermion"
    file_hdr="UMH_Tensor_Fermion"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    lattice = Lattice(size)
    lattice.initialize_spinor_fields()
    su2_log = []
    energy_log = []
    images_fs,images_top,images_pv,images_sc=[],[],[],[]
    total_winding_history = []
    su2_fidelity_history = []
    #energy_history = []



    num_snapshots = 10
    snapshot_steps = set(np.linspace(0, steps - 1, num_snapshots, dtype=int))

    print(f"{title}: Loop Starting, Lattice Size:{size}, Total Steps:{steps}.")
    for step in range(steps):
        lattice.apply_dynamics(dt)
        apply_su2_constraint(lattice.field)
        apply_su3_constraint(lattice.field)
        track_su2_fidelity(lattice.field, su2_log, step)
        track_energy_evolution(lattice, energy_log, step)

        if step in snapshot_steps or step == steps - 1:
            filename=plot_field_snapshot(lattice, step,file_path=file_path,title=title,dpi=dpi)
            images_fs.append(imageio.imread(filename))

            complex_field = lattice.get_complex_spinors()
            w_density = compute_winding_number_density_njit(complex_field)

            total_winding = np.sum(w_density)
            total_winding_history.append((step, total_winding))

            filename=plot_topological_density(lattice,w_density, step,file_path=file_path,title=title,dpi=dpi)
            images_top.append(imageio.imread(filename))

            filename=plot_spinor_phase_vector(lattice, step=step,file_path=file_path,title=title,dpi=dpi)
            images_pv.append(imageio.imread(filename))

            filename=plot_spinor_components(lattice, step=step,file_path=file_path,title=title,dpi=dpi)
            images_sc.append(imageio.imread(filename))

            # Compute spinor norm = |Ψ₁|² + |Ψ₂|²
            field = lattice.field
            norm = field[0]**2 + field[1]**2 + field[2]**2 + field[3]**2
            su2_error = np.mean((norm - 1.0)**2)
            su2_fidelity_history.append((step, su2_error))

            #ke = lattice.kinetic_energy()
            #pe = lattice.potential_energy()
            #ne = lattice.nonlinear_energy()
            #te = ke + pe + ne  # total energy

            #energy_history.append((step, ke, pe, ne, te))

            print(f"{title}: Loop Processing, Step:{step} of Total Steps:{steps}.")


    
    print(f"{title}: Loop Finished, Total Steps:{steps}.")

    imageio.mimsave(f"{file_path}_Field_Evolution.gif", images_fs, fps=3)
    imageio.mimsave(f"{file_path}_Topology_Evolution.gif", images_top, fps=3)
    imageio.mimsave(f"{file_path}_Phase_Vector_Evolution.gif", images_pv, fps=3)
    imageio.mimsave(f"{file_path}_Spinor_Components_Evolution.gif", images_sc, fps=3)

    plot_su2_deviation(su2_log, file_path=f"{file_path}_SU2_Fidelity",title=f"{file_path}_SU2_Fidelity")
    plot_energy_log(energy_log, file_path=f"{file_path}_Energy",title=f"{file_path}_Energy")


    plot_fermion_central2s_slice(lattice, file_path=file_path,title=title,dpi=dpi)

    np.savetxt(f"{file_path}_total_winding_vs_time.txt", total_winding_history, header="step total_winding")
    plot_total_winding(total_winding_history,file_path=file_path,title=title)

    np.savez(f"{file_path}_Run_Data.npz",
         total_winding=np.array(total_winding_history),
         fidelity=np.array(su2_fidelity_history),
         energy=np.array(energy_log))

    np.savetxt(f"{file_path}_SU2_Fidelity_vs_Time.txt", su2_fidelity_history, fmt="%.8e", header="step su2_error")

    with open(f"{file_path}_Energy_vs_Time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "KE", "PE", "NE", "TE"])
        writer.writerows(energy_log)


    metadata = {
        "initial_conditions": {
            "field_initialization": "Spinor field (normalized)",
            "SU(2)_enforced_at_start": True
        },
        "simulation_parameters": {
            "grid_size": size,
            "dt": dt,
            "steps": steps
        },
        "boundary_conditions": "Implicit periodic (unless overridden)",
        "topological_structure": "SU(2) spinor soliton assumed; SU(3) applied optionally"
    }
    export_metadata(metadata,file_path=f"{file_path}_Simulation_Parameters")

    print(f"✅ Finished Test: {title} Validated.")

    return lattice


if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    run_fermion_test(config)