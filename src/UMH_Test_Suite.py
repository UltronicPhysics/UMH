"""
UMH_Test_Suite.py

Author: Andrew Dodge  
Date: June 2025

Description:
Provides a Main Menu to Initiate tests and ensure all prerequisites are in place, for example downloading CMB Data.

Inputs:
- None

Outputs:
- Output Directory with Completed test valdation data depending on which test was selected.
- Logs Directory Tee-d with any on screen output data from each test.
"""

import sys
import os
import shutil
import signal
import importlib
import subprocess
import json
import os
import tempfile

# --- Ctrl C Exit function ---
# Global flag
interrupted = False

# Define a SIGINT handler
def handle_sigint(signum, frame):
    global interrupted
    interrupted = True
    if os.getpid() == os.getppid():  # Only the parent process should log this
        print("\n[INFO] SIGINT received (Ctrl+C). Exiting...")

# Get single character input (cross-platform)
try:
    import msvcrt  # Windows
    def getch():
        return msvcrt.getch().decode()
except ImportError:
    import tty
    import termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


OUTPUT_FOLDER = "./Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOGS_FOLDER = "./Logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)

if os.path.exists("sim.lock"):
    os.remove("sim.lock")

def is_simulation_running():
    return os.path.exists("sim.lock")

def set_lock():
    open("sim.lock", "w").close()

def clear_lock():
    if os.path.exists("sim.lock"):
        os.remove("sim.lock")

def run(config: dict):
    print(f"Running {config['type']} for {config['duration']} seconds.")

def file_exists(path):
    return os.path.exists(path)

def run_simulation_in_subprocess(module_path: str, log_file_path: str, config: dict):
    # Save config to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(config, f)
        config_path = f.name

    # Build subprocess command
    #cmd = ["python", "-m", module_path, config_path]
    cmd = ["python", "-u", "-m", module_path, config_path]
    print(f"Starting simulation: {module_path}")
    print("-" * 40)

    # Run with live output
    try:
        if log_file_path is None:
            print(f"No Log File Set")
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    print(line, end='')
        else:
            print(f"LogFile saved to: {log_file_path}")
            with open(log_file_path, "w") as f:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        print(line, end='')     # print to console
                        f.write(line)           # write to log file
                        f.flush()               # optional: ensure it's written immediately

        print("\n[Simulation completed]")
    finally:
        os.remove(config_path)

def run_core_step_if_needed(module_path, output_file_path, log_file_path, overrides=None):
    if overrides is None:
        overrides = {}

    if log_file_path is not None:
        log_file_path=f"Logs/{log_file_path}"

    full_path = f"Output/{output_file_path}"
    print(f"\nChecking for output file: {full_path}")

    if file_exists(full_path):
        print(f"[✔] Output already exists at: {full_path}")
        choice = input("Do you want to re-run this step? (y = yes, s = skip) [s]: ").strip().lower()

        if choice not in ['y', 'yes']:
            print(f"[→] Skipping step for {module_path}")
            return
        else:
            print(f"[↻] Re-running step for {module_path}")
    else:
        print(f"[✘] Output not found. Running step...")

    print(f"LogFile1 saved to: {log_file_path}")
    run_simulation_in_subprocess(module_path, log_file_path, overrides)




# === Function wrappers ===
def wave_speed():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}
        #try:
        #    timesteps = input("Number of timesteps (blank for default): ").strip()
        #    if timesteps:
        #        overrides["TIMESTEPS"] = int(timesteps)

        #    amplitude = input("Soliton amplitude (blank for default): ").strip()
        #    if amplitude:
        #        overrides["SOLITON_AMPLITUDE"] = float(amplitude)
        #except ValueError:
        #    print("Invalid input. Using defaults.")

        run_core_step_if_needed(
            module_path="Modules.UMH_Wave_Propagation_Speed.UMH_Wave_Simulation",
            output_file_path="UMH_Wave_Propagation_Speed/UMH_WaveSpeed_Wave_Radius_vs_Time.png",
            log_file_path="UMH_Wave_Propagation_Speed_Output.log",
            overrides=overrides
        )
    finally:
        clear_lock()


def mass_energy():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_SimData.UMH_Simulation",
            output_file_path="UMH_SimData/UMH_SimData.npy",
            log_file_path="UMH_SimData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Mass_Energy.UMH_Mass_Energy",
            output_file_path="UMH_Mass_Energy/UMH_Mass_Energy_Einstein_tensor.npy",
            log_file_path="UMH_Mass_Energy_Output.log",
            overrides=overrides
        )
    finally:
        clear_lock()


def planck_emergence():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Planck_Emergence.UMH_Planck_Emergence",
            output_file_path="UMH_Planck_Emergence/UMH_Planck_Emergence_Energy_vs_Freq.csv",
            log_file_path="UMH_Planck_Emergence_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def soliton_stability():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Scattering_Stability.UMH_Boson_Soliton",
            output_file_path="UMH_Scattering_Stability/UMH_Boson_Soliton/UMH_Scattering_Stability_Boson_Evolution.gif",
            log_file_path="UMH_Scattering_Stability_Boson_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Scattering_Stability.UMH_Fermion_Soliton",
            output_file_path="UMH_Scattering_Stability/UMH_Fermion_Soliton/UMH_Scattering_Stability_Fermion_Evolution.gif",
            log_file_path="UMH_Scattering_Stability_Fermion_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Scattering_Stability.UMH_Gravity_Tensor",
            output_file_path="UMH_Scattering_Stability/UMH_Gravity_Tensor/UMH_Scattering_Stability_Gravity_Evolution.gif",
            log_file_path="UMH_Scattering_Stability_Gravity_Tensor_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def wave_chirp():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.PlanckData.Planck_Downloader",
            output_file_path="PlanckData/H-H1_LOSC_4_V1-1126259446-32.hdf5",
            log_file_path="PlanckData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_vs_LIGO.UMH_Chirp_Generator",
            output_file_path="UMH_vs_LIGO/UMH_Chirp_Generator_Dynamic.npz",
            log_file_path="UMH_vs_LIGO_Chirp_Generator_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_vs_LIGO.UMH_Ligo_Compiler",
            output_file_path="UMH_vs_LIGO/UMH_Ligo_Compiler_CMP_Hanford_Match_Score.txt",
            log_file_path="UMH_vs_LIGO_Compiler_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def tensor_validation():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Boson",
            output_file_path="UMH_Tensor/UMH_Boson/UMH_Tensor_Boson_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Boson_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Fermion",
            output_file_path="UMH_Tensor/UMH_Fermion/UMH_Tensor_Fermion_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Fermion_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Gravity_Tensor",
            output_file_path="UMH_Tensor/UMH_Gravity/UMH_Tensor_Gravity_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Gravity_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Multibody_Tensor",
            output_file_path="UMH_Tensor/UMH_Multibody/UMH_Tensor_Multibody_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Multibody_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Tensor_Curvature",
            output_file_path="UMH_Tensor/UMH_Curvature/UMH_Tensor_Curvature_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Curvature_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()

def multibody_wave():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_MultiBody_GW.UMH_MultiBody_GW",
            output_file_path="UMH_MultiBody_GW/UMH_MultiBody_GW_Final_Field.npy",
            log_file_path="UMH_MultiBody_GW_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()

def cmb_validation():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.PlanckData.Planck_Downloader",
            output_file_path="PlanckData/COM_CMB_IQU-smica_2048_R3.00_full.fits",
            log_file_path="PlanckData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_SimData.UMH_Simulation",
            output_file_path="UMH_SimData/UMH_SimData.npy",
            log_file_path="UMH_SimData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_vs_CMB.UMH_vs_CMB_Analyzer",
            output_file_path="UMH_vs_CMB/UMH_vs_CMB_Projected_Strain.fits",
            log_file_path="UMH_vs_CMB_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()

def supernova():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.PantheonData.Pantheon_Downloader",
            output_file_path="PantheonData/lcparam_full_long.csv",
            log_file_path="PantheonData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_vs_Pantheon.UMH_vs_Pantheon",
            output_file_path="UMH_vs_Pantheon/UMH_vs_Pantheon_Pantheon_Supernovae.png",
            log_file_path="UMH_vs_Pantheon_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def quantum_statics():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Quantum_Stats",
            output_file_path="UMH_Tensor/UMH_Quantum_Stats/UMH_Quantum_Stats_Quantum_Energy_vs_Time.csv",
            log_file_path="UMH_Tensor_Quantum_Stats_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def redshift():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_RedShift.UMH_RedShift",
            output_file_path="UMH_RedShift/UMH_RedShift_Redshift.png",
            log_file_path="UMH_RedShift_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def gauge_symmetry():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    print("This test is completed through Gauge Symmetry Tests, which can take a long time to complete.")

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_U1",
            output_file_path="UMH_Gauge_Symmetry/UMH_U1/UMH_Gauge_Symmetry_U1_Psi.npy",
            log_file_path="UMH_Gauge_Symmetry_U1_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_SU2",
            output_file_path="UMH_Gauge_Symmetry/UMH_SU2/UMH_Gauge_Symmetry_SU2_Psi.npy",
            log_file_path="UMH_Gauge_Symmetry_SU2_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Symmetry.UMH_Gauge_Symmetry_SU3",
            output_file_path="UMH_Gauge_Symmetry/UMH_SU3/UMH_Gauge_Symmetry_SU3_3d_Evolution.gif",
            log_file_path="UMH_Gauge_Symmetry_SU3_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def gauge_coupling():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    print("This test is completed through Gauge Coupling Tests, which can take a long time to complete.")

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Coupling.UMH_MagneticCoupling",
            output_file_path="UMH_Gauge_Coupling/UMH_Magnetic/UMH_Gauge_Coupling_Magnetic_Coupling_Constant.txt",
            log_file_path="UMH_Gauge_Coupling_Magnetic_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Coupling.UMH_WeakCoupling",
            output_file_path="UMH_Gauge_Coupling/UMH_Weak/UMH_Gauge_Coupling_Weak_Coupling_Constants.txt",
            log_file_path="UMH_Gauge_Coupling_Weak_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Coupling.UMH_StrongCoupling",
            output_file_path="UMH_Gauge_Coupling/UMH_Strong/UMH_Gauge_Coupling_Strong_Energy_Convergence.png",
            log_file_path="UMH_Gauge_Coupling_Strong_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_Gauge_Coupling.UMH_RunningCoupling",
            output_file_path="UMH_Gauge_Coupling/UMH_Running/UMH_Gauge_Coupling_Running_Coupling_Constants.txt",
            log_file_path="UMH_Gauge_Coupling_Running_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def entanglement():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Quantum_Entanglement.UMH_CHSH_Entanglement",
            output_file_path="UMH_Quantum_Entanglement/UMH_Quantum_Entanglement_Statistical_Results.txt",
            log_file_path="UMH_Quantum_Entanglement_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def stress_vs_einstein():
    print("This is automatically generated with the Gauge Symmetry Tests or less detailed with the Einstein Tensor Tests.")
    return

def ricci_scalar():
    print("This is automatically generated with the Gauge Symmetry Tests or less detailed with the Einstein Tensor Tests.")
    return

def gw_flux():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_SimData.UMH_Simulation",
            output_file_path="UMH_SimData/UMH_SimData.npy",
            log_file_path="UMH_SimData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.UMH_GW_Flux.UMH_GW_Flux",
            output_file_path="UMH_GW_Flux/UMH_GW_Flux_Bin_Centers.npy",
            log_file_path="UMH_GW_Flux_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def renormalization():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Renormalization",
            output_file_path="UMH_Tensor/UMH_Renormalization/UMH_Renormalization_Evolution_3.gif",
            log_file_path="UMH_Tensor_Renormalization_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def partition_function():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.UMH_Tensor.UMH_Thermodynamics",
            output_file_path="UMH_Tensor/UMH_Thermodynamics/UMH_Thermodynamics_Field_Evolution.gif",
            log_file_path="UMH_Tensor_Thermodynamics_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def get_prerequisites():
    if is_simulation_running():
        print("A simulation is already running. Please wait.")
        return

    try:
        set_lock()
        overrides = {}

        run_core_step_if_needed(
            module_path="Modules.PlanckData.Planck_Downloader",
            output_file_path="PlanckData/COM_CMB_IQU-smica_2048_R3.00_full.fits",
            log_file_path="PlanckData_Output.log",
            overrides=overrides
        )

        run_core_step_if_needed(
            module_path="Modules.PantheonData.Pantheon_Downloader",
            output_file_path="PantheonData/lcparam_full_long.csv",
            log_file_path="PantheonData_Output.log",
            overrides=overrides
        )

    finally:
        clear_lock()


def run_all_mechanical():
    wave_speed()
    mass_energy()
    planck_emergence()
    soliton_stability()


def run_all_cosmological():
    wave_chirp()
    tensor_validation()
    multibody_wave()
    cmb_validation()
    supernova()
    redshift()


def run_all_field_dynamics():
    quantum_statics()
    gauge_symmetry()
    gauge_coupling()
    entanglement()
    gw_flux()
    renormalization()
    partition_function()


def run_all():
    get_prerequisites()
    run_all_mechanical()
    run_all_cosmological()
    run_all_field_dynamics()


# === Grouped menu items ===
menu_sections = {
    "Mechanical Foundations": [
        (1, "Wave Propagation Speed Constancy", wave_speed),
        (2, "Mass-Energy Equivalence", mass_energy),
        (3, "Planck Constant Emergence", planck_emergence),
        (4, "Soliton Scattering and Stability", soliton_stability),
    ],
    "Cosmological Structure": [
        (5, "Gravitational Wave Chirp", wave_chirp),
        (6, "Einstein Tensor Validation", tensor_validation),
        (7, "Multibody Gravitational Wave Test", multibody_wave),
        (8, "CMB Angular Power, Angular, BAO", cmb_validation),
        (9, "Pantheon Supernova Validation", supernova),
        (10, "Hubble Redshift Without Expansion", redshift),
    ],
    "Gauge Symmetries and Field Dynamics": [
        (11, "Quantum Statics Emergence", quantum_statics),
        (12, "Topological Phase-Lock & Gauge Symmetry Dynamics", gauge_symmetry),
        (13, "Gauge Coupling Constant Derivation", gauge_coupling),
        (14, "Quantum Entanglement Behavior", entanglement),

        (15, "Stress-Energy Tensor vs. Einstein Tensor", stress_vs_einstein),
        (16, "Ricci Scalar Isotropy and Angular Spread", ricci_scalar),

        (17, "Gravitational Wave Energy Flux Decay", gw_flux),
        (18, "Renormalization Behavior from Strain Thresholding", renormalization),
        (19, "Thermodynamic Consistency and Partition Function", partition_function),
    ],
    "Run All": [
        (20, "Download All Prerequisites", get_prerequisites),
        (21, "Run All Mechanical Foundations", run_all_mechanical),
        (22, "Run All Cosmological Structure", run_all_cosmological),
        (23, "Run All Gauge Symmetries and Field Dynamics", run_all_field_dynamics),
        (24, "Run All", run_all),
    ]
}

# === Custom pager that shows menu in chunks if needed ===
def show_menu_with_paging():
    lines = ["\n=== Scientific Topics Menu ==="]
    for category, items in menu_sections.items():
        lines.append(f"\n--- {category} ---")
        for number, desc, _ in items:
            lines.append(f"{number}. {desc}")
    lines.append("\n0. Exit")

    # Get terminal height
    term_height = shutil.get_terminal_size().lines
    buffer_height = term_height - 2

    # Paginate output
    for i in range(0, len(lines), buffer_height):
        chunk = lines[i:i + buffer_height]
        print("\n".join(chunk))

        if i + buffer_height < len(lines):
            print("\n[Press Space or Enter to continue...]", end='', flush=True)
            while True:
                ch = getch()
                if ch in (' ', '\r', '\n'):
                    break
            # Clear the previous prompt line
            sys.stdout.write('\033[F\033[K')  # Move up one line, clear line
            sys.stdout.flush()


# === Main menu loop ===
def main():
    # Register the handler
    #signal.signal(signal.SIGINT, handle_sigint)

    while True:
        show_menu_with_paging()

        try:
            choice = int(input("\nEnter the number of your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if interrupted: break

        if choice == 0:
            print("Exiting.")
            break

        found = False
        for items in menu_sections.values():
            for number, desc, func in items:
                if choice == number:
                    print(f"\nRunning: {desc}\n")
                    func()
                    found = True
                    break
            if found:
                break

        if not found:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
