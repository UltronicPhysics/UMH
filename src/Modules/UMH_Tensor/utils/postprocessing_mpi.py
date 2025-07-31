import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def aggregate_energy_logs(output_dir="outputs", pattern="mpi_energy_deviation_rank*.csv"):
    files = glob.glob(os.path.join(output_dir, pattern))
    plt.figure()
    for file in files:
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        steps = data[:, 1]
        total_energy = data[:, 5]
        rank = os.path.basename(file).split("rank")[1].split(".")[0]
        plt.plot(steps, total_energy, label=f"Rank {rank}")
    plt.xlabel("Step")
    plt.ylabel("Total Energy")
    plt.title("Per-Rank Energy Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/ranked_energy_plot.png")
    plt.close()
    print("[Saved] ranked_energy_plot.png")

def reconstruct_field_from_slices(output_dir="outputs", prefix="field", grid_size=64, size=4):
    assembled = np.zeros((3, grid_size, grid_size, grid_size), dtype=np.float32)
    offset = 0
    for rank in range(size):
        local = np.load(f"{output_dir}/{prefix}_rank{rank}.npy")
        z_size = local.shape[-1]
        assembled[:, :, :, offset:offset + z_size] = local
        offset += z_size
    np.save(f"{output_dir}/full_{prefix}.npy", assembled)
    print(f"[Reconstructed] full_{prefix}.npy")

def track_angular_momentum_and_pressure(tensor_file="outputs/T_mu_nu_evolution.npy"):
    T_series = np.load(tensor_file)  # shape: (steps, 3, 3, Nx, Ny, Nz)
    os.makedirs("outputs", exist_ok=True)

    Lz = []
    pressure_trace = []

    for step in range(T_series.shape[0]):
        T = T_series[step]
        x, y, z = np.indices(T[0, 0].shape)
        L = x * T[0,2] - y * T[0,1]
        Lz.append(np.sum(L))
        pressure = (T[1,1] + T[2,2]) / 2
        pressure_trace.append(np.mean(pressure))

    np.savetxt("outputs/angular_momentum_pressure.csv", np.column_stack([Lz, pressure_trace]),
               delimiter=",", header="Lz,PressureTrace", comments='')
    plt.plot(Lz, label="Lz (Angular Momentum)")
    plt.plot(pressure_trace, label="Pressure Trace")
    plt.legend()
    plt.title("Angular Momentum & Pressure Evolution")
    plt.xlabel("Step")
    plt.grid(True)
    plt.savefig("outputs/angular_momentum_pressure_plot.png")
    plt.close()
    print("[Saved] Angular momentum and pressure evolution plot.")