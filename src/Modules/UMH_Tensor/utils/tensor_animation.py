import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

def animate_T00_evolution(tensor_file="outputs/T_mu_nu_evolution.npy", output_file="outputs/T00_evolution.mp4"):
    T_series = np.load(tensor_file)  # shape (steps, 3, 3, Nx, Ny, Nz)
    os.makedirs("outputs", exist_ok=True)
    T00 = T_series[:, 0, 0]  # Extract energy density component over time
    z = T00.shape[-1] // 2  # Middle slice
    frames = []

    fig, ax = plt.subplots()
    def update(i):
        ax.clear()
        im = ax.imshow(T00[i, :, :, z], cmap='inferno', vmin=T00.min(), vmax=T00.max())
        ax.set_title(f"T00 at step {i}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T00.shape[0], blit=False)
    ani.save(output_file, fps=4, dpi=150)
    plt.close()
    print(f"[Saved] Animation: {output_file}")

def track_total_energy_momentum(tensor_file="outputs/T_mu_nu_evolution.npy"):
    T_series = np.load(tensor_file)
    os.makedirs("outputs", exist_ok=True)

    energy = np.sum(T_series[:, 0, 0], axis=(1, 2, 3))  # Integrate T00
    momentum_x = np.sum(T_series[:, 0, 1], axis=(1, 2, 3))  # T01
    momentum_y = np.sum(T_series[:, 0, 2], axis=(1, 2, 3))  # T02

    np.savetxt("outputs/total_energy_momentum.csv", np.column_stack([energy, momentum_x, momentum_y]),
               delimiter=",", header="Energy,Momentum_X,Momentum_Y", comments='')

    # Plot
    steps = np.arange(T_series.shape[0])
    plt.plot(steps, energy, label="Total Energy")
    plt.plot(steps, momentum_x, label="Momentum X")
    plt.plot(steps, momentum_y, label="Momentum Y")
    plt.xlabel("Step")
    plt.ylabel("Integrated Quantity")
    plt.title("Total Energy and Momentum vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/energy_momentum_plot.png")
    plt.close()
    print("[Saved] Energy and momentum time plot.")