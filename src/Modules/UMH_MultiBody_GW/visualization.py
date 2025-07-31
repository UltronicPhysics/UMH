import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tensor_slice(tensor, component=(0, 0), z_slice=64, file_path="", title="Tensor Component",dpi=300):
    """Plot a 2D slice of a specified tensor component at a given z-slice."""

    # Extract specified tensor component
    slice_2d = tensor[component[0], component[1], :, :, z_slice]

    plt.figure(figsize=(6, 5))
    plt.imshow(slice_2d.T, origin='lower', cmap='seismic')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(file_path,dpi=dpi)
    plt.close()
    print(f"{title}, Saved tensor slice plot to: {file_path}")

def plot_tensor_slice3d(tensor, title, output_dir, filename,dpi=300):
    if tensor.ndim != 3:
        raise ValueError("Expected a 3D tensor for slicing.")

    os.makedirs(output_dir, exist_ok=True)

    # Extract central slice along z-axis
    z_center = tensor.shape[2] // 2
    slice_2d = tensor[:, :, z_center]

    plt.figure(figsize=(6, 5))
    plt.imshow(slice_2d, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath,dpi=dpi)
    plt.close()
    print(f"Saved tensor slice plot to: {filepath}")


def plot_strain_slice(field, title, filename, output_dir="output", vmin=None, vmax=None,dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(field, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Strain")
    plt.title(title)
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath,dpi=dpi)
    plt.close()
    print(f"Saved plot to {filepath}")

def plot_energy_density(energy, title, filename, output_dir="output",dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.hist(energy.flatten(), bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Energy Density")
    plt.ylabel("Frequency")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath,dpi=dpi)
    plt.close()
    print(f"Saved energy density plot to {filepath}")



def plot_tensor_components(tensor, file_path, title="Tensor",dpi=300):
    """
    Plot 2D central z-slice of all 3x3 components of a rank-2 tensor field.
    Saves each component as an image.
    """

    z_mid = tensor.shape[3] // 2  # Assumes shape (3, 3, x, y, z)

    for i in range(3):
        for j in range(3):
            component = tensor[i, j, :, :, z_mid]
            plt.figure(figsize=(5, 4))
            plt.imshow(component.T, origin='lower', cmap='seismic')
            plt.colorbar(label='Magnitude')
            plt.title(f"{title} Component ({i}{j})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            plt.savefig(f"{file_path}_Component_{i}{j}.png",dpi=dpi)
            plt.close()





def plot_time_series_overlay(tensor_series, component=(0, 0), file_path="tensor_timeseries.png", title="Tensor Component", label="Tensor Component",dpi=300):
    """
    Plots the time evolution of a selected tensor component across simulation steps.

    Parameters:
    - tensor_series: list of np.ndarray, each tensor at a given time step (shape e.g., [4,4,N,N,N])
    - component: tuple, the tensor component to track (e.g., (0,0) for T_00)
    - save_as: str, filename to save the plot
    - label: str, label for the plot legend and Y-axis
    """

    values = []

    for tensor in tensor_series:
        # Extract selected component slice: 3D array
        comp_slice = tensor[component[0], component[1]]
        # Compute spatial average (could also use np.max or np.min for amplification studies)
        avg_value = np.mean(comp_slice)
        values.append(avg_value)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(values)), values, marker='o', linestyle='-', label=label)
    plt.xlabel("Time Step")
    plt.ylabel(f"Average {label}")
    plt.title(f"{title}: Time Evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path,dpi=dpi)
    plt.close()

    print(f"[INFO] Time-series overlay saved as '{file_path}'")
