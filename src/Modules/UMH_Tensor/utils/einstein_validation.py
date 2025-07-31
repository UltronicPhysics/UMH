import numpy as np
import matplotlib.pyplot as plt
import os

def plot_tensor_slice(tensor, z_index=None, title="", filename="slice.png"):
    if z_index is None:
        z_index = tensor.shape[-1] // 2
    plt.imshow(tensor[:, :, z_index], cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"outputs/{filename}")
    plt.close()

def validate_einstein_tensor(file_path="outputs/einstein_full_tensor.npy"):
    os.makedirs("outputs", exist_ok=True)
    T = np.load(file_path)

    # Slice plots
    plot_tensor_slice(T, title="Einstein Tensor Central Slice", filename="einstein_slice.png")

    # Divergence estimate (finite difference)
    grad_x = np.gradient(T, axis=0)
    grad_y = np.gradient(T, axis=1)
    grad_z = np.gradient(T, axis=2)
    div = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Norm of divergence
    divergence_norm = np.mean(np.abs(div))
    print(f"[Validation] Mean absolute divergence of Einstein tensor: {divergence_norm:.3e}")

    # Plot divergence norm slice
    plot_tensor_slice(div, title="Einstein Tensor Divergence", filename="einstein_divergence.png")

    # Save as npy
    np.save("outputs/einstein_divergence.npy", div)