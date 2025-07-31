import numpy as np
import matplotlib.pyplot as plt
import os

def plot_scalar_field(field, title, filename):
    z = field.shape[-1] // 2
    plt.imshow(field[:, :, z], cmap='inferno')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"outputs/{filename}")
    plt.close()

def validate_curvature_and_energy_consistency():
    os.makedirs("outputs", exist_ok=True)

    ricci = np.load("outputs/ricci_full_tensor.npy")
    einstein = np.load("outputs/einstein_full_tensor.npy")
    riemann = np.load("outputs/riemann_full_tensor.npy")

    # Compute Ricci norm
    ricci_norm = np.linalg.norm(ricci)
    plot_scalar_field(np.abs(ricci), "Ricci Norm (|Rμν|)", "ricci_norm.png")

    # Compute Einstein norm
    einstein_norm = np.linalg.norm(einstein)
    plot_scalar_field(np.abs(einstein), "Einstein Norm (|Gμν|)", "einstein_norm.png")

    # Compute Riemann norm (over all components)
    riemann_norm = np.linalg.norm(riemann)
    plot_scalar_field(np.linalg.norm(riemann, axis=(0, 1, 2)), "Riemann Norm (|Rρσμν|)", "riemann_norm.png")

    print(f"[Validation] Ricci norm:    {ricci_norm:.5e}")
    print(f"[Validation] Einstein norm: {einstein_norm:.5e}")
    print(f"[Validation] Riemann norm:  {riemann_norm:.5e}")

    # Energy-momentum tensor consistency: check if G ≈ 8πT
    T = einstein / (8 * np.pi)
    recovered_G = 8 * np.pi * T
    residual = einstein - recovered_G
    max_error = np.max(np.abs(residual))
    mean_error = np.mean(np.abs(residual))
    print(f"[Check G = 8πT] Max error: {max_error:.3e}, Mean error: {mean_error:.3e}")
    plot_scalar_field(np.abs(residual), "Residual in G - 8πT", "g_minus_8piT.png")

    np.save("outputs/tensor_residual_G_minus_8piT.npy", residual)