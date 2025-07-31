import numpy as np
import os
import matplotlib.pyplot as plt

def compute_stress_energy_tensor(field, strain):
    # Tμν ~ grad(ψ) * grad(ψ) + V(ψ) for scalar field approximation
    grad = np.gradient(field, axis=(1, 2, 3))  # ∂i ψ
    T = np.zeros_like(strain)
    for i in range(3):
        T[i] = grad[i]**2
    potential = 0.5 * field**2 + 0.25 * field**4
    T_total = np.sum(T, axis=0) + potential
    return T_total

def check_bianchi_identity(einstein_tensor):
    div_x = np.gradient(einstein_tensor, axis=0)
    div_y = np.gradient(einstein_tensor, axis=1)
    div_z = np.gradient(einstein_tensor, axis=2)
    bianchi_residual = np.sqrt(div_x**2 + div_y**2 + div_z**2)
    return bianchi_residual

def estimate_weyl_tensor(riemann_tensor, ricci_tensor):
    # Very rough scalar proxy (trace-free part of Riemann)
    ricci_trace = np.trace(ricci_tensor) if ricci_tensor.ndim == 2 else np.mean(ricci_tensor)
    weyl_proxy = np.linalg.norm(riemann_tensor) - ricci_trace
    return weyl_proxy

def run_tensor_extensions():
    os.makedirs("outputs", exist_ok=True)

    field = np.load("outputs/full_field.npy")  # Simulated field snapshot
    strain = np.gradient(field, axis=(1, 2, 3))

    T = compute_stress_energy_tensor(field, strain)
    np.save("outputs/stress_energy_tensor_T.npy", T)

    einstein = np.load("outputs/einstein_full_tensor.npy")
    ricci = np.load("outputs/ricci_full_tensor.npy")
    riemann = np.load("outputs/riemann_full_tensor.npy")

    bianchi_res = check_bianchi_identity(einstein)
    np.save("outputs/bianchi_residual.npy", bianchi_res)
    plt.imshow(bianchi_res[:, :, bianchi_res.shape[2]//2])
    plt.title("Bianchi Residual (∇·Gμν)")
    plt.colorbar()
    plt.savefig("outputs/bianchi_residual.png")
    plt.close()

    weyl_proxy = estimate_weyl_tensor(riemann, ricci)
    print(f"[Validation] Weyl tensor proxy scalar: {weyl_proxy:.4e}")