import numpy as np
import os

def compute_stress_energy_tensor_4d(field, strain):
    # Simulate Tμν field (3x3 tensor at each spatial point)
    shape = field.shape[1:]  # spatial shape
    T_full = np.zeros((3, 3) + shape, dtype=np.float32)

    grads = np.gradient(field, axis=(1, 2, 3))  # ∂i ψ
    for mu in range(3):
        for nu in range(3):
            T_full[mu, nu] += grads[mu] * grads[nu]
    potential = 0.5 * field[0]**2 + 0.25 * field[0]**4
    for mu in range(3):
        T_full[mu, mu] += potential  # diagonal elements
    return T_full

def run_tensor_evolution_test(steps=10, dt=0.01, grid_size=64):
    from modules.utils.lattice import Lattice
    os.makedirs("outputs", exist_ok=True)
    lattice = Lattice(grid_size)
    lattice.seed_scalar_soliton()
    T_evolution = []

    for step in range(steps):
        lattice.apply_dynamics(dt=dt)
        strain = lattice.strain
        T = compute_stress_energy_tensor_4d(lattice.field, strain)
        T_evolution.append(T)

    T_evolution = np.array(T_evolution)  # shape: (steps, 3, 3, Nx, Ny, Nz)
    np.save("outputs/T_mu_nu_evolution.npy", T_evolution)
    print(f"[Saved] Full stress-energy tensor evolution over {steps} steps.")