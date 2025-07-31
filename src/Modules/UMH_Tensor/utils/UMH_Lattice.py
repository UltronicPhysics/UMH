import numpy as np
from scipy.ndimage import laplace

from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True)
def compute_laplacian(field):
    laplacian = np.zeros_like(field)
    for i in prange(1, field.shape[1] - 1):
        for j in prange(1, field.shape[2] - 1):
            for k in prange(1, field.shape[3] - 1):
                for c in range(3):
                    laplacian[c, i, j, k] = (
                        field[c, i+1, j, k] + field[c, i-1, j, k] +
                        field[c, i, j+1, k] + field[c, i, j-1, k] +
                        field[c, i, j, k+1] + field[c, i, j, k-1] -
                        6.0 * field[c, i, j, k]
                    )
    return laplacian

@njit(cache=True, parallel=True, fastmath=True)
def compute_laplacian_channelwise(field):
    """
    Compute Laplacian for each channel using finite differences.
    Assumes field shape: (C, X, Y, Z)
    """
    C, X, Y, Z = field.shape
    result = np.zeros_like(field)

    for c in prange(C):
        for i in range(1, X - 1):
            for j in range(1, Y - 1):
                for k in range(1, Z - 1):
                    result[c, i, j, k] = (
                        -6.0 * field[c, i, j, k]
                        + field[c, i+1, j, k] + field[c, i-1, j, k]
                        + field[c, i, j+1, k] + field[c, i, j-1, k]
                        + field[c, i, j, k+1] + field[c, i, j, k-1]
                    )
    return result

class Lattice:
    def __init__(self, size):
        self.size = size
        self.field = np.zeros((3, size, size, size), dtype=np.float32)
        self.strain = np.zeros_like(self.field)
        self.energy = np.zeros((size, size, size), dtype=np.float32)

    #def initialize_spinor_fields(self):
    #    self.field = np.random.randn(3, self.size, self.size, self.size).astype(np.float32) * 0.01

    def initialize_spinor_fields(self):
        """
        Initialises SU(2) spinor with *real* channels:
            0 = Re(Ψ1) , 1 = Im(Ψ1)
            2 = Re(Ψ2) , 3 = Im(Ψ2)
        Field shape becomes (4, X, Y, Z), dtype=float32,
        so all existing float-based kernels keep working.
        """
        X = Y = Z = self.size

        # random normal, then small amplitude
        re1 = np.random.randn(X, Y, Z).astype(np.float32)
        im1 = np.random.randn(X, Y, Z).astype(np.float32)
        re2 = np.random.randn(X, Y, Z).astype(np.float32)
        im2 = np.random.randn(X, Y, Z).astype(np.float32)

        # compute norm = √(|Ψ1|² + |Ψ2|²)
        norm = np.sqrt(re1**2 + im1**2 + re2**2 + im2**2) + 1e-12
        re1, im1, re2, im2 = re1 / norm, im1 / norm, re2 / norm, im2 / norm

        self.field = np.stack([re1, im1, re2, im2])    # shape (4, X, Y, Z)



    def seed_scalar_soliton(self, center=True):
        x, y, z = np.indices((self.size, self.size, self.size))
        cx, cy, cz = self.size//2, self.size//2, self.size//2
        r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        soliton = np.exp(-r**2 / 20.0).astype(np.float32)
        self.field[0] = soliton

    def seed_multiple_solitons(self):
        self.seed_scalar_soliton()
        shift = self.size // 4
        for dx in [-shift, shift]:
            for dy in [-shift, shift]:
                for dz in [-shift, shift]:
                    self.field[0] += np.roll(np.roll(np.roll(self.field[0], dx, axis=0), dy, axis=1), dz, axis=2)

    def seed_quantum_stats(self):
        C, X, Y, Z = self.field.shape  # Extract dimensions from the field
        self.field = np.random.normal(scale=0.01, size=(C, X, Y, Z))

    def seed_soliton_configuration(self):
        self.seed_scalar_soliton()

    def seed_soliton_renormalize_configuration(self):
        N = self.size
        coords = np.arange(N, dtype=np.float32)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

        cx = cy = cz = N / 2
        r_squared = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2

        soliton = np.exp(-r_squared / 20.0).astype(np.float32)
        self.field[0] = soliton


    def apply_dynamics(self, dt=0.01):
        lap = compute_laplacian(self.field)
        self.field += dt * lap
        self.strain = lap

    def kinetic_energy(self):
        return 0.5 * np.sum(self.field**2)

    def potential_energy(self):
        return 0.5 * np.sum(self.strain**2)

    def nonlinear_energy(self):
        return 0.25 * np.sum(self.field**4)



    def compute_entropy(self):
        # Use |ψ|² as the probability density
        prob_density = np.abs(self.field[0])**2  # Assuming channel 0 holds amplitude
        prob_density /= np.sum(prob_density + 1e-12)  # Normalize to sum to 1
        entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))
        return entropy


    def compute_temperature(self):
        return np.mean(self.kinetic_energy())


    def get_complex_spinors(self):
        # Converts from real-valued representation to two complex spinors
        return np.stack([
            self.field[0] + 1j * self.field[1],
            self.field[2] + 1j * self.field[3]
        ])



    def get_curvature_tensor(self):
        """
        Computes curvature (as Laplacian) for real-valued field with shape (C, X, Y, Z).
        Returns same shape.
        """
        assert self.field.ndim == 4, "Expected field shape (C, X, Y, Z)"
        return compute_laplacian_channelwise(self.field)

