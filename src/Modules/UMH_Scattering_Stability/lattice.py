import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def compute_laplacian(u):
    laplacian = np.empty_like(u)
    nx, ny, nz = u.shape

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                laplacian[i, j, k] = (
                    -6 * u[i, j, k] +
                    u[i + 1, j, k] + u[i - 1, j, k] +
                    u[i, j + 1, k] + u[i, j - 1, k] +
                    u[i, j, k + 1] + u[i, j, k - 1]
                )

    # Optionally zero out the edges (since they weren't computed)
    laplacian[0, :, :] = 0
    laplacian[-1, :, :] = 0
    laplacian[:, 0, :] = 0
    laplacian[:, -1, :] = 0
    laplacian[:, :, 0] = 0
    laplacian[:, :, -1] = 0

    return laplacian




class Lattice:
    def __init__(self, size, tension, density, damping=0.005):
        self.size = size
        self.tension = tension
        self.density = density
        self.damping = damping
        self.u = np.zeros((size, size, size), dtype=np.complex128)
        self.v = np.zeros_like(self.u)

    def step(self, dt):
        laplacian = self.compute_laplacian(self.u)
        self.v += (self.tension / self.density) * laplacian * dt
        self.v *= (1 - self.damping)
        self.u += self.v * dt

    def compute_laplacian(self, u):
        return compute_laplacian(u)

    def compute_strain_magnitude(self):
        return np.abs(self.u)
