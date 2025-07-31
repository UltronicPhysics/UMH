import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True)
def apply_su2_constraint(field):
    for i in prange(field.shape[1]):
        for j in prange(field.shape[2]):
            for k in prange(field.shape[3]):
                norm = 1e-12
                for c in range(field.shape[0]):
                    norm += field[c, i, j, k]**2
                norm = np.sqrt(norm)
                for c in range(field.shape[0]):
                    field[c, i, j, k] /= norm

@njit(cache=True, fastmath=True)
def apply_soliton_potential(field):
    field[:] -= 0.01 * field**3

@njit(cache=True, fastmath=True)
def compute_ricci_tensor(strain):
    return np.sum(strain, axis=0)

def compute_riemann_tensor(strain):
    return np.array(np.gradient(strain, axis=(1, 2, 3)))

@njit(cache=True, fastmath=True)
def compute_einstein_tensor(ricci):
    trace = np.mean(ricci)
    return ricci - 0.5 * trace

@njit(cache=True, fastmath=True)
def compute_riemann_tensor_manual(strain):
    dx = np.empty_like(strain)
    dy = np.empty_like(strain)
    dz = np.empty_like(strain)

    # Use central differences (interior points)
    dx[:,1:-1,:,:] = (strain[:,2:,:,:] - strain[:,:-2,:,:]) / 2.0
    dy[:,:,1:-1,:] = (strain[:,:,2:,:] - strain[:,:,:-2,:]) / 2.0
    dz[:,:,:,1:-1] = (strain[:,:,:,2:] - strain[:,:,:,:-2]) / 2.0

    # Boundary handling (e.g. forward/backward diff) left as an exercise
    return dx, dy, dz





