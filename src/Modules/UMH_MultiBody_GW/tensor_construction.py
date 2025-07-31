import numpy as np
from scipy.ndimage import convolve

def compute_gradient(field, dx):
    grad = np.gradient(field, dx, edge_order=2)
    return grad

def compute_laplacian(field, dx):
    grad = compute_gradient(field, dx)
    lap = sum(np.gradient(grad[i], dx, axis=i, edge_order=2) for i in range(3))
    return lap

def compute_ricci_tensor(field, dx):
    grad = compute_gradient(field, dx)
    lap = compute_laplacian(field, dx)
    R = np.zeros((3, 3) + field.shape)
    for i in range(3):
        for j in range(3):
            R[i, j] = grad[i] * grad[j] - lap  # Approximate expression
    return R

def compute_riemann_tensor(field, dx):
    grad = compute_gradient(field, dx)
    riemann = np.zeros((3, 3, 3, 3) + field.shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    riemann[i, j, k, l] = grad[i] * grad[j] - grad[k] * grad[l]  # Simplified form
    return riemann



def compute_einstein_tensor(ricci_tensor, scalar_curvature):
    """Compute simplified Einstein tensor: G = R - 1/2 g R_scalar"""
    G = np.zeros_like(ricci_tensor)
    for i in range(3):
        for j in range(3):
            G[i, j] = ricci_tensor[i, j] - 0.5 * scalar_curvature * (1 if i == j else 0)
    return G

def compute_scalar_curvature(ricci_tensor):
    """Compute scalar curvature as trace of Ricci tensor"""
    return sum(ricci_tensor[i, i] for i in range(3))


def compute_spatial_derivatives(field):
    """
    Compute first spatial derivatives using central differences.
    Returns a dictionary of derivatives: df/dx, df/dy, df/dz
    """
    kernel = np.array([-1, 0, 1]) / 2.0

    dfx = convolve(field, kernel[None, None, :], mode='nearest')
    dfy = convolve(field, kernel[None, :, None], mode='nearest')
    dfz = convolve(field, kernel[:, None, None], mode='nearest')

    return {'x': dfx, 'y': dfy, 'z': dfz}

def compute_second_derivatives(field):
    """
    Compute second spatial derivatives using central differences.
    Returns a dictionary of second derivatives: d2f/dx2, d2f/dy2, d2f/dz2
    """
    kernel = np.array([1, -2, 1])

    d2fx = convolve(field, kernel[None, None, :], mode='nearest')
    d2fy = convolve(field, kernel[None, :, None], mode='nearest')
    d2fz = convolve(field, kernel[:, None, None], mode='nearest')

    return {'xx': d2fx, 'yy': d2fy, 'zz': d2fz}

