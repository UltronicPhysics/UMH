
import numpy as np
from scipy.ndimage import convolve

def compute_curvature(strain):
    """
    Compute a simple Laplacian-based approximation of curvature.
    """
    kernel = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 1, 0],
                        [1,-6, 1],
                        [0, 1, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]])
    curvature = convolve(strain, kernel, mode='constant', cval=0.0)
    return curvature

def compute_divergence(tensor_field):
    """
    Compute the divergence of a 3D tensor field.
    """
    grad = np.gradient(tensor_field)
    divergence = np.zeros_like(tensor_field)
    for i in range(3):
        divergence += grad[i][i]
    return divergence

def compute_tensor_fields(strain_field):
    """
    Generate both curvature and divergence tensors from strain.
    """
    curvature = compute_curvature(strain_field)
    divergence = compute_divergence(curvature)
    return curvature, divergence
