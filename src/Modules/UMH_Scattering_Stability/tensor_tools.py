
import numpy as np

def compute_strain_tensor(u_field):
    gradients = np.gradient(u_field.real)
    strain_tensor = np.zeros(u_field.shape + (3, 3))
    for i in range(3):
        for j in range(3):
            strain_tensor[..., i, j] = 0.5 * (gradients[i] * gradients[j])
    return strain_tensor

def compute_ricci_tensor(strain_tensor):
    return np.sum(strain_tensor, axis=(3, 4))

def compute_strain_scalar(strain_tensor):
    return np.trace(strain_tensor, axis1=-2, axis2=-1)
