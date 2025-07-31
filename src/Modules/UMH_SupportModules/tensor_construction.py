
import numpy as np

def compute_gradients(field, dx):
    grad_x = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dx)
    grad_z = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2 * dx)
    return grad_x, grad_y, grad_z

def compute_strain_tensor(field, T_u, dx):
    grad_x, grad_y, grad_z = compute_gradients(field, dx)

    T_xx = T_u * grad_x * grad_x
    T_yy = T_u * grad_y * grad_y
    T_zz = T_u * grad_z * grad_z
    T_xy = T_u * grad_x * grad_y
    T_xz = T_u * grad_x * grad_z
    T_yz = T_u * grad_y * grad_z

    tensor_fields = {
        'T_xx': T_xx, 'T_yy': T_yy, 'T_zz': T_zz,
        'T_xy': T_xy, 'T_xz': T_xz, 'T_yz': T_yz
    }
    return tensor_fields
