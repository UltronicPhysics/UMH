
import numpy as np

def laplacian(f, dx):
    return (
        -6 * f +
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
        np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2)
    ) / dx**2

def compute_curvature(tensor, dx):
    return laplacian(tensor, dx)

def compute_divergence(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx):
    div_x = (np.roll(T_xx, -1, axis=0) - np.roll(T_xx, 1, axis=0)) / (2 * dx) +             (np.roll(T_xy, -1, axis=1) - np.roll(T_xy, 1, axis=1)) / (2 * dx) +             (np.roll(T_xz, -1, axis=2) - np.roll(T_xz, 1, axis=2)) / (2 * dx)

    div_y = (np.roll(T_xy, -1, axis=0) - np.roll(T_xy, 1, axis=0)) / (2 * dx) +             (np.roll(T_yy, -1, axis=1) - np.roll(T_yy, 1, axis=1)) / (2 * dx) +             (np.roll(T_yz, -1, axis=2) - np.roll(T_yz, 1, axis=2)) / (2 * dx)

    div_z = (np.roll(T_xz, -1, axis=0) - np.roll(T_xz, 1, axis=0)) / (2 * dx) +             (np.roll(T_yz, -1, axis=1) - np.roll(T_yz, 1, axis=1)) / (2 * dx) +             (np.roll(T_zz, -1, axis=2) - np.roll(T_zz, 1, axis=2)) / (2 * dx)

    return div_x, div_y, div_z
