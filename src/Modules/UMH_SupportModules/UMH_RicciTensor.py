import numpy as np
import numba
from numba import njit, prange

try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence
   

@njit(cache=True,parallel=True)
def compute_gradient_components(field, dx):
    Nx, Ny, Nz = field.shape
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    gz = np.zeros_like(field)
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                gx[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
                gy[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dx)
                gz[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dx)
    return gx, gy, gz


@njit(cache=True,parallel=True)
def compute_ricci_tensor(strain, lap_strain, dx):
    Nx, Ny, Nz = strain.shape
    Ricci = np.zeros((3, 3, Nx, Ny, Nz), dtype=np.float32)
    gx, gy, gz = compute_gradient_components(strain, dx)

    for i in prange(3):
        for j in range(3):
            if i == 0:
                g_i = gx
            elif i == 1:
                g_i = gy
            else:
                g_i = gz

            g_ij = compute_gradient_components(g_i, dx)[j]

            for x in range(1, Nx - 1):
                for y in range(1, Ny - 1):
                    for z in range(1, Nz - 1):
                        if i == j:
                            Ricci[i, j, x, y, z] = lap_strain[x, y, z]
                        else:
                            Ricci[i, j, x, y, z] = g_ij[x, y, z]
    return Ricci



@njit(parallel=True, cache=True, fastmath=True)
def compute_scalar_curvature(R_xx, R_yy, R_zz):
    Nx, Ny, Nz = R_xx.shape
    scalar_curvature = np.zeros_like(R_xx)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                scalar_curvature[i, j, k] = R_xx[i, j, k] + R_yy[i, j, k] + R_zz[i, j, k]

    return scalar_curvature



@njit(parallel=True, cache=True, fastmath=True)
def compute_einstein_tensor(R_xx, R_yy, R_zz, scalar_curvature):
    Nx, Ny, Nz = R_xx.shape
    G_xx = np.zeros_like(R_xx)
    G_yy = np.zeros_like(R_xx)
    G_zz = np.zeros_like(R_xx)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                half_scalar = 0.5 * scalar_curvature[i, j, k]
                G_xx[i, j, k] = R_xx[i, j, k] - half_scalar
                G_yy[i, j, k] = R_yy[i, j, k] - half_scalar
                G_zz[i, j, k] = R_zz[i, j, k] - half_scalar

    return G_xx, G_yy, G_zz



@njit(parallel=True, cache=True, fastmath=True)
def compute_ricci_tensor_approx(strain, dx):
    Nx, Ny, Nz = strain.shape
    Ricci = np.zeros((3, 3, Nx, Ny, Nz), dtype=np.float64)
    gx, gy, gz = compute_gradient_components(strain, dx)

    lap_gx = compute_laplacian_27point(gx, dx)
    lap_gy = compute_laplacian_27point(gy, dx)
    lap_gz = compute_laplacian_27point(gz, dx)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Ricci[0, 0, i, j, k] = lap_gx[i, j, k]
                Ricci[1, 1, i, j, k] = lap_gy[i, j, k]
                Ricci[2, 2, i, j, k] = lap_gz[i, j, k]

                Ricci[0, 1, i, j, k] = 0.5 * (lap_gx[i, j, k] + lap_gy[i, j, k])
                Ricci[0, 2, i, j, k] = 0.5 * (lap_gx[i, j, k] + lap_gz[i, j, k])
                Ricci[1, 2, i, j, k] = 0.5 * (lap_gy[i, j, k] + lap_gz[i, j, k])

    return Ricci


def compute_ricci_tensor_from_components(curvatures, dx):
    Ricci_xx = compute_ricci_tensor_approx(curvatures['T_xx'], dx)
    Ricci_yy = compute_ricci_tensor_approx(curvatures['T_yy'], dx)
    Ricci_zz = compute_ricci_tensor_approx(curvatures['T_zz'], dx)
    return Ricci_xx[0, 0], Ricci_yy[1, 1], Ricci_zz[2, 2]



@njit(parallel=True, fastmath=True, cache=True)
def compute_einstein_tensor_complete(psi1, psi2, psi3, dx):
    Nx, Ny, Nz = psi1.shape
    # For simplicity, we'll just compute spatial G_ii (diagonal parts)
    G = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)  # 3 diagonal components

    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                # Laplacians for each channel and direction
                ricci = np.zeros(3, dtype=np.float64)
                for psi in (psi1, psi2, psi3):
                    # Second derivatives (Laplacian per direction)
                    d2psi_dx2 = (psi[i+1, j, k] - 2*psi[i, j, k] + psi[i-1, j, k]) / (dx*dx)
                    d2psi_dy2 = (psi[i, j+1, k] - 2*psi[i, j, k] + psi[i, j-1, k]) / (dx*dx)
                    d2psi_dz2 = (psi[i, j, k+1] - 2*psi[i, j, k] + psi[i, j, k-1]) / (dx*dx)
                    ricci[0] += d2psi_dx2.real  # or .real + .imag if you want trace, but typically .real
                    ricci[1] += d2psi_dy2.real
                    ricci[2] += d2psi_dz2.real
                # Ricci tensor (sum over all channels)
                # Ricci scalar (trace)
                ricci_scalar = ricci[0] + ricci[1] + ricci[2]
                for a in range(3):
                    G[a, i, j, k] = ricci[a] - 0.5 * ricci_scalar  # Einstein tensor (spatial diag components)
    return G  # shape (3, Nx, Ny, Nz)
