import numpy as np
import numba
from numba import njit, prange

# 27-point Laplacian with absorbing boundary condition.
@njit(cache=True, parallel=True, fastmath=True)
def update_wave_27_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask): #_theory_tuned
    Nx, Ny, Nz = psi.shape
    inv_dx2 = 1.0 / dx**2
    center_coeff = 7.0 / 3.0  # = 2.333333 # Derived from Taylor expansion for 4th-order accuracy.  2.32 - 2.5.  2.32 could not dampen wave speed from numeric summing as much.  2.325 for UMH_Sim?

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                lap = 0.0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            cnt = abs(di) + abs(dj) + abs(dk)

                            if cnt == 1:
                                weight = 1.0 / 6.0   # Face neighbors
                            elif cnt == 2:
                                weight = 1.0 / 12.0  # Edge neighbors
                            elif cnt == 3:
                                weight = 1.0 / 24.0  # Corner neighbors

                            lap += weight * psi[i + di, j + dj, k + dk]

                lap = (lap - center_coeff * psi[i, j, k]) * inv_dx2

                damping = absorption_mask[i, j, k]
                psi_next[i, j, k] = damping * (
                    2.0 * psi[i, j, k] - psi_prev[i, j, k] + (dt ** 2) * v2 * lap
                )

                

# 7-point Laplacian with absorbing boundary condition.
@njit(cache=True, parallel=True, fastmath=True)
def update_wave_7_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask):
    Nx, Ny, Nz = psi.shape
    inv_dx2 = 1.0 / dx**2
    face_weight = 1.0 / 6.0
    center_coeff = 1.0  # Matches neighbor sum

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                lap = face_weight * (
                    psi[i+1, j, k] + psi[i-1, j, k] +
                    psi[i, j+1, k] + psi[i, j-1, k] +
                    psi[i, j, k+1] + psi[i, j, k-1]
                ) - center_coeff * psi[i, j, k]

                lap *= inv_dx2

                damping = absorption_mask[i, j, k]
                psi_next[i, j, k] = damping * (
                    2.0 * psi[i, j, k] - psi_prev[i, j, k] + (dt ** 2) * v2 * lap
                )



# 49-point Laplacian with absorbing boundary condition.
@njit(cache=True, parallel=True, fastmath=True)
def update_wave_49_wPML(psi, psi_prev, psi_next, v2, dt, dx, absorption_mask):
    Nx, Ny, Nz = psi.shape
    inv_dx2 = 1.0 / dx**2
    center_coeff = 2.37  # Adjusted for weaker face2 correction  center_coeff between 2.35 and 2.38.  2.35 might loosen the wave make it not as slow.

    center_x = Nx // 2
    center_y = Ny // 2
    center_z = Nz // 2
    damping_factor = 0.99  # Adjust between 0.98 - 0.995 if needed

    for i in prange(2, Nx - 2):  # Accommodate �2 neighbors
        for j in range(2, Ny - 2):
            for k in range(2, Nz - 2):
                lap = 0.0

                # Standard 27-point neighbors (�1)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            cnt = abs(di) + abs(dj) + abs(dk)

                            if cnt == 1:
                                weight = 0.16   # Face neighbors
                            elif cnt == 2:
                                weight = 1.0 / 12.0  # Edge neighbors
                            elif cnt == 3:
                                weight = 1.0 / 24.0  # Corner neighbors

                            lap += weight * psi[i + di, j + dj, k + dk]

                # Face2 neighbors �2
                face2_weight = -1.0 / 128.0

                lap += face2_weight * (
                    psi[i+2, j, k] + psi[i-2, j, k] +
                    psi[i, j+2, k] + psi[i, j-2, k] +
                    psi[i, j, k+2] + psi[i, j, k-2]
                )

                lap = (lap - center_coeff * psi[i, j, k]) * inv_dx2

                if i == center_x and j == center_y and k == center_z:
                    psi_next[i, j, k] = damping_factor * (
                        2.0 * psi[i, j, k] - psi_prev[i, j, k] + (dt ** 2) * v2 * lap
                    )
                else:
                    damping = absorption_mask[i, j, k]
                    psi_next[i, j, k] = damping * (
                        2.0 * psi[i, j, k] - psi_prev[i, j, k] + (dt ** 2) * v2 * lap
                    )



#Absorption Mask to remove any reflection from boundary areas of grid.
@njit(parallel=True, cache=True, fastmath=True)
def create_absorption_mask(shape, thickness, absorption_coeff, power=3, dtype=np.float64):
    Nx, Ny, Nz = shape
    mask = np.ones((Nx, Ny, Nz), dtype=dtype)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dist = min(i, Nx - 1 - i, j, Ny - 1 - j, k, Nz - 1 - k)
                if dist < thickness:
                    norm_dist = dist / thickness
                    damping_factor = absorption_coeff * (1.0 - norm_dist) ** power
                    mask[i, j, k] = np.exp(-damping_factor)
    return mask



#Initial Gaussian to create noise across a grid.
@njit(cache=True,parallel=True, fastmath=True)
def initialize_gaussian(psi, center, width):
    Nx, Ny, Nz = psi.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r2 = (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2
                psi[i, j, k] = np.exp(-r2 / (2 * width**2))



# Gaussian Kick to get a wave started.
@njit(cache=True, parallel=True, fastmath=True)
def apply_gaussian_kick(psi_prev, psi, center, kick_radius, kick_strength):
    Nx, Ny, Nz = psi.shape
    total_energy = 0.0

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r2 = (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2
                kick = kick_strength * np.exp(-r2 / (2 * kick_radius**2))
                psi[i, j, k] += kick  # This acts like an initial "velocity" condition
                total_energy += kick**2

    return total_energy


@njit(parallel=True, cache=True, fastmath=True)
def create_gaussian_field(psi, center, radius, strength):
    Nx, Ny, Nz = psi.shape
    gaussian = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    radius2 = 2.0 * radius ** 2

    for i in prange(Nx):
        dx = i - center[0]
        for j in range(Ny):
            dy = j - center[1]
            for k in range(Nz):
                dz = k - center[2]
                r2 = dx * dx + dy * dy + dz * dz
                gaussian[i, j, k] = strength * np.exp(-r2 / radius2)

    return gaussian




# Center Damping to remove numerical artifacts from Gaussian Kick.
@njit(parallel=True, cache=True, fastmath=True)
def apply_center_damping(psi, center, damp_radius=4, max_damping=0.95): #_quadratic
    Nx, Ny, Nz = psi.shape
    cx, cy, cz = center
    r2_max = damp_radius * damp_radius

    for i in prange(cx - damp_radius, cx + damp_radius + 1):
        if 0 <= i < Nx:
            for j in range(cy - damp_radius, cy + damp_radius + 1):
                if 0 <= j < Ny:
                    for k in range(cz - damp_radius, cz + damp_radius + 1):
                        if 0 <= k < Nz:
                            dx = i - cx
                            dy = j - cy
                            dz = k - cz
                            r2 = dx * dx + dy * dy + dz * dz
                            if r2 <= r2_max:
                                r = np.sqrt(r2)
                                # Quadratic falloff: strongest at center, fades toward edge
                                damping = 1.0 - (1.0 - max_damping) * (1.0 - (r / damp_radius) ** 2)
                                psi[i, j, k] *= damping



#Smooth 3d field.
@njit(parallel=True, cache=True, fastmath=True)
def smooth_field_3d(psi, kernel_size=1):
    Nx, Ny, Nz = psi.shape
    result = np.zeros_like(psi)
    kernel_range = 2 * kernel_size + 1
    kernel_volume = kernel_range ** 3

    for x in prange(kernel_size, Nx - kernel_size):
        for y in range(kernel_size, Ny - kernel_size):
            for z in range(kernel_size, Nz - kernel_size):
                acc = 0.0
                for dx in range(-kernel_size, kernel_size + 1):
                    for dy in range(-kernel_size, kernel_size + 1):
                        for dz in range(-kernel_size, kernel_size + 1):
                            acc += psi[x + dx, y + dy, z + dz]
                result[x, y, z] = acc / kernel_volume

    return result



#Smooth 2d field.
@njit(cache=True, parallel=True, fastmath=True)
def smooth_field_2d(input_field, kernel_size=1):
    nx, ny = input_field.shape
    output_field = np.zeros_like(input_field)
    kernel_range = 2 * kernel_size + 1
    kernel_area = kernel_range * kernel_range

    for i in prange(kernel_size, nx - kernel_size):
        for j in range(kernel_size, ny - kernel_size):
            acc = 0.0
            for di in range(-kernel_size, kernel_size + 1):
                for dj in range(-kernel_size, kernel_size + 1):
                    acc += input_field[i + di, j + dj]
            output_field[i, j] = acc / kernel_area

    return output_field



#Smooth numba array for jagged steps.
@njit(parallel=True, cache=True, fastmath=True)
def smooth_data_numba(data, window=5):
    smoothed = np.empty_like(data)
    half_window = window // 2
    N = len(data)

    for i in prange(N):
        total = 0.0
        count = 0
        for j in range(i - half_window, i + half_window + 1):
            if 0 <= j < N:
                total += data[j]
                count += 1
        smoothed[i] = total / count

    return smoothed











@njit(parallel=True, cache=True, fastmath=True)
def compute_laplacian_27point(Psi, dx):
    Nx, Ny, Nz = Psi.shape
    laplacian = np.zeros_like(Psi)
    inv_dx2 = 1.0 / dx**2
    center_coeff = 7.0 / 3.0  # Matches your UMH stencil

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                lap = 0.0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            cnt = abs(di) + abs(dj) + abs(dk)
                            if cnt == 1:
                                weight = 1.0 / 6.0   # Face neighbors
                            elif cnt == 2:
                                weight = 1.0 / 12.0  # Edge neighbors
                            elif cnt == 3:
                                weight = 1.0 / 24.0  # Corner neighbors

                            lap += weight * Psi[i + di, j + dj, k + dk]

                laplacian[i, j, k] = (lap - center_coeff * Psi[i, j, k]) * inv_dx2

    return laplacian



@njit(parallel=True, cache=True, fastmath=True)
def compute_divergence(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx):
    Nx, Ny, Nz = T_xx.shape
    div_x = np.zeros_like(T_xx)
    div_y = np.zeros_like(T_xx)
    div_z = np.zeros_like(T_xx)
    inv_2dx = 1.0 / (2.0 * dx)

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                div_x[i, j, k] = (
                    (T_xx[i + 1, j, k] - T_xx[i - 1, j, k]) +
                    (T_xy[i, j + 1, k] - T_xy[i, j - 1, k]) +
                    (T_xz[i, j, k + 1] - T_xz[i, j, k - 1])
                ) * inv_2dx

                div_y[i, j, k] = (
                    (T_xy[i + 1, j, k] - T_xy[i - 1, j, k]) +
                    (T_yy[i, j + 1, k] - T_yy[i, j - 1, k]) +
                    (T_yz[i, j, k + 1] - T_yz[i, j, k - 1])
                ) * inv_2dx

                div_z[i, j, k] = (
                    (T_xz[i + 1, j, k] - T_xz[i - 1, j, k]) +
                    (T_yz[i, j + 1, k] - T_yz[i, j - 1, k]) +
                    (T_zz[i, j, k + 1] - T_zz[i, j, k - 1])
                ) * inv_2dx

    return div_x, div_y, div_z







#Calculates DT based on Medium with a Safety factor for numerical simulation.
def calculate_cfl_dt(v=1.0, dx=1.0, dimensions=3, safety_factor=0.25):
    """
    Calculate a stable dt based on CFL condition for wave propagation in d dimensions.
    Assumes normalized units with c=1 and dx=1 unless specified.
    
    Args:
        c (float): Wave speed (UMH uses c = sqrt(T/ρ), default 1.0)
        dx (float): Grid spacing
        dimensions (int): Number of spatial dimensions (3D by default)
        safety_factor (float): Fraction of CFL limit to use (e.g., 0.25 for conservative)
    
    Returns:
        float: Recommended dt
    """
    cfl_limit = dx / (v * (dimensions ** 0.5))
    return cfl_limit * safety_factor

#Get Clamp for Safety factor function, this creates the minimum and maximum safety factor values.
def get_adaptive_safety_factor(grid_size, base=0.5, min_factor=0.35, max_factor=0.65):
    #correction = 1.0 - (grid_size / 128.0) * 0.10   #Light Correction Touch.
    correction = 1.0 - (grid_size / 384.0) * 0.30   #Light Correction Touch.
    factor = base * correction
    return min(max(factor, min_factor), max_factor)

#This gives us around 0.2.
def get_auto_cfl_clamp(grid_size, steps, max_cfl=0.45, min_cfl=0.35):
    #scale_factor = (grid_size / 256) * (steps / 500) * 0.005  # Halved scaling 0.00384
    scale_factor = (grid_size / 384) * (steps / 256) * 0.00384  # Gives us 0.00384.
    upper = max_cfl - scale_factor
    lower = min_cfl - (scale_factor * 0.5)
    return max(lower, 0.25), max(upper, 0.40)
