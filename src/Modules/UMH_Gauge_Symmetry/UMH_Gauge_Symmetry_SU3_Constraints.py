import numpy as np
import numba
import os
import sys
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import matplotlib.colors as mcolors


from numba import njit, prange
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter1d
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates


try:
    from ..UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_Stencil import update_wave_27_wPML, initialize_gaussian, create_absorption_mask,update_wave_7_wPML,update_wave_49_wPML,apply_center_damping,apply_gaussian_kick,create_gaussian_field,smooth_data_numba,smooth_field_3d,compute_laplacian_27point,compute_divergence,calculate_cfl_dt,get_adaptive_safety_factor,get_auto_cfl_clamp
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components



@njit(parallel=True, fastmath=True, cache=True)
def enforce_su3_phase_constraint(psi1, psi2, psi3, blend=0.01, eps=1e-8):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                a = psi1[i, j, k]
                b = psi2[i, j, k]
                c = psi3[i, j, k]

                # Project to zero trace (a+b+c=0)
                avg = (a + b + c) / 3.0
                a_hat = a - avg
                b_hat = b - avg
                c_hat = c - avg

                # Soft normalize so |a_hat|^2 + |b_hat|^2 + |c_hat|^2 = original
                orig_norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2) + eps
                proj_norm = np.sqrt(np.abs(a_hat)**2 + np.abs(b_hat)**2 + np.abs(c_hat)**2) + eps

                if proj_norm > eps:
                    # Rescale projected components to preserve original norm
                    scale = orig_norm / proj_norm
                    a_proj = a_hat * scale
                    b_proj = b_hat * scale
                    c_proj = c_hat * scale

                    # Soft blend: keeps components from collapsing abruptly
                    psi1[i, j, k] = (1.0 - blend) * a + blend * a_proj
                    psi2[i, j, k] = (1.0 - blend) * b + blend * b_proj
                    psi3[i, j, k] = (1.0 - blend) * c + blend * c_proj
                else:
                    # If proj_norm is tiny, skip update (should be rare)
                    pass

    return psi1, psi2, psi3


@njit(parallel=True, fastmath=True, cache=True)
def apply_center_locking(psi1, psi2, psi3, cx, cy, cz, radius, strength):
    Nx, Ny, Nz = psi1.shape

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dx = i - cx
                dy = j - cy
                dz = k - cz
                r2 = dx*dx + dy*dy + dz*dz

                if r2 <= radius**2:
                    a = psi1[i, j, k]
                    b = psi2[i, j, k]
                    c = psi3[i, j, k]

                    # Store original total norm
                    orig_norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2)

                    # Phase-balance enforcement (subtract average)
                    avg = (a + b + c) / 3.0
                    a -= avg
                    b -= avg
                    c -= avg

                    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2)
                    if norm > 1e-8:
                        a /= norm
                        b /= norm
                        c /= norm

                        # Restore original amplitude
                        a *= orig_norm
                        b *= orig_norm
                        c *= orig_norm

                    # Optional phase lock
                    phase = np.angle(a + b + c)
                    a *= np.exp(-1j * phase)
                    b *= np.exp(-1j * phase)
                    c *= np.exp(-1j * phase)

                    # Radial falloff blend
                    distance = np.sqrt(r2)
                    fade = max(0.0, 1.0 - (distance / radius))
                    blend = min(1.0, strength * fade / 10.0)

                    psi1[i, j, k] = (1.0 - blend) * psi1[i, j, k] + blend * a
                    psi2[i, j, k] = (1.0 - blend) * psi2[i, j, k] + blend * b
                    psi3[i, j, k] = (1.0 - blend) * psi3[i, j, k] + blend * c

    return psi1, psi2, psi3

def generate_trefoil_points(num_points, scale, center):
    t_vals = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_list = scale * (np.sin(t_vals) + 2 * np.sin(2 * t_vals)) + center[0]
    y_list = scale * (np.cos(t_vals) - 2 * np.cos(2 * t_vals)) + center[1]
    z_list = scale * (-np.sin(3 * t_vals)) + center[2]
    return np.stack([x_list, y_list, z_list], axis=1)


@njit(parallel=True, cache=True, fastmath=True)
def initialize_su3_trefoil_knot(Nx, Ny, Nz, knot_points, grid_scale=0.02, amplitude=0.5, noise_level=1e-9):
    thickness = grid_scale * min(Nx, Ny, Nz)
    psi1 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    psi2 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    psi3 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    for idx in prange(len(knot_points)):
        x0, y0, z0 = knot_points[idx]
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    dx = i - x0
                    dy = j - y0
                    dz = k - z0
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= thickness**2:
                        r = np.sqrt(dx * dx + dy * dy + 1e-10)
                        theta = np.arctan2(dy, dx) if r > 0 else 0.0

                        phase1 = theta
                        phase2 = theta + 2.0 * np.pi / 3.0
                        phase3 = theta - 2.0 * np.pi / 3.0

                        decay = np.exp(-r2 / (2.0 * thickness**2))
                        amp = amplitude * decay

                        psi1[i, j, k] += amp * np.exp(1j * phase1)
                        psi2[i, j, k] += amp * np.exp(1j * phase2)
                        psi3[i, j, k] += amp * np.exp(1j * phase3)

    # Add a tiny bit of noise everywhere
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                noise1 = noise_level * (np.random.randn() + 1j * np.random.randn())
                noise2 = noise_level * (np.random.randn() + 1j * np.random.randn())
                noise3 = noise_level * (np.random.randn() + 1j * np.random.randn())
                psi1[i, j, k] += noise1
                psi2[i, j, k] += noise2
                psi3[i, j, k] += noise3

                norm = np.sqrt(
                    np.abs(psi1[i,j,k])**2 +
                    np.abs(psi2[i,j,k])**2 +
                    np.abs(psi3[i,j,k])**2
                )
                if norm > 1e-8:
                    psi1[i, j, k] /= norm
                    psi2[i, j, k] /= norm
                    psi3[i, j, k] /= norm

    return psi1, psi2, psi3




@njit(parallel=True, fastmath=True, cache=True)
def project_to_su3_old(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    psi1_proj = np.empty_like(psi1)
    psi2_proj = np.empty_like(psi2)
    psi3_proj = np.empty_like(psi3)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                v1 = psi1[i, j, k]
                v2 = psi2[i, j, k]
                v3 = psi3[i, j, k]

                # Normalize v1
                norm1 = np.sqrt(v1.real**2 + v1.imag**2)
                if norm1 > 1e-12:
                    v1 /= norm1
                else:
                    v1 = 1.0 + 0j

                # Ortho v2 to v1
                dot12 = v1.real * v2.real + v1.imag * v2.imag + 1j * (v1.real * v2.imag - v1.imag * v2.real)
                v2 = v2 - dot12 * v1
                norm2 = np.sqrt(v2.real**2 + v2.imag**2)
                if norm2 > 1e-12:
                    v2 /= norm2
                else:
                    v2 = 1.0j

                # Ortho v3 to v1 and v2
                dot13 = v1.real * v3.real + v1.imag * v3.imag + 1j * (v1.real * v3.imag - v1.imag * v3.real)
                dot23 = v2.real * v3.real + v2.imag * v3.imag + 1j * (v2.real * v3.imag - v2.imag * v3.real)
                v3 = v3 - dot13 * v1 - dot23 * v2
                norm3 = np.sqrt(v3.real**2 + v3.imag**2)
                if norm3 > 1e-12:
                    v3 /= norm3
                else:
                    # Use deterministic but nontrivial fallback
                    # E.g. alternate sign based on i+j+k to break symmetry
                    phase = ((i + j + k) % 8) * (2*np.pi/8)
                    v3 = (np.cos(phase) + 1j * np.sin(phase)) / np.sqrt(3.0)

                # Normalize triple so |v1|^2 + |v2|^2 + |v3|^2 == 1
                triple_norm = np.sqrt(np.abs(v1)**2 + np.abs(v2)**2 + np.abs(v3)**2)
                if triple_norm > 1e-12:
                    v1 /= triple_norm
                    v2 /= triple_norm
                    v3 /= triple_norm

                psi1_proj[i, j, k] = v1
                psi2_proj[i, j, k] = v2
                psi3_proj[i, j, k] = v3

    return psi1_proj, psi2_proj, psi3_proj



@njit(parallel=True, cache=True, fastmath=True) #Uses qr.
def project_to_su3_qr(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    psi1_proj = np.empty_like(psi1)
    psi2_proj = np.empty_like(psi2)
    psi3_proj = np.empty_like(psi3)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Collect local 3-vector
                v = np.empty(3, dtype=np.complex128)
                v[0] = psi1[i, j, k]
                v[1] = psi2[i, j, k]
                v[2] = psi3[i, j, k]

                # Gram-Schmidt orthonormalization (manual, complex)
                # Step 1: Normalize first vector
                u1 = v[0]
                norm1 = np.sqrt((u1.real**2 + u1.imag**2))
                if norm1 > 1e-12:
                    u1 = u1 / norm1
                else:
                    u1 = 1.0 + 0j  # fallback

                # Step 2: Orthogonalize second vector to first
                v2 = v[1] - (u1.conjugate() * v[1]) * u1
                norm2 = np.sqrt((v2.real**2 + v2.imag**2))
                if norm2 > 1e-12:
                    u2 = v2 / norm2
                else:
                    u2 = 1j  # fallback

                # Step 3: Orthogonalize third vector to first two
                v3 = v[2] - (u1.conjugate() * v[2]) * u1 - (u2.conjugate() * v[2]) * u2
                norm3 = np.sqrt((v3.real**2 + v3.imag**2))
                if norm3 > 1e-12:
                    u3 = v3 / norm3
                else:
                    u3 = (-1.0 + 0j)  # fallback

                # Step 4: Normalize so total amplitude is 1 (SU(3) norm)
                total_norm = np.sqrt(np.abs(u1)**2 + np.abs(u2)**2 + np.abs(u3)**2)
                if total_norm > 1e-12:
                    u1 /= total_norm
                    u2 /= total_norm
                    u3 /= total_norm

                psi1_proj[i, j, k] = u1
                psi2_proj[i, j, k] = u2
                psi3_proj[i, j, k] = u3

    return psi1_proj, psi2_proj, psi3_proj



@njit(parallel=True, fastmath=True, cache=True) #Fallback
def project_to_su3(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    psi1_proj = np.empty_like(psi1)
    psi2_proj = np.empty_like(psi2)
    psi3_proj = np.empty_like(psi3)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                v1 = psi1[i, j, k]
                v2 = psi2[i, j, k]
                v3 = psi3[i, j, k]
                # Step 1: normalize v1
                n1 = np.sqrt(v1.real**2 + v1.imag**2)
                if n1 > 1e-12:
                    u1 = v1 / n1
                else:
                    u1 = 1.0 + 0j
                # Step 2: v2 orthogonal to u1
                dot12 = u1.conjugate() * v2
                v2 = v2 - dot12 * u1
                n2 = np.sqrt(v2.real**2 + v2.imag**2)
                if n2 > 1e-12:
                    u2 = v2 / n2
                else:
                    u2 = 1j
                # Step 3: v3 orthogonal to u1 and u2
                dot13 = u1.conjugate() * v3
                dot23 = u2.conjugate() * v3
                v3 = v3 - dot13 * u1 - dot23 * u2
                n3 = np.sqrt(v3.real**2 + v3.imag**2)
                if n3 > 1e-12:
                    u3 = v3 / n3
                else:
                    # Use random phase rescue to ensure nonzero
                    phi = 2*np.pi*(i+j+k)/Nx
                    u3 = np.cos(phi) + 1j*np.sin(phi)
                # Final norm
                total = np.sqrt(np.abs(u1)**2 + np.abs(u2)**2 + np.abs(u3)**2)
                if total > 1e-12:
                    u1 /= total
                    u2 /= total
                    u3 /= total
                psi1_proj[i, j, k] = u1
                psi2_proj[i, j, k] = u2
                psi3_proj[i, j, k] = u3
    return psi1_proj, psi2_proj, psi3_proj






@njit(parallel=True, cache=True, fastmath=True)
def initialize_su3_full_grid(Nx, Ny, Nz, phase_amplitude=0.02, base_amplitude=0.0):
    psi1 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    psi2 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    psi3 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Random SU(3)-like complex components
                a1 = base_amplitude + phase_amplitude * (np.random.randn() + 1j * np.random.randn())
                a2 = base_amplitude + phase_amplitude * (np.random.randn() + 1j * np.random.randn())
                a3 = base_amplitude + phase_amplitude * (np.random.randn() + 1j * np.random.randn())
                
                # Normalize to unit vector in C^3
                norm = np.sqrt(np.abs(a1)**2 + np.abs(a2)**2 + np.abs(a3)**2)
                if norm != 0:
                    a1 /= norm
                    a2 /= norm
                    a3 /= norm

                psi1[i, j, k] = a1
                psi2[i, j, k] = a2
                psi3[i, j, k] = a3

    return psi1, psi2, psi3

@njit(parallel=True, cache=True, fastmath=True)
def apply_radial_damping(psi1, psi2, psi3, cx, cy, cz, r_max):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dx = i - cx
                dy = j - cy
                dz = k - cz
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                s = r / r_max
                if s > 1.0:
                    s = 1.0
                damp = np.exp(-s**4)
                psi1[i, j, k] *= damp
                psi2[i, j, k] *= damp
                psi3[i, j, k] *= damp



@njit(parallel=True, cache=True, fastmath=True)
def renormalize_su3_fields_OLD(psi1, psi2, psi3, max_norm=1.0):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Use real/imag for full Numba speed, or keep abs() if you prefer
                norm = np.sqrt(
                    psi1[i, j, k].real**2 + psi1[i, j, k].imag**2 +
                    psi2[i, j, k].real**2 + psi2[i, j, k].imag**2 +
                    psi3[i, j, k].real**2 + psi3[i, j, k].imag**2
                )
                if norm > max_norm:
                    scale = max_norm / norm
                    psi1[i, j, k] *= scale
                    psi2[i, j, k] *= scale
                    psi3[i, j, k] *= scale
    return psi1, psi2, psi3

@njit(parallel=True, fastmath=True, cache=True)
def renormalize_su3_fields(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                norm = np.sqrt(
                    np.abs(psi1[i, j, k])**2 +
                    np.abs(psi2[i, j, k])**2 +
                    np.abs(psi3[i, j, k])**2
                )
                if norm > 1e-12:
                    psi1[i, j, k] /= norm
                    psi2[i, j, k] /= norm
                    psi3[i, j, k] /= norm
    return psi1, psi2, psi3


@njit(parallel=True, cache=True, fastmath=True)
def compute_dynamic_center(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    total = 0.0
    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0

    max_density = 0.0

    # First pass: find max density
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                norm = (
                    np.abs(psi1[i, j, k])**2 +
                    np.abs(psi2[i, j, k])**2 +
                    np.abs(psi3[i, j, k])**2
                )
                if norm > max_density:
                    max_density = norm

    threshold = 0.5 * max_density  # only count strong core

    # Second pass: weighted sum over high-density regions
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                norm = (
                    np.abs(psi1[i, j, k])**2 +
                    np.abs(psi2[i, j, k])**2 +
                    np.abs(psi3[i, j, k])**2
                )
                if norm >= threshold:
                    total += norm
                    x_sum += i * norm
                    y_sum += j * norm
                    z_sum += k * norm

    if total > 1e-12:
        cx = int(x_sum / total)
        cy = int(y_sum / total)
        cz = int(z_sum / total)
    else:
        # fallback to center
        cx = Nx // 2
        cy = Ny // 2
        cz = Nz // 2

    return cx, cy, cz


@njit(parallel=True, fastmath=True, cache=True)
def apply_su3_constraint_hard(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                v1 = psi1[i, j, k]
                v2 = psi2[i, j, k]
                v3 = psi3[i, j, k]

                # Normalize v1
                norm1 = np.sqrt(v1.real**2 + v1.imag**2)
                if norm1 > 1e-8:
                    v1 /= norm1
                else:
                    v1 = 1.0 + 0j  # fallback

                # Orthogonalize v2 to v1
                proj12 = v1.conjugate() * v2
                v2 = v2 - proj12 * v1
                norm2 = np.sqrt(v2.real**2 + v2.imag**2)
                if norm2 > 1e-8:
                    v2 /= norm2
                else:
                    v2 = 1.0j

                # Orthogonalize v3 to v1 and v2
                proj13 = v1.conjugate() * v3
                proj23 = v2.conjugate() * v3
                v3 = v3 - proj13 * v1 - proj23 * v2
                norm3 = np.sqrt(v3.real**2 + v3.imag**2)
                if norm3 > 1e-8:
                    v3 /= norm3
                else:
                    v3 = -1.0 + 0j

                psi1[i, j, k] = v1
                psi2[i, j, k] = v2
                psi3[i, j, k] = v3
    return psi1, psi2, psi3




#def apply_localized_kick(psi, velocity, cx, cy, cz, strength=0.01, threshold_ratio=0.1):
#    size = psi.shape[0]  # assumes cubic grid
#    X, Y, Z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
#    angle = np.arctan2(Y - cy, X - cx)
#    core_mask = np.abs(psi) > (threshold_ratio * np.max(np.abs(psi)))
#    velocity[core_mask] += psi[core_mask] * (1j * strength * angle[core_mask])
#    return velocity

def apply_localized_kick_njit(psi, velocity, cx, cy, cz, strength=0.01, core_radius=None):
    Nx, Ny, Nz = psi.shape
    if core_radius is None:
        core_radius = int(Nx / 24)  # Scale with grid size!
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dx = i - cx
                dy = j - cy
                dz = k - cz
                r2 = dx*dx + dy*dy + dz*dz
                if r2 <= core_radius**2:
                    # Apply small phase kick
                    phi = np.arctan2(dy, dx)
                    velocity[i, j, k] += psi[i, j, k] * (1j * strength * phi)


def apply_localized_kick(psi, velocity, cx, cy, cz, strength=0.005, core_radius=None):
    Nx, Ny, Nz = psi.shape
    if core_radius is None:
        core_radius = int(Nx / 24)
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    r2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    mask = r2 <= core_radius**2
    phi = np.arctan2(Y - cy, X - cx)
    velocity[mask] += psi[mask] * (1j * strength * phi[mask])





@njit(parallel=True, fastmath=True, cache=True)
def su3_symmetric_normalize(psi1, psi2, psi3):
    Nx, Ny, Nz = psi1.shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                a = psi1[i, j, k]
                b = psi2[i, j, k]
                c = psi3[i, j, k]
                amps = np.array([np.abs(a), np.abs(b), np.abs(c)])
                avg_amp = np.mean(amps)
                # Rescale each to match avg_amp, preserving phase
                if amps[0] > 0:
                    a = a * (avg_amp / amps[0])
                if amps[1] > 0:
                    b = b * (avg_amp / amps[1])
                if amps[2] > 0:
                    c = c * (avg_amp / amps[2])
                # Normalize triple
                triple_norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2)
                if triple_norm > 1e-12:
                    a /= triple_norm
                    b /= triple_norm
                    c /= triple_norm
                psi1[i, j, k] = a
                psi2[i, j, k] = b
                psi3[i, j, k] = c
    return psi1, psi2, psi3




@njit(parallel=True, fastmath=True, cache=True)
def compute_stress_energy_tensor(psi1, psi2, psi3, dx):
    Nx, Ny, Nz = psi1.shape
    T = np.zeros((3, 3, Nx, Ny, Nz), dtype=np.float64)
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                # Gradients for each field/channel
                grads = np.zeros((3, 3), dtype=np.complex128)  # [channel, spatial direction]
                for ch, psi in enumerate((psi1, psi2, psi3)):
                    grads[ch, 0] = (psi[i+1, j, k] - psi[i-1, j, k]) / (2*dx)  # x
                    grads[ch, 1] = (psi[i, j+1, k] - psi[i, j-1, k]) / (2*dx)  # y
                    grads[ch, 2] = (psi[i, j, k+1] - psi[i, j, k-1]) / (2*dx)  # z
                # Build T_ij = sum_channels (d_i psi^*)(d_j psi)
                for a in range(3):
                    for b in range(3):
                        T_ij = 0.0
                        for ch in range(3):
                            T_ij += (np.conj(grads[ch, a]) * grads[ch, b]).real
                        T[a, b, i, j, k] = T_ij
    return T



@njit(parallel=True, fastmath=True, cache=True)
def compute_stress_energy_density(psi1, psi2, psi3, vel1, vel2, vel3, dx, m2=0.0):
    Nx, Ny, Nz = psi1.shape
    energy_density = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                # --- Kinetic Energy ---
                ke = 0.0
                for vel in (vel1, vel2, vel3):
                    ke += np.abs(vel[i, j, k])**2
                ke *= 0.5

                # --- Gradient Energy ---
                grad2 = 0.0
                for psi in (psi1, psi2, psi3):
                    dpsi_dx = (psi[i+1, j, k] - psi[i-1, j, k]) / (2*dx)
                    dpsi_dy = (psi[i, j+1, k] - psi[i, j-1, k]) / (2*dx)
                    dpsi_dz = (psi[i, j, k+1] - psi[i, j, k-1]) / (2*dx)
                    grad2 += (
                        np.abs(dpsi_dx)**2 +
                        np.abs(dpsi_dy)**2 +
                        np.abs(dpsi_dz)**2
                    )
                grad2 *= 0.5

                # --- Potential Energy (default: harmonic) ---
                pot = m2 * (np.abs(psi1[i, j, k])**2 +
                            np.abs(psi2[i, j, k])**2 +
                            np.abs(psi3[i, j, k])**2)
                # For zero potential, set m2=0.0

                # --- Total Energy Density ---
                energy_density[i, j, k] = ke + grad2 + pot
    return energy_density



def compute_ricci_scalar(T_xx, T_yy, T_zz, T_xy, T_xz, T_yz, dx):
    # Second derivatives (Laplacians) of the stress-energy tensor components
    d2T_xx = compute_laplacian_27point(T_xx, dx)
    d2T_yy = compute_laplacian_27point(T_yy, dx)
    d2T_zz = compute_laplacian_27point(T_zz, dx)

    # Mixed second derivatives: divergence of off-diagonal terms
    d2T_xy = compute_laplacian_27point(T_xy, dx)
    d2T_xz = compute_laplacian_27point(T_xz, dx)
    d2T_yz = compute_laplacian_27point(T_yz, dx)

    # Combine components: this is a proxy for Ricci scalar
    R_scalar = d2T_xx + d2T_yy + d2T_zz - (d2T_xy + d2T_xz + d2T_yz)
    return R_scalar



@njit(parallel=True, fastmath=True, cache=True)
def sample_ricci_angular_spread(R_scalar, radius, thetas):
    Nx, Ny, Nz = R_scalar.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    num_angles = len(thetas)

    results = np.full(num_angles, np.nan)

    for i in prange(num_angles):
        theta = thetas[i]
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta))
        z = cz
        if 0 <= x < Nx and 0 <= y < Ny:
            results[i] = R_scalar[x, y, z]

    return results

def safe_sample_ricci_angular_spread(R_scalar, radius=15, num_angles=360):
    thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    R_samples = sample_ricci_angular_spread(R_scalar, radius, thetas)
    return thetas, R_samples


def compute_stress_tensor_components(psi1, psi2, psi3, dx):
    # Gradients for each channel (returns tuple of arrays)
    grad1 = np.gradient(psi1, dx)
    grad2 = np.gradient(psi2, dx)
    grad3 = np.gradient(psi3, dx)
    # gradX = (gradX_x, gradX_y, gradX_z), each (Nx, Ny, Nz)

    # T_ab = sum over channels: Re[grad_a psi^* * grad_b psi]
    # We'll sum across all channels for each component

    T_xx = (
        (np.conj(grad1[0]) * grad1[0]).real +
        (np.conj(grad2[0]) * grad2[0]).real +
        (np.conj(grad3[0]) * grad3[0]).real
    )
    T_yy = (
        (np.conj(grad1[1]) * grad1[1]).real +
        (np.conj(grad2[1]) * grad2[1]).real +
        (np.conj(grad3[1]) * grad3[1]).real
    )
    T_zz = (
        (np.conj(grad1[2]) * grad1[2]).real +
        (np.conj(grad2[2]) * grad2[2]).real +
        (np.conj(grad3[2]) * grad3[2]).real
    )
    T_xy = (
        (np.conj(grad1[0]) * grad1[1]).real +
        (np.conj(grad2[0]) * grad2[1]).real +
        (np.conj(grad3[0]) * grad3[1]).real
    )
    T_xz = (
        (np.conj(grad1[0]) * grad1[2]).real +
        (np.conj(grad2[0]) * grad2[2]).real +
        (np.conj(grad3[0]) * grad3[2]).real
    )
    T_yz = (
        (np.conj(grad1[1]) * grad1[2]).real +
        (np.conj(grad2[1]) * grad2[2]).real +
        (np.conj(grad3[1]) * grad3[2]).real
    )
    return T_xx, T_yy, T_zz, T_xy, T_xz, T_yz