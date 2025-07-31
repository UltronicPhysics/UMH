import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
import os
import sys
from numba import njit, prange

# -------------------------------
# Project to HEALPix
# -------------------------------
@njit(parallel=True)
def project_to_healpix_numba_old(strain_field, theta, phi, cx, cy, cz, r_max,dtype=np.float32):
    npix = theta.shape[0]
    projected_map = np.zeros(npix, dtype=dtype)
    for i in prange(npix):
        th = theta[i]
        ph = phi[i]
        acc = 0.0
        count = 0
        for j in range(1, r_max):
            r = j
            x = int(cx + r * np.sin(th) * np.cos(ph))
            y = int(cy + r * np.sin(th) * np.sin(ph))
            z = int(cz + r * np.cos(th))
            if 0 <= x < strain_field.shape[0] and 0 <= y < strain_field.shape[1] and 0 <= z < strain_field.shape[2]:
                acc += strain_field[x, y, z]
                count += 1
        if count > 0:
            projected_map[i] = acc / count
    return projected_map

@njit(parallel=True)
def project_to_healpix_numba_rev2(strain_field, theta, phi, cx, cy, cz, r_max, dtype=np.float32):
    npix = theta.shape[0]
    projected_map = np.zeros(npix, dtype=dtype)
    
    for i in prange(npix):
        th = theta[i]
        ph = phi[i]
        acc = 0.0
        weight_sum = 0.0
        
        # Project only thin outer shell (last 2 units)
        r_start = max(int(r_max) - 2, 1)
        for r in range(r_start, int(r_max)):
            x = int(cx + r * np.sin(th) * np.cos(ph))
            y = int(cy + r * np.sin(th) * np.sin(ph))
            z = int(cz + r * np.cos(th))
            
            if 0 <= x < strain_field.shape[0] and 0 <= y < strain_field.shape[1] and 0 <= z < strain_field.shape[2]:
                weight = 1.0 / (r * r) if r > 0 else 1.0  # Apply 1/r² weighting
                acc += strain_field[x, y, z] * weight
                weight_sum += weight
        
        if weight_sum > 0.0:
            projected_map[i] = acc / weight_sum
        else:
            projected_map[i] = 0.0
    
    return projected_map


@njit(cache=True, parallel=True, fastmath=True)
def project_to_healpix_numba_rev3(strain_field, theta, phi, cx, cy, cz, r_min, r_max, apply_weighting=True, dtype=np.float32): # Project onto Healpix using thin spherical shell with optional 1/r² weighting.
    npix = theta.shape[0]
    projected_map = np.zeros(npix, dtype=dtype)

    r_min = max(r_min, 1)

    for i in prange(npix):
        th = theta[i]
        ph = phi[i]
        acc = 0.0
        weight_sum = 0.0

        for r in range(r_min, r_max):
            x = int(cx + r * np.sin(th) * np.cos(ph))
            y = int(cy + r * np.sin(th) * np.sin(ph))
            z = int(cz + r * np.cos(th))

            if 0 <= x < strain_field.shape[0] and 0 <= y < strain_field.shape[1] and 0 <= z < strain_field.shape[2]:
                weight = (1.0 / (r * r)) if apply_weighting else 1.0
                acc += strain_field[x, y, z] * weight
                weight_sum += weight

        projected_map[i] = acc / weight_sum if weight_sum > 0.0 else 0.0

    return projected_map


@njit(cache=True, parallel=True, fastmath=True)
def project_to_healpix_numba_rev4(strain_field, theta, phi, cx, cy, cz, r_min, r_max, apply_weighting=True, flip_sign=False, dtype=np.float32):
    npix = theta.shape[0]
    projected_map = np.zeros(npix, dtype=dtype)

    r_min = max(r_min, 1)

    for i in prange(npix):
        th = theta[i]
        ph = phi[i]
        acc = 0.0
        weight_sum = 0.0

        for r in range(r_min, r_max):
            x = int(cx + r * np.sin(th) * np.cos(ph))
            y = int(cy + r * np.sin(th) * np.sin(ph))
            z_shift = -r * np.cos(th) if flip_sign else r * np.cos(th)
            z = int(cz + z_shift)

            if 0 <= x < strain_field.shape[0] and 0 <= y < strain_field.shape[1] and 0 <= z < strain_field.shape[2]:
                weight = (1.0 / (r * r)) if apply_weighting else 1.0
                acc += strain_field[x, y, z] * weight
                weight_sum += weight

        projected_map[i] = acc / weight_sum if weight_sum > 0.0 else 0.0

    return projected_map




@njit(cache=True, parallel=True, fastmath=True)
def project_to_healpix_numba(strain_field, theta, phi, cx, cy, cz, r_min, r_max, apply_weighting=True, flip_sign=False, dtype=np.float32):
    npix = theta.shape[0]
    projected_map = np.zeros(npix, dtype=dtype)

    r_min = max(r_min, 1)

    for i in prange(npix):
        th = theta[i]
        ph = phi[i]
        acc = 0.0
        weight_sum = 0.0

        for r in range(r_min, r_max):
            sign = -1.0 if flip_sign else 1.0

            x = int(cx + sign * r * np.sin(th) * np.cos(ph))
            y = int(cy + sign * r * np.sin(th) * np.sin(ph))
            z = int(cz + sign * r * np.cos(th))

            if 0 <= x < strain_field.shape[0] and 0 <= y < strain_field.shape[1] and 0 <= z < strain_field.shape[2]:
                weight = (1.0 / (r * r)) if apply_weighting else 1.0
                #if r_max > r_min:
                #    window = 0.5 * (1 + np.cos(np.pi * (r - r_min) / (r_max - r_min)))  # Hann
                #    weight *= weight * window

                acc += strain_field[x, y, z] * weight
                weight_sum += weight

        projected_map[i] = acc / weight_sum if weight_sum > 0.0 else 0.0

    return projected_map


