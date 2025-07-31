import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def radial_profile_3d_njit(data, center, max_radius):
    zc, yc, xc = center
    nz, ny, nx = data.shape

    n_bins = max_radius + 1
    n_threads = 8  # or use `numba.config.NUMBA_NUM_THREADS` if available

    # Thread-local accumulators
    tbin_thread = np.zeros((n_threads, n_bins), dtype=np.float64)
    nr_thread = np.zeros((n_threads, n_bins), dtype=np.int64)

    for i in prange(nz):
        thread_id = i % n_threads
        for j in range(ny):
            for k in range(nx):
                r = int(np.sqrt((k - xc)**2 + (j - yc)**2 + (i - zc)**2))
                if r <= max_radius:
                    tbin_thread[thread_id, r] += data[i, j, k]
                    nr_thread[thread_id, r] += 1

    # Merge thread-local results
    tbin = np.zeros(n_bins, dtype=np.float64)
    nr = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for r in range(n_bins):
            tbin[r] += tbin_thread[t, r]
            nr[r] += nr_thread[t, r]

    # Compute final profile
    profile = np.empty(n_bins, dtype=np.float64)
    for r in range(n_bins):
        if nr[r] > 0:
            profile[r] = tbin[r] / nr[r]
        else:
            profile[r] = 0.0

    return profile

def get_max_radius(data_shape, center):
    zc, yc, xc = center
    nz, ny, nx = data_shape
    return int(np.sqrt(max(zc, nz - zc)**2 + max(yc, ny - yc)**2 + max(xc, nx - xc)**2))


#Original Below.


def radial_profile_3d(data, center):
    zc, yc, xc = center
    z, y, x = np.indices(data.shape)
    r = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2).astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)
