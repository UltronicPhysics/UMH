import numpy as np
from numba import njit, prange
from mpi4py import MPI

@njit(cache=True, parallel=True, fastmath=True)
def apply_su3_constraint(field):
    for i in prange(field.shape[1]):
        for j in prange(field.shape[2]):
            for k in prange(field.shape[3]):
                norm = 1e-12
                for c in range(field.shape[0]):
                    norm += field[c, i, j, k]**2
                norm = np.sqrt(norm)
                for c in range(field.shape[0]):
                    field[c, i, j, k] /= norm

def mpi_log_energy_deviation(lattice, initial_energy, step, rank):
    ke = np.sum(lattice.kinetic_energy())
    pe = np.sum(lattice.potential_energy())
    ne = np.sum(lattice.nonlinear_energy())
    total = ke + pe + ne
    deviation = abs(total - initial_energy) / (initial_energy + 1e-12)

    # Create MPI-aware log
    log_row = f"{rank},{step},{ke:.6f},{pe:.6f},{ne:.6f},{total:.6f},{deviation:.8e}\n"
    log_path = f"outputs/mpi_energy_deviation_rank{rank}.csv"

    with open(log_path, "a") as f:
        if step == 0 and rank == 0:
            f.write("Rank,Step,KE,PE,NE,Total,Deviation\n")
        f.write(log_row)

def calculate_local_z_bounds(grid_size, size, rank):
    base = grid_size // size
    remainder = grid_size % size
    if rank < remainder:
        z_start = rank * (base + 1)
        z_end = z_start + (base + 1)
    else:
        z_start = rank * base + remainder
        z_end = z_start + base
    return z_start, z_end
