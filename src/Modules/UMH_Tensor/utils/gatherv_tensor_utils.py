from mpi4py import MPI
import numpy as np
import os
import matplotlib.pyplot as plt
from modules.utils.lattice import Lattice
from modules.utils.field_utils import compute_ricci_tensor
from modules.utils.tracking_advanced import calculate_local_z_bounds

def gatherv_tensor_to_rank0(tensor_local, grid_size, rank, size, name, slice_plot=True):
    comm = MPI.COMM_WORLD
    local_flat = tensor_local.flatten()
    local_size = local_flat.size
    sizes = comm.gather(local_size, root=0)

    if rank == 0:
        total_size = sum(sizes)
        flat_full = np.empty(total_size, dtype=np.float32)
        displs = [sum(sizes[:i]) for i in range(size)]
    else:
        flat_full = None
        displs = None

    comm.Gatherv(local_flat, (flat_full, sizes, displs, MPI.FLOAT), root=0)

    if rank == 0:
        full_tensor = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        offset = 0
        for i in range(size):
            z0, z1 = calculate_local_z_bounds(grid_size, size, i)
            d = z1 - z0
            flat_chunk = flat_full[offset:offset + grid_size * grid_size * d]
            full_tensor[:, :, z0:z1] = flat_chunk.reshape((grid_size, grid_size, d))
            offset += grid_size * grid_size * d
        np.save(f"outputs/{name}_full_tensor.npy", full_tensor)

        if slice_plot:
            z = full_tensor.shape[-1] // 2
            plt.imshow(full_tensor[:, :, z], cmap='viridis')
            plt.title(f"{name} Tensor Slice (z={z})")
            plt.colorbar()
            plt.savefig(f"outputs/{name}_slice.png")
            plt.close()
            print(f"[Saved] {name}_full_tensor.npy and slice plot.")