
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_slice(name, array, axis, outdir="output", cmap="viridis"):
    os.makedirs(outdir, exist_ok=True)
    if axis == 'xy':
        slice_ = array[:, :, array.shape[2] // 2]
    elif axis == 'xz':
        slice_ = array[:, array.shape[1] // 2, :]
    elif axis == 'yz':
        slice_ = array[array.shape[0] // 2, :, :]
    else:
        raise ValueError("Axis must be 'xy', 'xz', or 'yz'")

    plt.figure()
    plt.imshow(slice_, cmap=cmap)
    plt.title(f"{name} slice {axis.upper()}")
    plt.colorbar()
    plt.savefig(os.path.join(outdir, f"{name}_slice_{axis}.png"))
    plt.close()
