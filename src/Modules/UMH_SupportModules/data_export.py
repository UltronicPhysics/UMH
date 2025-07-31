
import numpy as np
import pandas as pd
import os

def save_npy(name, array, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, name + ".npy"), array)

def save_csv_slice(name, array, axis='z', outdir="output"):
    import pandas as pd
    import os

    os.makedirs(outdir, exist_ok=True)

    if len(array.shape) == 3:
        if axis == 'z':
            array = array[:, :, array.shape[2] // 2]
        elif axis == 'y':
            array = array[:, array.shape[1] // 2, :]
        elif axis == 'x':
            array = array[array.shape[0] // 2, :, :]
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

    df = pd.DataFrame(array)
    df.to_csv(os.path.join(outdir, f"{name}_slice_{axis}.csv"), index=False)


def save_csv_3d(name, array, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    nx, ny, nz = array.shape
    flat = [(x, y, z, array[x, y, z]) for x in range(nx) for y in range(ny) for z in range(nz)]
    df = pd.DataFrame(flat, columns=["X", "Y", "Z", "Value"])
    df.to_csv(os.path.join(outdir, name + "_3D.csv"), index=False)
