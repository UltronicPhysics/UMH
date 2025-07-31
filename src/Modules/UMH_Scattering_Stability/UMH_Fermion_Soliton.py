"""
UMH_Scattering_Stability_Fermion.py

Author: Andrew Dodge
Date: June 2025

Description:
UMH Scattering Stability Fermion.

Parameters:
- OUTPUT_FOLDER, LATTICE_SIZE, TIMESTEPS, DT, DAMPING, etc.

Inputs:
- None

Output:
- Produces
"""
import numpy as np
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


try:
    from ..UMH_Scattering_Stability.lattice import Lattice
    from ..UMH_Scattering_Stability.soliton import create_soliton
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_Scattering_Stability.lattice import Lattice
    from UMH_Scattering_Stability.soliton import create_soliton


def get_default_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        #All Settings.
        "LATTICE_SIZE": 300, #500,300
        "TIME_STEPS": 300, #500, 200

        "LATTICE_SPACING":1.0, #Grid Spacing, Planck equivalent, distance in UMH in Lattice.
    
        "MEDIUM_DENSITY":1.0, # Medium Tension (normalized or in units)
        "MEDIUM_PRESSURE":1.0, # Medium Pressure (normalized or in units)
        
        "PML_THICKNESS":24, #was 20, go to 20 under 500 grid size.  Thickness of PML layer at boundary. 30
        "PML_ABORPTION":0.18, #0.15 How impactful is the PML layers, adjust for reflection at boundary. 0.15
        "PML_POWER": 3,



        "DPI":300, #PNG Resolution.

        "DTYPE":np.float64, #Precision.

        "NUMBER_SNAPSHOTS":10,

        "OUTPUT_FOLDER": os.path.join(base, "Output")
    }

def zoom_slice_for_display(slice_, zoom_factor=4):
    return zoom(slice_, zoom=zoom_factor, order=1)  # Linear interpolation

def center_crop(slice_, crop_size=128):
    cx, cy = np.array(slice_.shape) // 2
    return slice_[cx - crop_size//2: cx + crop_size//2, cy - crop_size//2: cy + crop_size//2]

def crop_field(field, thickness):
    return field[thickness:-thickness, thickness:-thickness, thickness:-thickness]


def fermion_soliton_test(config_overrides=None):
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    size = config["LATTICE_SIZE"] # Memory-safe grid size.
    steps = config["TIME_STEPS"]

    dx=config["LATTICE_SPACING"]
    
    Tu=config["MEDIUM_DENSITY"]
    rho_u=config["MEDIUM_PRESSURE"]

    pml_thickness = config["PML_THICKNESS"]
    pml_absorption = config["PML_ABORPTION"]
    pml_power = config["PML_POWER"]
    

    dtype=config["DTYPE"]
    
    dpi=config["DPI"]

    num_snapshots = config["NUMBER_SNAPSHOTS"] # Gaussian Blend Wave Front.

    outdir = config["OUTPUT_FOLDER"]

    title="UMH Scattering Stability (Fermion)"
    file_root="UMH_Scattering_Stability"
    file_sub="UMH_Fermion_Soliton"
    file_hdr="UMH_Scattering_Stability_Fermion"
  
    print(f"✅ Starting Test: {title} Validation.")

    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_root)
    os.makedirs(outdir, exist_ok=True)
    outdir=os.path.join(outdir, file_sub)
    os.makedirs(outdir, exist_ok=True)
    file_path=os.path.join(outdir, file_hdr)

    print(f"{title}: Files Will be Saved to {outdir}.")


    lattice = Lattice(size=size, tension=10.0, density=1.0, damping=0.005)

    center1 = (size//2 - 10, size//2, size//2)
    center2 = (size//2 + 10, size//2, size//2)

    create_soliton(lattice, center=center1, radius=8, amplitude=1.0, phase=0.0)
    create_soliton(lattice, center=center2, radius=8, amplitude=1.0, phase=np.pi)

    dt = 0.01

    images_fs=[]

    global_min = float('inf')
    global_max = float('-inf')

    for step in range(steps):
        lattice.step(dt)
        if step % 10 == 0:
            strain = lattice.compute_strain_magnitude()
            normt = np.abs(lattice.u)  # still 3D

            # Use center Z-slice
            z_slice = normt.shape[2] // 2
            slice_2d = normt[:, :, z_slice]  # used for contrast stats

            # Optional: constrain min/max intensity
            global_min = 0.0
            global_max = 1.0

            # Physical zoom around center (XY plane at fixed Z)
            center = size // 2
            zoom_radius = 32
            xmin = center - zoom_radius
            xmax = center + zoom_radius
            zoom_slice = strain[xmin:xmax, xmin:xmax, z_slice]

            zoom_slice_smooth=gaussian_filter(zoom_slice, sigma=0.5)

            print(f"{title}, step:{step}, vmin={global_min}, vmax={global_max}")

            plt.imshow(zoom_slice_smooth, cmap='plasma', origin='lower',interpolation='nearest',vmin=global_min, vmax=global_max)
            plt.title(f"{title}: Step {step}")
            plt.colorbar()
            filename = f"{file_path}_{step}.png"
            plt.savefig(filename)
            plt.close()

            images_fs.append(imageio.imread(filename))



    imageio.mimsave(f"{file_path}_Evolution.gif", images_fs, fps=2)


    print(f"✅ Finished Test: {title} Validated.")



if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config = json.load(f)
    fermion_soliton_test()